"""Actual TE rectangular attention and FLA state carry; requires the training image."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Native detached-memory kernels require CUDA")
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine.pytorch")


@pytest.fixture
def model_parallel(nccl_world):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(43)
    try:
        yield parallel_state.get_tensor_model_parallel_group()
    finally:
        parallel_state.destroy_model_parallel()


def _packed():
    from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens

    cu = torch.tensor([0, 13, 22, 24], dtype=torch.int32, device="cuda")
    return PackedSeqParamsWithHostCuSeqlens(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=13,
        max_seqlen_kv=13,
        cu_seqlens_host=(0, 13, 22, 24),
        critic_segment_lengths=((1, 4, 8), (3, 6)),
    )


@pytest.mark.parametrize("head_dim", [64, 256])
def test_native_te_bottom_right_memory_attention_values_gradients_and_recompute(monkeypatch, head_dim):
    # Match the training recipe: cuDNN backward rejects the one-token segment.
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    # The Qwen FA4 window fix applies to its native 256-wide heads. Keep the
    # generic 64-wide case on FA2; the pinned FA4 still mishandles that window.
    monkeypatch.setenv("NVTE_FLASH_ATTN_V4", "1" if head_dim == 256 else "0")
    _require_cuda()
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.qwen3_5_memory import DetachedMemoryDotProductAttention

    torch.manual_seed(41)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4 * head_dim,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=head_dim,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.bfloat16,
    )
    layer = DetachedMemoryDotProductAttention(config, 1, AttnMaskType.causal, "self").cuda()
    inputs = [
        torch.randn(24, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (4, 2, 2)
    ]
    packed = _packed()

    def forward(*values):
        return layer(*values, None, AttnMaskType.causal, packed_seq_params=packed)

    output = forward(*inputs)
    references = []
    for start, end in ((0, 13), (13, 22), (22, 24)):
        q, k, v = [value[start:end].transpose(0, 1).float() for value in inputs]
        references.append(
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(0, 1).flatten(1)
        )
    torch.testing.assert_close(output.float(), torch.cat(references), atol=0.025, rtol=0.025)
    grads = torch.autograd.grad(output[5:13].float().square().sum(), inputs)
    assert torch.count_nonzero(grads[1][:5]) == torch.count_nonzero(grads[2][:5]) == 0
    assert torch.count_nonzero(grads[1][5:13]) > 0

    plain = forward(*inputs)
    expected = torch.autograd.grad(plain.float().square().sum(), inputs)
    replay = checkpoint(forward, *inputs, use_reentrant=True)
    replay.float().square().sum().backward()
    torch.testing.assert_close(replay, plain)
    for value, grad in zip(inputs, expected, strict=True):
        torch.testing.assert_close(value.grad, grad, atol=0.025, rtol=0.025)


def test_native_fla_conv_and_recurrent_memory_state_gradients_and_recompute(model_parallel):
    _require_cuda()
    pytest.importorskip("fla.ops.gated_delta_rule")
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads
    from miles_plugins.models.detached_memory import memory_segments
    from miles_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

    torch.manual_seed(43)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    model = Qwen3_5GatedDeltaNet(
        config,
        heads=LinearAttnHeads(num_k_heads=1, num_v_heads=2, head_k_dim=32, head_v_dim=32),
        conv_kernel_size=4,
        norm_eps=1e-6,
        tp_group=model_parallel,
        backend="fla",
    )
    x = torch.randn(1, 24, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    packed = _packed()

    def forward(value):
        core = model(value, packed.cu_seqlens_q, segments=memory_segments(packed, value.shape[1]))
        return model.out_proj(core)

    memory = forward(x)
    ordinary = model.out_proj(model(x, packed.cu_seqlens_q))
    torch.testing.assert_close(memory, ordinary, atol=0.025, rtol=0.025)
    grad = torch.autograd.grad(memory[:, 5:13].float().square().sum(), x)[0]
    assert torch.count_nonzero(grad[:, :5]) == 0
    assert torch.count_nonzero(grad[:, 5:13]) > 0

    plain = forward(x)
    plain.float().square().sum().backward()
    expected_x = x.grad.clone()
    expected_parameters = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    x.grad = None
    replay = checkpoint(forward, x, use_reentrant=True)
    replay.float().square().sum().backward()
    torch.testing.assert_close(plain, replay)
    torch.testing.assert_close(x.grad, expected_x, atol=0.025, rtol=0.025)
    for parameter, grad in zip(model.parameters(), expected_parameters, strict=True):
        torch.testing.assert_close(parameter.grad, grad, atol=0.025, rtol=0.025)


@pytest.mark.parametrize("detached", [False, True], ids=["packed-actor", "detached-critic"])
def test_sm100_fa4_causal_hd256_against_reference(monkeypatch, detached):
    monkeypatch.setenv("NVTE_FLASH_ATTN_V4", "1")
    _require_cuda()
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 FA4 regression")

    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from miles_plugins.models.qwen3_5_memory import DetachedMemoryDotProductAttention, QwenCausalDotProductAttention

    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    torch.manual_seed(45)
    config = TransformerConfig(num_layers=1, hidden_size=1024, num_attention_heads=4, num_query_groups=2,
                               kv_channels=256, attention_dropout=0.0, params_dtype=torch.bfloat16)
    cls = DetachedMemoryDotProductAttention if detached else QwenCausalDotProductAttention
    layer = cls(config, 1, AttnMaskType.causal, "self").cuda()
    versions = []
    layer.flash_attention.register_forward_pre_hook(
        lambda module, args, kwargs: versions.append(kwargs["flash_attention_backend"].major), with_kwargs=True
    )
    packed = _packed()
    if not detached:
        packed.critic_segment_lengths = None
    inputs = [torch.randn(24, h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (4, 2, 2)]
    output = layer(*inputs, None, AttnMaskType.causal, packed_seq_params=packed).reshape(24, 4, 256)
    reference_inputs = [x.detach().float().requires_grad_() for x in inputs]
    refs = []
    for start, end in ((0, 13), (13, 22), (22, 24)):
        q, k, v = (x[start:end] for x in reference_inputs)
        # Take the loss only on the last segment of each real episode.
        if detached:
            prefix = {0: 5, 13: 3, 22: 0}[start]
            k = torch.cat((k[:prefix].detach(), k[prefix:]))
            v = torch.cat((v[:prefix].detach(), v[prefix:]))
        scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, dim=1)) / 16
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(1), -torch.inf)
        refs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), v.repeat_interleave(2, dim=1)))
    reference = torch.cat(refs)
    torch.testing.assert_close(output.float(), reference, rtol=0.025, atol=0.025)
    probe = torch.randn_like(output)
    if detached:
        probe[:5] = 0
        probe[13:16] = 0
    actual = torch.autograd.grad(output, inputs, probe)
    expected = torch.autograd.grad(reference, reference_inputs, probe.float())
    for grad, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(grad.float(), target, rtol=0.04, atol=0.04)
    if detached:
        # TE can dispatch short rectangular segments to another flash backend.
        assert 4 in versions
    else:
        assert versions == [4]
