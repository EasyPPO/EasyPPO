"""Real Megatron/TE beta ingress check; run in the production CUDA environment."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires production CUDA Megatron/TE stack")
def test_native_te_memory_bias_forward_and_backward(tmp_path):
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    from agentic_critic.context_memory import reference_attention
    from miles_plugins.models.critic_memory_attention import CriticMemoryAttention

    assert not dist.is_initialized(), "Run this standalone smoke test outside an active trainer"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=(tmp_path / "process_group").as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=4, hidden_size=4096, num_attention_heads=16, num_query_groups=4,
            kv_channels=256, params_dtype=torch.bfloat16, attention_dropout=0,
            hidden_dropout=0, sequence_parallel=False,
        )
        # Exercise the real native spec, whose core is a class in pinned MCore.
        from types import SimpleNamespace
        from transformers import Qwen3_5TextConfig
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
        from megatron.core.transformer.spec_utils import ModuleSpec
        from miles_plugins.models.qwen3_5 import get_qwen3_5_spec

        Qwen3_5TextConfig(num_hidden_layers=4).save_pretrained(tmp_path)
        args = SimpleNamespace(
            num_experts=None, hf_checkpoint=str(tmp_path),
            critic_context_mode="attention_matching", kv_channels=256,
            rotary_percent=0.25, rotary_base=10000000,
        )
        spec = get_qwen3_5_spec(args, config, None)
        wrapped = spec.layer_specs[3].submodules.self_attention.submodules.core_attention
        assert isinstance(wrapped, ModuleSpec)
        assert wrapped.module is CriticMemoryAttention
        assert wrapped.params["memory_rotary_dim"] == 64
        args.critic_context_mode = "shared"
        shared = get_qwen3_5_spec(args, config, None)
        original = shared.layer_specs[3].submodules.self_attention.submodules.core_attention
        assert original is TEDotProductAttention
        assert not hasattr(TEDotProductAttention, "module")

        core = CriticMemoryAttention(
            config, layer_number=4, attn_mask_type=AttnMaskType.causal,
            attention_type="self", memory_rotary_dim=64, memory_rotary_base=10000000,
        ).cuda()
        torch.manual_seed(43)
        q = torch.randn(16, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(4, 96, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(4, 96, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        beta = torch.randn(4, 32, device="cuda")
        actual = core._read_native(q, k, v, fixed_bias=beta)
        expected = reference_attention(q, k, v, fixed_bias=beta)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
        grad = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (q, k, v), grad)
        expected_grad = torch.autograd.grad(expected, (q, k, v), grad)
        for a, b in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, b, atol=0.03, rtol=0.03)
        # eval selects a different native TE backend on sm86; exercise it too.
        core.eval()
        with torch.no_grad():
            evaluated = core._read_native(q, k, v, fixed_bias=beta)
        torch.testing.assert_close(evaluated, expected.detach(), atol=0.03, rtol=0.03)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
