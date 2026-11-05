"""Real torch gradients with CPU reference kernels at the optional TE/FLA boundary.

The production layer classes are loaded from their AST because the CPU test
image has no Megatron/TE/FLA. GPU tests exercise those actual kernels separately.
"""

import ast
import copy
from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from miles_plugins.models.detached_memory import (
    MemorySegment,
    detached_attention,
    detached_stateful_segments,
    memory_segments,
)

ROOT = Path(__file__).resolve().parents[3]


def _load_class(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def _packed(lengths=((2, 3), (3,)), *, padding=1):
    cu = [0]
    for own in lengths:
        cu.append(cu[-1] + sum(own))
    if padding:
        cu.append(cu[-1] + padding)
    return SimpleNamespace(
        critic_segment_lengths=lengths,
        cu_seqlens_host=tuple(cu),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        qkv_format="thd",
    )


def _attention(q, k, v):
    # Explicit bottom-right alignment is also an oracle for the TE mask choice.
    mask = torch.arange(k.shape[0])[None, :] <= torch.arange(q.shape[0])[:, None] + k.shape[0] - q.shape[0]
    return F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, enable_gqa=True
    ).transpose(0, 1)


class _Mask(Enum):
    causal = 1
    causal_bottom_right = 2


class _NativeTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(context_parallel_size=1)
        self.calls = []

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        *,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs
    ):
        self.calls.append((query.shape, key.shape, attn_mask_type, packed_seq_params, kwargs))
        if packed_seq_params is not None:
            cu = packed_seq_params.cu_seqlens_host
            outputs = [_attention(query[a:b], key[a:b], value[a:b]) for a, b in zip(cu[:-1], cu[1:], strict=True)]
            return torch.cat(outputs).flatten(1)
        assert query.ndim == key.ndim == value.ndim == 4
        assert attn_mask_type == _Mask.causal_bottom_right
        return _attention(query[:, 0], key[:, 0], value[:, 0]).flatten(1).unsqueeze(1)


def _memory_attention():
    cls = _load_class(
        "miles_plugins/models/qwen3_5_memory.py",
        "DetachedMemoryDotProductAttention",
        dict(
            TEDotProductAttention=_NativeTE,
            AttnMaskType=_Mask,
            memory_segments=memory_segments,
            detached_attention=detached_attention,
        ),
    )
    return cls()


def test_attention_forward_preserves_prefix_values_and_stops_history_gradients():
    torch.manual_seed(17)
    q = torch.randn(9, 4, 3, dtype=torch.double, requires_grad=True)
    k = torch.randn(9, 2, 3, dtype=torch.double, requires_grad=True)
    v = torch.randn(9, 2, 3, dtype=torch.double, requires_grad=True)
    packed = _packed()
    layer = _memory_attention()
    output = layer(q, k, v, None, _Mask.causal, packed_seq_params=packed)
    reference = torch.cat([_attention(q[a:b], k[a:b], v[a:b]) for a, b in ((0, 5), (5, 8), (8, 9))])
    torch.testing.assert_close(output, reference.flatten(1))
    output[2:5].square().sum().backward()
    assert torch.count_nonzero(k.grad[:2]) == torch.count_nonzero(v.grad[:2]) == 0
    assert torch.count_nonzero(k.grad[2:5]) > 0 and torch.count_nonzero(v.grad[2:5]) > 0
    assert torch.count_nonzero(q.grad[2:5]) > 0
    # Each local TP head partition stays local; the prefix only changes token length.
    assert [call[0][:3] for call in layer.calls] == [(2, 1, 4), (3, 1, 4), (3, 1, 4), (1, 1, 4)]
    assert [call[1][0] for call in layer.calls] == [2, 5, 3, 1]


def test_attention_without_metadata_delegates_native_arguments_unchanged():
    layer = _memory_attention()
    packed = _packed()
    packed.critic_segment_lengths = None
    q = torch.randn(9, 2, 3)
    output = layer(q, q, q, None, _Mask.causal, packed_seq_params=packed, num_splits=1)
    assert len(layer.calls) == 1
    assert layer.calls[0][2:] == (_Mask.causal, packed, {"num_splits": 1})
    assert output.shape == (9, 6)


@pytest.mark.parametrize("reentrant", [False, True])
def test_attention_checkpoint_replays_with_no_persistent_memory(reentrant):
    torch.manual_seed(20)
    layer = _memory_attention()
    packed = _packed()
    inputs = [torch.randn(9, 2, 3, requires_grad=True) for _ in range(3)]

    def forward(*inputs):
        return layer(*inputs, None, _Mask.causal, packed_seq_params=packed)

    plain = forward(*inputs)
    expected = torch.autograd.grad(plain.square().sum(), inputs)
    replay = checkpoint(forward, *inputs, use_reentrant=reentrant)
    replay.square().sum().backward()
    torch.testing.assert_close(replay, plain)
    for value, grad in zip(inputs, expected, strict=True):
        torch.testing.assert_close(value.grad, grad)
    torch.testing.assert_close(forward(*inputs), plain)


def _conv_reference(x, weight, bias=None, initial_state=None, output_final_state=False, activation="silu", **kwargs):
    channels, width = weight.shape
    previous = x.new_zeros(1, channels, width) if initial_state is None else initial_state
    history = torch.cat((previous.transpose(1, 2), x), dim=1)
    extended = history[:, -(x.shape[1] + width - 1) :]
    result = F.conv1d(extended.transpose(1, 2), weight[:, None], bias, groups=channels).transpose(1, 2)
    if activation is not None:
        result = F.silu(result)
    return result, history[:, -width:].transpose(1, 2) if output_final_state else None


def _delta_reference(q, k, v, g, beta, initial_state=None, output_final_state=False, **kwargs):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    state = q.new_zeros(1, q.shape[2], q.shape[3], v.shape[3]) if initial_state is None else initial_state
    outputs = []
    for t in range(q.shape[1]):
        state = state * g[:, t].exp()[..., None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        update = beta[:, t, :, None] * (v[:, t] - prediction)
        state = state + k[:, t, :, :, None] * update[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state) / q.shape[-1] ** 0.5)
    return torch.stack(outputs, dim=1), state if output_final_state else None


def _varlen_kernel(kernel, *inputs, cu_seqlens=None, **kwargs):
    if cu_seqlens is None:
        return kernel(*inputs, **kwargs)
    cu = cu_seqlens.tolist()
    outputs = []
    for a, b in zip(cu[:-1], cu[1:], strict=True):
        sliced = {key: value[:, a:b] if key in ("g", "beta") else value for key, value in kwargs.items()}
        result, _ = kernel(*(value[:, a:b] for value in inputs), **sliced)
        outputs.append(result)
    return torch.cat(outputs, dim=1), None


class _Conv(nn.Conv1d):
    activation = "silu"
    backend = "triton"

    def forward(self, x, cu_seqlens=None, **kwargs):
        return _varlen_kernel(_conv_reference, x, cu_seqlens=cu_seqlens, weight=self.weight.squeeze(1), bias=self.bias)


class _Norm(nn.Module):
    def forward(self, x, z):
        return F.rms_norm(x, (x.shape[-1],)) * F.silu(z)


def _gdn():
    cls = _load_class(
        "miles_plugins/models/qwen3_5.py",
        "Qwen3_5GatedDeltaNet",
        dict(
            torch=torch,
            nn=nn,
            F=F,
            memory_segments=memory_segments,
            detached_stateful_segments=detached_stateful_segments,
            causal_conv1d=_conv_reference,
            build_gdn_cp_context=lambda *args: None,
        ),
    )
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.key_dim, model.value_dim = 2, 4
    model.head_k_dim = model.head_v_dim = 2
    model.num_k_heads, model.num_v_heads = 1, 2
    model.gdn_backend = "fla"
    model.in_proj_qkv, model.in_proj_z = nn.Linear(4, 8, bias=False), nn.Linear(4, 4, bias=False)
    model.in_proj_a, model.in_proj_b = nn.Linear(4, 2, bias=False), nn.Linear(4, 2, bias=False)
    model.conv1d = _Conv(8, 8, 3, groups=8, bias=False)
    model.A_log, model.dt_bias = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    model.out_proj, model.norm = nn.Linear(4, 4, bias=False), _Norm()
    model.chunk_gated_delta_rule = lambda *args, **kwargs: _varlen_kernel(_delta_reference, *args, **kwargs)
    return model.double()


def test_gdn_preserves_conv_and_recurrent_values_but_detaches_both_states():
    torch.manual_seed(23)
    model, packed = _gdn(), _packed(((1, 1, 3), (3,)))
    x = torch.randn(1, 9, 4, dtype=torch.double, requires_grad=True)
    memory = model(x, packed.cu_seqlens_q, packed)
    native = model(x, packed.cu_seqlens_q)
    torch.testing.assert_close(memory, native)
    grad = torch.autograd.grad(memory[:, 2:5].square().sum(), x, retain_graph=True)[0]
    assert torch.count_nonzero(grad[:, :2]) == 0
    assert torch.count_nonzero(grad[:, 2:5]) > 0
    native_grad = torch.autograd.grad(native[:, 2:5].square().sum(), x)[0]
    assert torch.count_nonzero(native_grad[:, :2]) > 0
    # Changing a previous episode must never affect the next episode or padding.
    changed = x.detach().clone()
    changed[:, :5] += 3
    torch.testing.assert_close(model(changed, packed.cu_seqlens_q, packed)[:, 5:], memory[:, 5:])


@pytest.mark.parametrize("reentrant", [False, True])
def test_gdn_checkpoint_preserves_parameter_and_input_gradients(reentrant):
    torch.manual_seed(29)
    model, packed = _gdn(), _packed(((1, 1, 3), (3,)))
    x = torch.randn(1, 9, 4, dtype=torch.double, requires_grad=True)

    def forward(value):
        return model(value, packed.cu_seqlens_q, packed)

    plain = forward(x)
    plain.square().sum().backward()
    expected_x = x.grad.clone()
    expected_parameters = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    x.grad = None
    replay = checkpoint(forward, x, use_reentrant=reentrant)
    replay.square().sum().backward()
    torch.testing.assert_close(plain, replay)
    torch.testing.assert_close(x.grad, expected_x)
    for parameter, grad in zip(model.parameters(), expected_parameters, strict=True):
        torch.testing.assert_close(parameter.grad, grad)


def test_segment_metadata_isolates_padding_and_rejects_incomplete_coverage():
    assert memory_segments(_packed(), 9) == (
        MemorySegment(0, 2, 0),
        MemorySegment(2, 5, 0),
        MemorySegment(5, 8, 5),
        MemorySegment(8, 9, 8),
    )
    bad = _packed()
    bad.critic_segment_lengths = ((2, 2), (3,))
    with pytest.raises(ValueError, match="cover their episode"):
        memory_segments(bad, 9)


def test_packed_metadata_ingress_freezes_segment_lengths_for_recompute():
    path = ROOT / "miles/backends/megatron_utils/parallel.py"
    names = {"PackedSeqParamsWithHostCuSeqlens", "get_packed_seq_params"}
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    base = make_dataclass(
        "PackedSeqParams",
        [(name, object) for name in ("cu_seqlens_q", "cu_seqlens_kv", "max_seqlen_q", "max_seqlen_kv", "qkv_format")],
    )
    namespace = dict(dataclass=dataclass, field=field, PackedSeqParams=base, torch=torch, Namespace=SimpleNamespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    source = [[2, 3], [3]]
    batch = dict(
        cu_seqlens=torch.tensor([0, 5, 8, 9]),
        cu_seqlens_host=(0, 5, 8, 9),
        max_seqlen=5,
        critic_segment_lengths=source,
    )
    packed = namespace["get_packed_seq_params"](batch, SimpleNamespace(qkv_format="thd"))
    source[0][0] = 100
    assert packed.critic_segment_lengths == ((2, 3), (3,))
    assert memory_segments(packed, 9)[-1] == MemorySegment(8, 9, 8)
    del batch["critic_segment_lengths"]
    assert namespace["get_packed_seq_params"](batch, SimpleNamespace(qkv_format="thd")).critic_segment_lengths is None


@pytest.mark.parametrize("mode", ["shared", "actor-summary", "full_detach"])
def test_qwen_spec_only_installs_memory_core_for_full_detach(mode):
    path = ROOT / "miles_plugins/models/qwen3_5.py"
    node = next(node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "get_qwen3_5_spec")
    native_core, memory_core, linear_attention = object(), object(), object()
    seed = SimpleNamespace(
        layer_specs=[
            SimpleNamespace(
                submodules=SimpleNamespace(
                    self_attention=SimpleNamespace(submodules=SimpleNamespace(core_attention=native_core))
                )
            )
            for _ in range(2)
        ]
    )
    module_spec = make_dataclass("ModuleSpec", [("module", object), ("params", object, field(default=None))])
    namespace = dict(
        copy=copy,
        ModuleSpec=module_spec,
        get_gpt_decoder_block_spec=lambda *_args, **_kwargs: seed,
        get_num_layers_to_build=lambda *_args, **_kwargs: 2,
        get_transformer_layer_offset=lambda *_args, **_kwargs: 0,
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
                layer_types=["linear_attention", "full_attention"]
            )
        ),
        _get_text_config=lambda value: value,
        Attention=linear_attention,
        DetachedMemoryDotProductAttention=memory_core,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(num_layers=2, pipeline_model_parallel_layout=None)
    args = SimpleNamespace(num_experts=0, hf_checkpoint="unused", critic_context_mode=mode)
    result = namespace["get_qwen3_5_spec"](args, config, None)
    assert result.layer_specs[0].submodules.self_attention.module is linear_attention
    core = result.layer_specs[1].submodules.self_attention.submodules.core_attention
    assert (core is memory_core) == (mode == "full_detach")
