"""Explicit Qwen3.5 kernels; missing FLA is an error, never a silent slow path."""

import importlib.metadata

import torch
from torch.nn import functional as F


def depthwise_conv(x, weight, bias=None, activation=None, **kwargs):
    """Chosen PyTorch depthwise convolution (also usable without a CUDA toolkit)."""
    if kwargs.get("seq_idx") is not None or kwargs.get("cu_seqlens") is not None:
        raise ValueError("This convolution backend supports one unpadded episode, not packed sequences")
    y = F.conv1d(x.to(weight.dtype), weight[:, None], bias, padding=weight.shape[-1] - 1, groups=x.shape[1])
    y = y[..., : x.shape[-1]]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"Unsupported convolution activation: {activation}")
    return y.to(x.dtype)


def depthwise_update(x, state, weight, bias=None, activation=None, **kwargs):
    if x.ndim != 3:
        raise ValueError("Expected cached convolution input [batch, channels, tokens]")
    joined = torch.cat((state, x), dim=-1).to(weight.dtype)
    state.copy_(joined[..., -state.shape[-1] :])
    y = F.conv1d(joined, weight[:, None], bias, groups=x.shape[1])[..., -x.shape[-1] :]
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f"Unsupported convolution activation: {activation}")
    return y.to(x.dtype)


def configure_training_kernels():
    """Select FLA + torch convolution explicitly for both training and generation."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        if not hasattr(modeling, name):
            raise RuntimeError(f"Unsupported Transformers Qwen3.5 API: missing {name}; use the pinned environment")
    modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    modeling.causal_conv1d_fn = depthwise_conv
    modeling.causal_conv1d_update = depthwise_update
    # Checkpoint recomputation happens during backward, after a forward-local
    # sdpa_kernel context has exited. Keep the same kernel enabled for both passes.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "linear_attention": "fla.ops.gated_delta_rule",
        "fla_version": importlib.metadata.version("flash-linear-attention"),
        "convolution": "torch.nn.functional.conv1d (explicit)",
        "dense_attention": "SDPA flash only (selected by forward context)",
    }
