"""Stateless full-attention memory for native Miles packed actor segments.

Inputs are post-RoPE Q/K, using actor-summary's segment-local positions. Each
whole new segment gets one common logical offset, preserving its internal RoPE
distances. Its fresh problem replaces the previous problem block physically;
compressed history keys retain their earlier logical positions.
Linear attention is outside this module and resets at each packed segment.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from agentic_critic.context_layout import ContextSegment


@dataclass(frozen=True)
class Memory:
    keys: torch.Tensor
    values: torch.Tensor
    bias: torch.Tensor
    logical_end: int


def shift_rotary(x, *, offset, rotary_dim, base):
    """Compose split-half RoPE with a constant offset, preserving nonrotary dims."""
    if not offset or not rotary_dim:
        return x
    frequencies = base ** (-torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    angles = torch.cat([frequencies, frequencies]) * offset
    rotated = x[..., :rotary_dim].float()
    left, right = rotated.chunk(2, dim=-1)
    result = rotated * angles.cos() + torch.cat([-right, left], dim=-1) * angles.sin()
    return torch.cat([result.to(x.dtype), x[..., rotary_dim:]], dim=-1)


def _sdpa(q, k, v, **kwargs):
    # cuDNN cached/chunked reads were incorrect in the production GB200 stack.
    # Keep this policy inside the function so backward recomputation sees it too.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _read_chunk(q, k, v, bias, memory_slots, first, stop, scale):
    groups = q.shape[0] // k.shape[0]
    position = torch.arange(first, stop, device=q.device) + memory_slots
    mask = q.new_zeros(1, q.shape[0], stop - first, k.shape[1])
    mask[..., :memory_slots] = bias.repeat_interleave(groups, 0)[None, :, None].to(q.dtype)
    mask.masked_fill_((torch.arange(k.shape[1], device=q.device)[None] > position[:, None])[None, None], -torch.inf)
    return _sdpa(
        q[None, :, first:stop],
        k[None],
        v[None],
        attn_mask=mask,
        scale=scale,
        enable_gqa=groups != 1,
    )[0]


def _read(q, k, v, *, memory_slots, bias, scale):
    """All history is visible; only the current segment suffix is causal."""
    outputs = []
    for first in range(0, q.shape[1], 128):
        stop = min(first + 128, q.shape[1])
        args = (q, k, v, bias, memory_slots, first, stop, scale)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
            out = checkpoint(_read_chunk, *args, use_reentrant=False)
        else:
            out = _read_chunk(*args)
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def segment_attention(q, k, v, layout: ContextSegment, memory, *, rotary_dim, base, scale=None):
    """Read one original segment; do not persist its duplicate problem or summary.

    Return output plus detached inputs for the later memory fit. No input tensor
    or incoming memory is modified, including during activation recomputation.
    """
    p = layout.problem_prefix
    offset = 0 if memory is None else memory.logical_end
    q = shift_rotary(q, offset=offset, rotary_dim=rotary_dim, base=base)
    k = shift_rotary(k, offset=offset, rotary_dim=rotary_dim, base=base)
    suffix_q, suffix_k = q[:, p:], k[:, p:]
    prefix = _sdpa(
        q[None, :, :p],
        k[None, :, :p],
        v[None, :, :p],
        is_causal=True,
        scale=scale,
        enable_gqa=q.shape[0] != k.shape[0],
    )[0]
    old_k = k[:, :0] if memory is None else memory.keys
    old_v = v[:, :0] if memory is None else memory.values
    old_bias = k.new_zeros(k.shape[0], 0, dtype=torch.float32) if memory is None else memory.bias
    fixed_bias = torch.cat([old_bias.new_zeros(k.shape[0], p), old_bias], dim=1)
    suffix = _read(
        suffix_q,
        torch.cat([k[:, :p], old_k, suffix_k], dim=1),
        torch.cat([v[:, :p], old_v, v[:, p:]], dim=1),
        memory_slots=p + old_k.shape[1],
        bias=fixed_bias,
        scale=scale,
    )
    output = torch.cat([prefix, suffix], dim=1)
    logical_end = offset + layout.length
    if layout.index + 1 == layout.count:
        return output, None
    a, b = layout.summary_span
    # Generated summary Q probes the source, but its full K/V is not retained.
    fit = (
        torch.cat([old_k, suffix_k[:, : a - p]], dim=1).detach(),
        torch.cat([old_v, v[:, p:a]], dim=1).detach(),
        torch.cat([old_bias, old_bias.new_zeros(k.shape[0], a - p)], dim=1),
        suffix_q[:, a - p : b - p].detach(),
        logical_end,
    )
    return output, fit


@torch.no_grad()
def fit_memory(fit, *, ratio, ridge=1e-4):
    """Fit only the history block, excluding the problem and current summary."""
    # Optional upstream package is needed only when an actual boundary is fitted.
    from agentic_critic.kv_compact import _algorithm, _ridge_value_fit

    keys, values, biases, queries, logical_end = fit
    total = keys.shape[1]
    if total == 0:
        return Memory(keys, values, biases, logical_end)
    keep = max(1, min(total, round(ratio * total)))
    if keep == total:
        return Memory(keys, values, biases, logical_end)
    alg = _algorithm()
    cuda = [keys.device.index] if keys.is_cuda else []
    new_k, new_v, new_b = [], [], []
    groups = queries.shape[0] // keys.shape[0]
    with torch.random.fork_rng(devices=cuda), torch.autocast(device_type=keys.device.type, enabled=False):
        torch.manual_seed(0)
        for h in range(keys.shape[0]):
            q = queries[h * groups : (h + 1) * groups].flatten(0, 1).float()
            k, v, bias = keys[h].float(), values[h].float(), biases[h][None]
            c1, beta, selected = alg._select_keys_highest_attention(k, q, keep, bias)
            beta = beta + biases[h, selected]
            c2 = _ridge_value_fit(k, v, q, c1, beta, selected, bias, False, ridge)
            order = torch.tensor(selected, device=keys.device).argsort()
            new_k.append(c1[order].to(keys.dtype))
            new_v.append(c2[order].to(values.dtype))
            new_b.append(beta[order])
    return Memory(torch.stack(new_k), torch.stack(new_v), torch.stack(new_b), logical_end)


def packed_attention(query, key, value, layouts, boundaries, *, ratio, rotary_dim, base, scale=None):
    """Native THD core attention, with episode-local immutable memory per layer."""
    outputs, memory, episode = [], None, None
    for index, layout in enumerate(layouts):
        start, end = boundaries[index : index + 2]
        if layout.episode != episode:
            episode, memory = layout.episode, None
        q, k, v = [x[start:end].transpose(0, 1) for x in (query, key, value)]
        out, fit = segment_attention(q, k, v, layout, memory, rotary_dim=rotary_dim, base=base, scale=scale)
        outputs.append(out.transpose(0, 1))
        memory = None if fit is None else fit_memory(fit, ratio=ratio)
    # Packed padding is a separate independent sequence, never episode memory.
    end = boundaries[len(layouts)]
    if end < query.shape[0]:
        q, k, v = [x[end:].transpose(0, 1)[None] for x in (query, key, value)]
        pad = _sdpa(q, k, v, is_causal=True, scale=scale, enable_gqa=q.shape[1] != k.shape[1])
        outputs.append(pad[0].transpose(0, 1))
    return torch.cat(outputs, dim=0).flatten(1)
