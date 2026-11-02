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


def reference_attention(q, k, v, *, fixed_bias=None, scale=None):
    """Official HF-style additive beta mask followed by one backend call.

    Native Miles supplies its TE reader instead; this dense mask is only the
    CPU/HF reference. Unequal lengths use prefix-offset causal visibility.
    """
    groups = q.shape[0] // k.shape[0]
    if fixed_bias is None:
        return _sdpa(q[None], k[None], v[None], is_causal=True,
                     scale=scale, enable_gqa=groups != 1)[0]
    fixed = k.shape[1] - q.shape[1]
    mask = q.new_zeros(1, q.shape[0], q.shape[1], k.shape[1])
    mask[..., :fixed] += fixed_bias.repeat_interleave(groups, 0)[None, :, None].to(q.dtype)
    future = torch.arange(k.shape[1], device=q.device)[None] > fixed + torch.arange(q.shape[1], device=q.device)[:, None]
    mask.masked_fill_(future[None, None], -torch.inf)
    return _sdpa(q[None], k[None], v[None], attn_mask=mask,
                 scale=scale, enable_gqa=groups != 1)[0]


def segment_attention(q, k, v, layout: ContextSegment, memory, *, rotary_dim, base, scale=None, attention=reference_attention):
    """Read one original segment; do not persist its duplicate problem or summary.

    Return output plus detached inputs for the later memory fit. No input tensor
    or incoming memory is modified, including during activation recomputation.
    """
    p = layout.problem_prefix
    offset = 0 if memory is None else memory.logical_end
    q = shift_rotary(q, offset=offset, rotary_dim=rotary_dim, base=base)
    k = shift_rotary(k, offset=offset, rotary_dim=rotary_dim, base=base)
    suffix_q, suffix_k = q[:, p:], k[:, p:]
    old_k = k[:, :0] if memory is None else memory.keys
    old_v = v[:, :0] if memory is None else memory.values
    old_bias = k.new_zeros(k.shape[0], 0, dtype=torch.float32) if memory is None else memory.bias
    if old_k.shape[1] == 0:
        output = attention(q, k, v, scale=scale)
    else:
        prefix = attention(q[:, :p], k[:, :p], v[:, :p], scale=scale)
        fixed_bias = torch.cat([old_bias.new_zeros(k.shape[0], p), old_bias], dim=1)
        suffix = attention(
            suffix_q, torch.cat([k[:, :p], old_k, suffix_k], dim=1),
            torch.cat([v[:, :p], old_v, v[:, p:]], dim=1),
            fixed_bias=fixed_bias, scale=scale,
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
def fit_memory(fit, *, ratio):
    """Use unchanged upstream selection/NNLS and C2 least squares.

    Repeated biased history needs an absolute stored beta: upstream selection
    solves with exp(logit + source_beta), so its log-weight is an increment.
    Apply that conversion before upstream C2 fitting and subsequent reads.
    """
    from agentic_critic.official_am import algorithm

    keys, values, biases, queries, logical_end = fit
    total = keys.shape[1]
    keep = max(1, min(total, int(ratio * total)))
    if total == 0 or keep == total:
        return Memory(keys, values, biases, logical_end)
    alg = algorithm()
    new_k, new_v, new_b = [], [], []
    groups = queries.shape[0] // keys.shape[0]
    with torch.autocast(device_type=keys.device.type, enabled=False):
        for h in range(keys.shape[0]):
            q = queries[h * groups : (h + 1) * groups].flatten(0, 1)
            k, v, source = keys[h], values[h], biases[h][None]
            c1, delta, selected = alg._select_keys_highest_attention(k, q, keep, source)
            beta = delta + biases[h, selected]
            c2 = alg._compute_C2_with_method(
                c1, beta, k, v, q, method=alg.c2_method, indices=selected,
                attention_bias=source, ridge_lambda=alg.c2_ridge_lambda,
                solver=alg.c2_solver, ridge_scale=alg.c2_ridge_scale,
            )
            new_k.append(c1)
            new_v.append(c2)
            new_b.append(beta)
    return Memory(torch.stack(new_k), torch.stack(new_v), torch.stack(new_b), logical_end)


def packed_attention(query, key, value, layouts, boundaries, *, ratio, rotary_dim, base, scale=None, attention=reference_attention):
    """Native THD core attention, with episode-local immutable memory per layer."""
    outputs, memory, episode = [], None, None
    for index, layout in enumerate(layouts):
        start, end = boundaries[index : index + 2]
        if layout.episode != episode:
            episode, memory = layout.episode, None
        q, k, v = [x[start:end].transpose(0, 1) for x in (query, key, value)]
        out, fit = segment_attention(q, k, v, layout, memory, rotary_dim=rotary_dim, base=base, scale=scale, attention=attention)
        outputs.append(out.transpose(0, 1))
        memory = None if fit is None else fit_memory(fit, ratio=ratio)
    # Packed padding is a separate independent sequence, never episode memory.
    end = boundaries[len(layouts)]
    if end < query.shape[0]:
        q, k, v = [x[end:].transpose(0, 1) for x in (query, key, value)]
        outputs.append(attention(q, k, v, scale=scale).transpose(0, 1))
    return torch.cat(outputs, dim=0).flatten(1)
