"""Segment boundaries for a packed, detached-memory critic forward.

Only the current forward owns these states. The model's usual activation
checkpointing can therefore call the same forward again without a cache reset.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MemorySegment:
    start: int
    end: int
    episode_start: int


def memory_segments(packed_seq_params, total_tokens: int) -> tuple[MemorySegment, ...] | None:
    """Resolve real episode segments and isolate any trailing THD padding."""
    lengths = getattr(packed_seq_params, "critic_segment_lengths", None)
    if lengths is None:
        return None
    if packed_seq_params.qkv_format != "thd":
        raise ValueError("Detached critic memory requires THD packed sequences")
    boundaries = packed_seq_params.cu_seqlens_host
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != total_tokens:
        raise ValueError("Detached critic memory boundaries do not cover the packed tokens")
    if len(boundaries) - len(lengths) not in (1, 2):
        raise ValueError("Detached critic memory needs one segment-length list per real sequence")
    segments = []
    for i, own in enumerate(lengths):
        start = boundaries[i]
        if not own or any(type(length) is not int or length <= 0 for length in own):
            raise ValueError("Detached critic segment lengths must be positive integers")
        if sum(own) != boundaries[i + 1] - start:
            raise ValueError("Detached critic segment lengths do not cover their episode")
        for length in own:
            segments.append(MemorySegment(start, start + length, boundaries[i]))
            start += length
    if len(boundaries) == len(lengths) + 2:
        start, end = boundaries[-2:]
        if end <= start:
            raise ValueError("Detached critic padding sequence must be nonempty")
        segments.append(MemorySegment(start, end, start))
    return tuple(segments)


def detached_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segments: tuple[MemorySegment, ...],
    attend: Callable,
) -> torch.Tensor:
    """Call an existing bottom-right causal kernel with detached prefix KV.

    Q/K already carry the original episode positions from native RoPE. Only
    K/V belonging to earlier segments are detached; this segment's K/V retain
    their gradients. Head dimensions remain local to the native TP rank.
    """
    outputs = []
    for segment in segments:
        start, end, episode_start = segment.start, segment.end, segment.episode_start
        own_key, own_value = key[start:end], value[start:end]
        if start > episode_start:
            own_key = torch.cat((key[episode_start:start].detach(), own_key), dim=0)
            own_value = torch.cat((value[episode_start:start].detach(), own_value), dim=0)
        outputs.append(attend(query[start:end], own_key, own_value))
    return torch.cat(outputs, dim=0)


def detached_stateful_segments(segments: tuple[MemorySegment, ...], step: Callable) -> torch.Tensor:
    """Run an existing conv/recurrent kernel and detach its state at each boundary."""
    outputs, state = [], None
    episode_start = None
    for segment in segments:
        if segment.episode_start != episode_start:
            state = None
            episode_start = segment.episode_start
        output, state = step(segment.start, segment.end, state)
        outputs.append(output)
        state = tuple(tensor.detach() for tensor in state)
    return torch.cat(outputs, dim=1)
