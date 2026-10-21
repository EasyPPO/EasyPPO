"""Bounded-memory hidden states at causal value positions for both critic inputs."""

import torch

from .cache import assigning_cache_updates
from .critic_input import Segment

MODES = ("shared", "full")


def value_positions(seg: Segment, limit: int | None = None) -> list[int]:
    """Read the hidden state immediately before each supervised actor token."""
    start = seg.response_start
    assert start > 0
    pos = [start + i - 1 for i, mask in enumerate(seg.loss_mask) if mask]
    if limit is not None and len(pos) > limit:
        pos = [pos[round(k * (len(pos) - 1) / (limit - 1))] for k in range(limit)]
    return pos


@torch.no_grad()
def _feed(model, ids, cache, *, position_start, chunk=4096):
    """Read one segment in chunks, retaining the model's attention cache."""
    device = ids.device
    total = ids.shape[1]
    in_cache = (
        0
        if cache is None
        else next(layer.keys.shape[2] for layer in cache.layers if getattr(layer, "keys", None) is not None)
    )
    hidden = []
    for start in range(0, total, chunk):
        piece = ids[:, start : start + chunk]
        length = piece.shape[1]
        positions = torch.arange(position_start + start, position_start + start + length, device=device).unsqueeze(0)
        if cache is None:
            out = model(input_ids=piece, position_ids=positions, use_cache=True, output_hidden_states=True)
        else:
            mask = torch.ones(1, in_cache + start + length, dtype=torch.long, device=device)
            out = model(
                input_ids=piece,
                past_key_values=cache,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                output_hidden_states=True,
            )
        cache = out.past_key_values
        hidden.append(out.hidden_states[-1][0])
    return torch.cat(hidden), cache


@torch.no_grad()
def episode_features(
    model, segments: list[Segment], mode: str, device, *, chunk: int = 4096, limit: int | None = None
) -> list[torch.Tensor]:
    """Return actor-token features for separate segments or the concatenated history."""
    if mode not in MODES:
        raise ValueError(f"Unsupported critic input mode: {mode}")
    out, cache, seen = [], None, 0
    with assigning_cache_updates():
        for seg in segments:
            ids = torch.tensor([seg.tokens], device=device)
            if mode == "shared":
                hidden, _ = _feed(model, ids, None, position_start=0, chunk=chunk)
            else:
                hidden, cache = _feed(model, ids, cache, position_start=seen, chunk=chunk)
            out.append(hidden[torch.tensor(value_positions(seg, limit), device=device)].float().cpu())
            seen += len(seg.tokens)
    return out
