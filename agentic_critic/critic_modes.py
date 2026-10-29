"""Critic forward passes with independent context and memory on a Hugging Face causal LM.

All settings predict a value for every actor-generated token and are trained on the same
trajectories with the same targets. They differ only in what the critic can see of the earlier
segments of an episode, and in whether the loss can back-propagate into them:

  shared       each segment alone: what the actor saw after its own compaction
  state        earlier segments reach the critic only through the recurrent states of the
               linear-attention layers; their attention KV is hidden and positions restart
               (hybrid models only)
  full_detach  earlier segments fully visible, but as constants: no gradient into them
  full         the whole append-only log in one differentiable forward pass
  attention_matching  fitted pre-summary KV memory, using all original actor summary query positions
  kv           earlier segments reach the critic as a compacted attention KV (plus the recurrent states), as
               constants. After each segment its keys are compacted to a fraction; the memory is recomputed
               with the current weights in every forward pass (fully synchronous). With every key kept this
               is full_detach.

The value of token t is read from the hidden state at position t-1, i.e. before the token is
generated, as in EasyPPO.
"""

from contextlib import contextmanager

import torch
from torch import nn

from agentic_critic.critic_input import Segment
from agentic_critic.memory_config import KV_DEFAULT, resolve_memory_config  # noqa: F401 (public compatibility)

MODES = ("shared", "state", "full_detach", "full", "kv", "attention_matching")
# kv settings: fraction of each segment's keys kept; evict | am | am_cv (agentic_critic/kv_compact.py);
# queries from the next segment's tokens ("hindsight", known at training time) or from the segment itself ("causal").


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.float()).squeeze(-1)


@contextmanager
def assigning_cache_updates():
    """Make the linear-attention cache store new states by assignment instead of in-place copy.

    transformers writes the recurrent and conv states into the cache with `copy_` (to keep a static
    address for compiled inference). During training that overwrites a tensor the graph still needs,
    and backward fails. Assignment keeps the old tensor intact; the values are identical.
    """
    from transformers.cache_utils import LinearAttentionLayer

    orig_conv, orig_rec = LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state

    def update_conv_state(self, conv_states, **kwargs):
        if not self.is_conv_states_initialized:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            self.max_batch_size, self.conv_kernel_size = conv_states.shape[0], conv_states.shape[-1]
            self.is_conv_states_initialized = True
        k, new = self.conv_kernel_size, conv_states.shape[-1]
        if not self.has_previous_state or new >= k:
            self.conv_states = conv_states[..., -k:]
        else:
            self.conv_states = torch.cat([self.conv_states[..., new:], conv_states], dim=-1)
        self.has_previous_state = True
        return self.conv_states

    def update_recurrent_state(self, recurrent_states, **kwargs):
        self.is_recurrent_states_initialized = True
        self.recurrent_states = recurrent_states
        return self.recurrent_states

    LinearAttentionLayer.update_conv_state = update_conv_state
    LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    try:
        yield
    finally:
        LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state = orig_conv, orig_rec


def _detach_cache(cache, keep_kv: bool):
    """Cut the graph at a segment boundary. With keep_kv=False only recurrent/conv states survive as content;
    the KV entries stay in the cache object (their length is needed) and are hidden by the attention mask."""
    for layer in cache.layers:
        for name in ("keys", "values", "recurrent_states", "conv_states"):
            t = getattr(layer, name, None)
            if t is None:
                continue
            if isinstance(t, (list, tuple)):
                setattr(layer, name, type(t)(x.detach() for x in t))
            else:
                setattr(layer, name, t.detach())
    return cache


def _values_for_segment(hidden: torch.Tensor, head: ValueHead, seg: Segment, offset: int, features: list | None):
    """Values for the response region of `seg`, whose tokens start at `offset` in `hidden` ([T, H])."""
    start = offset + seg.response_start
    assert start > 0, "the first response token needs a preceding position to read its value from"
    h = hidden[start - 1 : start - 1 + seg.response_length]
    if features is not None:
        features.append(h.detach())
    return head(h)


def episode_values(
    model, head: ValueHead, segments: list[Segment], mode: str, device, features: list | None = None, kv: dict | None = None
) -> list[torch.Tensor]:  # fmt: skip
    """Per segment, a [response_length] tensor of values aligned with that segment's loss_mask.

    If `features` is a list, the hidden states the values were read from are appended to it, one
    [response_length, H] tensor per segment (detached). They give the value head's per-token gradient
    without a backward pass: d(0.5 (V - G)^2)/dW = (V - G) h.
    """
    assert mode in MODES
    config = resolve_memory_config(mode, kv)
    with assigning_cache_updates():
        if config is not None:
            return _episode_values_kv(model, head, segments, device, features, config)
        return _episode_values(model, head, segments, mode, device, features)


def _episode_values_kv(model, head, segments, device, features, kv) -> list[torch.Tensor]:
    from agentic_critic import kv_compact

    kv_compact.validate_model(model)
    kv_compact.validate_summary_segments(segments, kv)
    out, cache, biases, seen, memory = [], None, {}, 0, 0
    for s, seg in enumerate(segments):
        ids = torch.tensor([seg.tokens], device=device)
        n = len(seg.tokens)
        query_ctx = kv_compact.segment_queries(model, seg, kv)
        with query_ctx as own_queries:
            if cache is None:
                res = model(input_ids=ids, output_hidden_states=True, use_cache=True)
            else:
                res = kv_compact.continue_from(model, cache, biases, ids, seen, output_hidden_states=True)
        out.append(_values_for_segment(res.hidden_states[-1][0], head, seg, 0, features))
        seen += n
        if s == len(segments) - 1:
            break
        cache = _detach_cache(res.past_key_values, keep_kv=True)
        biases, memory = kv_compact.compact_segment(
            model, cache, segments[s + 1].tokens, seen=seen, segment_length=n, memory_length=memory,
            biases=biases, own_queries=own_queries, config=kv, summary_span=seg.summary_span,
        )
    return out


def _episode_values(model, head, segments, mode, device, features) -> list[torch.Tensor]:
    if mode == "full":
        ids = torch.tensor([[t for s in segments for t in s.tokens]], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
        out, offset = [], 0
        for seg in segments:
            out.append(_values_for_segment(hidden, head, seg, offset, features))
            offset += len(seg.tokens)
        return out

    out, cache, seen = [], None, 0
    for seg in segments:
        ids = torch.tensor([seg.tokens], device=device)
        n = len(seg.tokens)
        kwargs = {}
        if mode != "shared" and cache is not None:
            mask = torch.ones(1, seen + n, dtype=torch.long, device=device)
            if mode == "state":
                mask[:, :seen] = 0
                positions = torch.arange(n, device=device).unsqueeze(0)
            else:
                positions = torch.arange(seen, seen + n, device=device).unsqueeze(0)
            kwargs = dict(past_key_values=cache, attention_mask=mask, position_ids=positions)
        res = model(input_ids=ids, output_hidden_states=True, use_cache=mode != "shared", **kwargs)
        out.append(_values_for_segment(res.hidden_states[-1][0], head, seg, 0, features))
        if mode != "shared":
            cache = _detach_cache(res.past_key_values, keep_kv=mode == "full_detach")
            seen += n
    return out


def value_loss(values: list[torch.Tensor], segments: list[Segment], target: float) -> tuple[torch.Tensor, int]:
    """Sum of squared errors over actor tokens, and their count (for token-mean reduction over a batch)."""
    total, count = values[0].new_zeros(()), 0
    for v, seg in zip(values, segments, strict=True):
        m = torch.tensor(seg.loss_mask, device=v.device, dtype=torch.bool)
        total = total + ((v[m] - target) ** 2).sum()
        count += int(m.sum())
    return total, count
