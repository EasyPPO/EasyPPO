"""Reading an episode under a critic setting with bounded memory, for training on the Hugging Face model.

`agentic_critic.critic_modes.episode_values` builds one graph for the whole episode. That is fine for a tiny
model; a 9B critic on segments of tens of thousands of tokens needs gradient checkpointing, and the library's
own checkpointing drops the cache, which is exactly what the State, Full-detach and KV settings read. Here:

  - every decoder layer is checkpointed by us, with the cache left in place;
  - while gradients are taken the cache is read-only (`frozen_cache_updates`): a layer sees the memory followed
    by its own new entries but stores nothing, so the re-run that checkpointing does during backward computes
    the same thing as the first run;
  - the memory moves forward in a separate pass without gradients.

Because the memory is a constant in all settings but Full, the loss of a segment can be back-propagated as soon
as the segment has been read (`on_values` is called inside the contexts that the backward pass needs), and only
one segment's activations are alive at a time. Full is one pass over the whole log.
"""

import copy
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from agentic_critic import kv_compact
from agentic_critic.critic_input import Segment
from agentic_critic.critic_modes import MODES, _detach_cache, assigning_cache_updates


@contextmanager
def frozen_cache_updates():
    """Cache layers return what a layer needs (memory followed by the new entries) and store nothing."""
    from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

    originals = (DynamicLayer.update, LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state)

    def update(self, key_states, value_states, *args, **kwargs):
        if not getattr(self, "is_initialized", False) or self.keys.numel() == 0:
            return key_states, value_states
        return torch.cat([self.keys, key_states], dim=-2), torch.cat([self.values, value_states], dim=-2)

    def update_conv_state(self, conv_states, **kwargs):
        return conv_states

    def update_recurrent_state(self, recurrent_states, **kwargs):
        return recurrent_states

    DynamicLayer.update, LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state = update, update_conv_state, update_recurrent_state
    try:
        yield
    finally:
        DynamicLayer.update, LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state = originals


@contextmanager
def checkpointed_layers(model):
    """Run every decoder layer under torch's checkpoint (non-reentrant), keeping keyword arguments such as the cache."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer in layers:
        layer.forward = partial(checkpoint, type(layer).forward, layer, use_reentrant=False)
    try:
        yield
    finally:
        for layer in layers:
            del layer.forward


MEMORY_ATTENTION_CALLS = [0]  # how often the square-causal path below was taken (for tests)


def _memory_sdpa(module, query, key, value, attention_mask, dropout: float = 0.0, scaling: float | None = None, is_causal=None, **kwargs):
    """Attention of L new tokens over M memory entries followed by the L tokens themselves, on the fast kernel.

    The library hands an explicit mask to scaled_dot_product_attention as soon as there are more keys than
    queries, and the kernel then materialises an L x (M + L) matrix per head: tens of GB for one segment. A plain
    causal mask does not need that. Put M dummy queries in front: the problem is square, row M + i sees the M
    memory entries and the new tokens up to i, which is exactly what token i must see, and `is_causal=True`
    dispatches to the flash kernel. The dummy rows are dropped. Only valid when nothing else is masked.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    new, total = query.shape[2], key.shape[2]
    if total == new:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    MEMORY_ATTENTION_CALLS[0] += 1
    padded = torch.cat([query.new_zeros(*query.shape[:2], total - new, query.shape[3]), query], dim=2)
    out = torch.nn.functional.scaled_dot_product_attention(
        padded, key, value, attn_mask=None, dropout_p=dropout, scale=scaling, is_causal=True, enable_gqa=key.shape[1] != query.shape[1]
    )
    return out[:, :, total - new :].transpose(1, 2).contiguous(), None


@contextmanager
def fast_memory_attention(model):
    """Use `_memory_sdpa` in the full-attention layers. For passes whose only mask is the causal one."""
    from transformers import AttentionInterface

    AttentionInterface.register("memory_sdpa", _memory_sdpa)
    previous = model.config._attn_implementation
    model.config._attn_implementation = "memory_sdpa"
    try:
        yield
    finally:
        model.config._attn_implementation = previous


def _values(hidden: torch.Tensor, head, seg: Segment, offset: int) -> torch.Tensor:
    start = offset + seg.response_start
    assert start > 0, "the first response token needs a preceding position to read its value from"
    return head(hidden[start - 1 : start - 1 + seg.response_length])


def read_episode(model, head, segments: list[Segment], mode: str, device, on_values, *, train: bool, kv: dict | None = None,
                 use_checkpoint: bool = True) -> None:  # fmt: skip
    """Read `segments` under `mode` and hand the values to `on_values([(segment index, values), ...])`.

    With train=True the values carry gradients and `on_values` must call backward itself before returning
    (the caller wraps the whole call in whatever autocast it wants). With train=False run it under no_grad.
    Calls come in episode order: one segment at a time, or all segments at once in the Full setting.
    """
    assert mode in MODES
    lm = model
    model = getattr(lm, "model", lm)  # the backbone: the wrapper would also compute vocabulary logits at every position
    grad_ctx = (lambda: checkpointed_layers(model)) if (train and use_checkpoint) else nullcontext
    if mode == "full":
        ids = torch.tensor([[t for s in segments for t in s.tokens]], device=device)
        with grad_ctx():
            hidden = model(input_ids=ids, use_cache=False).last_hidden_state[0]
            out, offset = [], 0
            for i, seg in enumerate(segments):
                out.append((i, _values(hidden, head, seg, offset)))
                offset += len(seg.tokens)
            on_values(out)
        return

    cache, biases, seen, memory = None, {}, 0, 0
    for s, seg in enumerate(segments):
        ids = torch.tensor([seg.tokens], device=device)
        n = len(seg.tokens)
        kwargs = {}
        if mode != "shared" and cache is not None:
            if mode == "state":  # the memory's attention KV is hidden and positions restart
                mask = torch.ones(1, seen + n, dtype=torch.long, device=device)
                mask[:, :seen] = 0
                kwargs = dict(past_key_values=cache, attention_mask=mask, position_ids=torch.arange(n, device=device).unsqueeze(0))
            else:
                kwargs = dict(past_key_values=cache, position_ids=torch.arange(seen, seen + n, device=device).unsqueeze(0))
                if mode == "full_detach":
                    kwargs["attention_mask"] = torch.ones(1, seen + n, dtype=torch.long, device=device)
        # A memory read with nothing but the causal mask goes through the fast kernel: Full-detach always, KV while
        # every bias is zero (eviction keeping the same number of keys in every head).
        plain = mode == "full_detach" or (mode == "kv" and all(bool((b == 0).all()) for b in biases.values()))
        if mode == "kv" and biases and not plain:
            bias_ctx = lambda: kv_compact.attention_bias(model, biases)  # noqa: E731
        elif plain and cache is not None:
            bias_ctx = lambda: fast_memory_attention(model)  # noqa: E731
        else:
            bias_ctx = nullcontext
        last = s == len(segments) - 1
        if train:
            with frozen_cache_updates(), grad_ctx(), bias_ctx():
                res = model(input_ids=ids, use_cache="past_key_values" in kwargs, **kwargs)
                on_values([(s, _values(res.last_hidden_state[0], head, seg, 0))])
            del res
            if last or mode == "shared":
                continue
        # Move the memory forward (and, without gradients, this pass is also where the values come from).
        with torch.no_grad(), assigning_cache_updates(), bias_ctx(), kv_compact.capture_queries(model) as own_queries:
            res = model(input_ids=ids, use_cache=mode != "shared", **kwargs)
        if not train:
            on_values([(s, _values(res.last_hidden_state[0], head, seg, 0))])
        if last or mode == "shared":
            continue
        cache = _detach_cache(res.past_key_values, keep_kv=mode in ("full_detach", "kv"))
        seen += n
        if mode != "kv":
            continue
        with torch.no_grad():
            if kv["queries"] == "hindsight":
                # The tokens that will read this memory are known: run them once on the uncompacted keys.
                uncompacted = {i: torch.cat([b, b.new_zeros(b.shape[0], n)], dim=1) for i, b in biases.items()}
                nxt = torch.tensor([segments[s + 1].tokens], device=device)
                positions = torch.arange(seen, seen + nxt.shape[1], device=device).unsqueeze(0)
                if plain:
                    bias_next = lambda: fast_memory_attention(model)  # noqa: E731
                else:
                    bias_next = (lambda: kv_compact.attention_bias(model, uncompacted)) if uncompacted else nullcontext
                with kv_compact.capture_queries(model) as queries, bias_next():
                    model(input_ids=nxt, past_key_values=copy.deepcopy(cache), position_ids=positions, use_cache=True)
                queries = dict(queries)
            else:
                queries = dict(own_queries)
            keep = max(1, round(kv["ratio"] * n))
            # the fits are done in float32: no autocast inside (a caller may have it on)
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                biases = kv_compact.compact_cache(
                    model, cache, queries, keep, method=kv["method"], causal=kv["queries"] == "causal", start=memory, prev_biases=biases
                )  # fmt: skip
            memory += keep


# ---- EasyPPO's critic update (as ported to Miles: miles/backends/training_utils/loss/hub/losses.py) ----


def problem_weights(rewards: dict[str, list[float]], std_floor: float) -> dict[str, float]:
    """Noise-normalised weights, one per problem: 1 / max(std of its episodes' rewards, floor), divided by the
    mean of that over the problems of the step. The std is the population std over EPISODES."""
    raw = {p: 1.0 / max(float(torch.tensor(r, dtype=torch.float32).var(correction=0).sqrt()), std_floor) for p, r in rewards.items()}
    mean = sum(raw.values()) / len(raw)
    return {p: w / mean for p, w in raw.items()}


def clipped_value_loss(values: torch.Tensor, old_values: torch.Tensor, target: float, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token PPO value loss max((v_clipped - G)^2, (v - G)^2) and whether the clip binds."""
    clipped = old_values + (values - old_values).clamp(-clip, clip)
    return torch.max((clipped - target) ** 2, (values - target) ** 2), (values - old_values).abs() > clip


class AdamWHalfMoments(torch.optim.Optimizer):
    """AdamW with the two moments stored in bfloat16 and the parameters in their own precision.

    For a 9B critic in float32 the moments are 72 GB in float32 and 36 GB like this. The arithmetic of a step is
    done in float32. (Megatron's precision-aware optimiser, which the Miles runs use, does the same.)
    """

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01, moment_dtype=torch.bfloat16):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.moment_dtype = moment_dtype

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=self.moment_dtype)
                    state["v"] = torch.zeros_like(p, dtype=self.moment_dtype)
                state["step"] += 1
                g = p.grad.float()
                m = state["m"].float().mul_(b1).add_(g, alpha=1 - b1)
                v = state["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                state["m"].copy_(m)
                state["v"].copy_(v)
                t = state["step"]
                step_size = group["lr"] / (1 - b1**t)
                denom = (v / (1 - b2**t)).sqrt_().add_(group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_((m / denom).to(p.dtype), alpha=-step_size)
