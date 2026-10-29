"""KV compaction of a Qwen3.5 (hybrid) cache with Attention Matching, on Hugging Face transformers.

Only the full-attention layers are compacted. The linear-attention layers keep their recurrent and
conv states, which are fixed-size.

Key selection and mass fitting reuse the pinned upstream implementation from
"Fast KV Compaction via Attention Matching" (github.com/adamzweiger/compaction).
The am_ridge variant fits values locally with regularization towards retained values.
This module adapts the algorithm to this model family:

  prefill_with_queries   run a prefix, return its cache and the attention queries of every full-attention layer
  compact_cache          replace each layer's keys/values by the compacted ones, return the biases
  attention_bias         context manager that adds the biases to the attention logits of the compacted keys
  continue_from          run more tokens on top of a (compacted) cache at their original positions

Everything here is inference-side and gradient-free with respect to the compacted prefix.
"""

import copy
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from agentic_critic.memory_attention import biased_attention
from agentic_critic.memory_config import DEFAULT_QUERY_COUNT

UPSTREAM = os.environ.get("KV_COMPACTION_REPO", str(Path(__file__).resolve().parents[1] / "third_party/compaction"))


def _attention_layers(model):
    """(layer index, attention module) for every full-attention layer."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    return [(i, layer.self_attn) for i, layer in enumerate(layers) if getattr(layer, "layer_type", None) == "full_attention"]


def _queries(attn, hidden_states, position_embeddings) -> torch.Tensor:
    """The rotated, normalised queries exactly as the attention module computes them: [B, heads, T, d]."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    q, _ = torch.chunk(attn.q_proj(hidden_states).view(*hidden_states.shape[:-1], -1, attn.head_dim * 2), 2, dim=-1)
    q = attn.q_norm(q.view(shape)).transpose(1, 2)
    cos, sin = position_embeddings
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q


def evenly_spaced_positions(length: int, count: int, *, device=None) -> torch.Tensor:
    """Distinct positions spanning the segment, including both ends when N > 1."""
    if length < 1 or count < 1:
        raise ValueError("query segment length and sample count must be positive")
    count = min(length, count)
    if count == 1:
        return torch.tensor([length // 2], device=device)
    return torch.div(torch.arange(count, device=device) * (length - 1), count - 1, rounding_mode="floor")


def validate_model(model):
    """This adapter reconstructs Qwen3.5's gated, normalized, post-RoPE queries."""
    if model.config.model_type != "qwen3_5_text" or not _attention_layers(model):
        raise ValueError("KV compaction currently supports Qwen3.5 text backbones with full-attention layers")
    if model.config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("KV compaction requires eager or sdpa attention to apply per-head mass biases")


@contextmanager
def capture_queries(model, query_count: int | None = None, *, query_span=None):
    """Yield a dict that each forward pass inside the block fills with {layer index: queries [heads, T, d]}, detached."""
    queries, hooks = {}, []
    for idx, attn in _attention_layers(model):

        def pre(module, args, kwargs, idx=idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            embeddings = kwargs["position_embeddings"]
            if query_span is not None:
                start, end = query_span
                end = min(end, start + query_count) if query_count else end
                hidden = hidden[:, start:end]
                embeddings = tuple(e[..., start:end, :] for e in embeddings)
            elif query_count is not None:
                # N counts token positions: all GQA query heads contribute at each position.
                count = query_count or math.ceil(5000 / module.num_key_value_groups)
                positions = evenly_spaced_positions(hidden.shape[1], count, device=hidden.device)
                hidden = hidden.index_select(1, positions)
                embeddings = tuple(e.index_select(-2, positions) for e in embeddings)
            with torch.no_grad():
                queries[idx] = _queries(module, hidden, embeddings)[0].detach()

        hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield queries
    finally:
        for h in hooks:
            h.remove()


def validate_summary_segments(segments, config):
    if config["queries"] == "summary":
        for idx, segment in enumerate(segments[:-1]):
            if segment.summary_span is None or segment.summary_span[0] <= 0:
                raise ValueError(
                    f"segment {idx} has no actor summary_span; re-export rollouts with summary metadata "
                    "or use mode=kv with queries=hindsight/causal explicitly"
                )


def segment_queries(model, segment, config):
    """Capture critic queries from its existing segment pass, never from the actor model."""
    if config is None:
        return nullcontext({})
    if config["queries"] == "summary" and segment.summary_span is not None:
        return capture_queries(model, query_span=segment.summary_span)
    if config["queries"] == "causal":
        return capture_queries(model)
    return nullcontext({})


@torch.no_grad()
def prefill_with_queries(model, input_ids: torch.Tensor, chunk: int | None = None):
    """Forward a prefix. Returns (cache, {layer index: queries [heads, T, d]}).

    `chunk`: feed the prefix this many tokens at a time. The result is the same, but the attention of one pass
    is chunk x T instead of T x T, which is what makes prefixes of tens of thousands of tokens fit in memory.
    """
    total = input_ids.shape[1]
    step = chunk or total
    cache, parts = None, []
    for start in range(0, total, step):
        ids = input_ids[:, start : start + step]
        with capture_queries(model) as queries:
            if cache is None:
                cache = model(input_ids=ids, use_cache=True).past_key_values
            else:
                positions = torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)
                cache = model(input_ids=ids, past_key_values=cache, position_ids=positions, use_cache=True).past_key_values
        parts.append(dict(queries))
    return cache, {idx: torch.cat([p[idx] for p in parts], dim=1) for idx in parts[0]}


@torch.no_grad()
def probe_queries(model, cache, probe_ids: torch.Tensor, logical_prefix_len: int, *, biases=None, query_count=DEFAULT_QUERY_COUNT) -> dict:
    """Probe only the first N tokens of the next segment for the hindsight ablation.

    The private cache includes earlier compacted memory and its biases. Probe tokens never enter
    persistent memory. Explicit query_count=0 retains the legacy full-segment, uniform-sampling path.
    """
    backbone = getattr(model, "model", model)
    if query_count > 0:
        probe_ids = probe_ids[:, :query_count]
    # Even the first probe has a prefix. Use the chunked read path for zero-bias memory too,
    # rather than letting transformers allocate a full next-segment-by-history mask.
    if not biases:
        biases = {
            idx: cache.layers[idx].keys.new_zeros(cache.layers[idx].keys.shape[1:3], dtype=torch.float32)
            for idx, _ in _attention_layers(backbone)
        }
    with capture_queries(backbone, query_count=0 if query_count == 0 else None) as queries:
        continue_from(backbone, copy.deepcopy(cache), biases, probe_ids, logical_prefix_len)
    return dict(queries)


def _algorithm(select_only: bool = False):
    if not (Path(UPSTREAM) / "compaction/algorithms/highest_attention_keys.py").is_file():
        raise FileNotFoundError(
            "KV compaction dependency not found. Run `git submodule update --init third_party/compaction` "
            "or set KV_COMPACTION_REPO to an adamzweiger/compaction checkout."
        )
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)
    from compaction.algorithms.highest_attention_keys import HighestAttentionKeysCompaction

    if select_only:
        # Eviction needs upstream's key selection and nothing else: no bias fit, no value fit. The selection
        # does not depend on these two settings.
        return HighestAttentionKeysCompaction(score_method="rms", beta_method="zero", c2_method="direct")
    # The settings of the upstream demo (examples/qa_demo.py).
    return HighestAttentionKeysCompaction(
        score_method="rms", nnls_iters=2, nnls_lower_bound=0.05, nnls_upper_bound=20.0, c2_method="lsq"
    )


def _ridge_value_fit(k, v, q, c1, beta, sel, bias, mask_compact: bool, ridge: float):
    """Values by least squares with a ridge towards the original values of the kept keys.

    bias: the mask on the original keys ([n, T], or None). mask_compact: apply it to the compact keys as well, so
    that a causal query does not attend to compact keys from its own future during the fit; queries that then see
    no compact key are dropped. A key that no query constrains keeps its own value.
    """
    scale = k.shape[1] ** -0.5
    sel = torch.as_tensor(sel, device=k.device)
    orig = q @ k.T * scale
    y = torch.softmax(orig if bias is None else orig + bias, dim=1) @ v
    logits = q @ c1.T * scale + beta
    if mask_compact:
        logits = logits + bias[:, sel]
        rows = torch.isfinite(logits).any(dim=1)
        logits, y = logits[rows], y[rows]
    x = torch.softmax(logits, dim=1)
    v0 = v[sel]
    if x.shape[0] == 0:
        return v0
    # The dual solve covers N*G < K (short next segments) without a K-by-K allocation.
    lam = (ridge * x.square().sum() / x.shape[1]).clamp_min(torch.finfo(x.dtype).eps)
    residual = y - x @ v0
    if x.shape[0] < x.shape[1]:
        gram = x @ x.T
        gram.diagonal().add_(lam)
        return v0 + x.T @ torch.linalg.solve(gram, residual)
    gram = x.T @ x
    gram.diagonal().add_(lam)
    return v0 + torch.linalg.solve(gram, x.T @ residual)


PAD_BIAS = -1e9  # bias on padding slots: they receive no attention


def head_keeps(model, total: int, keep) -> dict:
    """Number of keys to keep per KV head, as {(layer index, head): t}.

    `keep` is either an int (the same for every head) or `(budgets, ratio)`: `budgets` maps "L{layer}H{head}"
    to that head's share of all kept keys (shares sum to 1 over the full-attention KV heads, the format of the
    upstream head_budgets files) and `ratio` is the overall fraction of keys kept.
    """
    heads = [(idx, g) for idx, attn in _attention_layers(model) for g in range(attn.config.num_key_value_heads)]
    if isinstance(keep, int):
        return dict.fromkeys(heads, min(keep, total))
    budgets, ratio = keep
    pool = ratio * total * len(heads)
    return {(idx, g): max(1, min(total, round(budgets[f"L{idx}H{g}"] * pool))) for idx, g in heads}


@torch.no_grad()
def compact_cache(
    model, cache, queries: dict, keep, method: str = "am", max_queries: int = 5000, seed: int = 0, causal: bool = False,
    ridge: float = 1e-4, start: int = 0, prev_biases: dict | None = None, end: int | None = None,
):  # fmt: skip
    """Compact every full-attention layer of `cache` in place.

    `keep`: keys kept per KV head, an int or `(budgets, ratio)` for non-uniform budgets (see `head_keeps`).
    Heads of one layer share a tensor, so shorter heads are padded to the longest one in the layer; padding
    slots carry a bias of PAD_BIAS and are never attended to. (As upstream notes, padding means non-uniform
    budgets do not yet save memory in this representation.)

    method: "am" (Attention Matching: selected keys, fitted bias and values),
            "evict" (the same selected keys with their original values, no bias),
            "evict_beta" (the same selected keys with their original values and the fitted bias),
            "am_cv" (Attention Matching with the value fit redone under the causal mask on BOTH sides; needs
            causal=True. Upstream applies the mask only to the original keys in the value fit, so there a
            query also attends to compact keys from its own future),
            "am_ridge" (Attention Matching with the value fit redone as a ridge towards the original values of
            the kept keys, masks as upstream; lam -> 0 is the plain fit, lam -> infinity is evict_beta),
            "recent", "random" (keep the last / a random subset of the keys unchanged; `queries` is not used),
            "none" (keep everything; for checking the plumbing).
    causal: set it when `queries` are the prefix's own tokens (from `prefill_with_queries`). Query number i then
            only sees keys up to position i, as it did in the forward pass; without it every query is scored
            against all keys, including later ones it never attended to.
    start, prev_biases: compact only the keys from index `start` on and leave the first `start` (an already
            compacted memory, with its biases `prev_biases`) as they are. The fit then sees the new keys alone:
            attention is treated as if it were restricted to them. Needs an int `keep`.
    end: stop the compacted range here; preserve all suffix K/V with zero bias (the actor summary).
    Returns {layer index: bias [kv_heads, t_layer]}, t_layer being the longest head of that layer (plus `start`).
    """
    assert start == 0 or isinstance(keep, int)
    assert method in ("am", "am_cv", "am_ridge", "evict", "evict_beta", "recent", "random", "none")
    assert method != "am_cv" or causal
    alg = _algorithm(select_only=method == "evict") if method not in ("none", "recent", "random") else None
    gen = torch.Generator().manual_seed(seed)
    # The upstream fits draw from torch's global generator, so a compaction used to depend on how many had run
    # before it. Give each call the same stream and leave the caller's generator as it was.
    cuda = sorted({layer.keys.device.index for layer in cache.layers if getattr(layer, "keys", None) is not None and layer.keys.is_cuda})
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        return _compact_cache(model, cache, queries, keep, method, max_queries, causal, ridge, start, prev_biases, alg, gen, seed, end)


def _compact_cache(model, cache, queries, keep, method, max_queries, causal, ridge, start, prev_biases, alg, gen, seed, end):
    biases = {}
    for idx, attn in _attention_layers(model):
        layer = cache.layers[idx]
        old_keys, old_values = layer.keys[:, :, :start], layer.values[:, :, :start]
        stop = layer.keys.shape[-2] if end is None else end
        if not start < stop <= layer.keys.shape[-2]:
            raise ValueError("compaction requires a nonempty cache range [start, end)")
        keys, values = layer.keys[:, :, start:stop], layer.values[:, :, start:stop]  # [1, kv_heads, T, d]
        suffix_keys, suffix_values = layer.keys[:, :, stop:], layer.values[:, :, stop:]
        suffix_bias = keys.new_zeros(keys.shape[1], suffix_keys.shape[-2], dtype=torch.float32)
        kv_heads, total = keys.shape[1], keys.shape[2]
        old_bias = torch.zeros(kv_heads, start, dtype=torch.float32, device=keys.device)
        if prev_biases and idx in prev_biases:
            old_bias = prev_biases[idx].to(keys.device)
            assert old_bias.shape == (kv_heads, start)
        keeps = head_keeps(model, total, keep)
        if method == "none":
            biases[idx] = torch.cat([old_bias, torch.zeros(kv_heads, total, dtype=torch.float32, device=keys.device), suffix_bias], dim=1)
            continue
        group = attn.num_key_value_groups
        new_k, new_v, beta = [], [], []
        for g in range(kv_heads):
            if keeps[(idx, g)] == total:
                # Exact identity even for rank-deficient or undersampled fits at ratio=1.
                new_k.append(keys[0, g])
                new_v.append(values[0, g])
                beta.append(torch.zeros(total, device=keys.device))
                continue
            if method in ("recent", "random"):
                t = min(keeps[(idx, g)], total)
                sel = torch.arange(total - t, total) if method == "recent" else torch.randperm(total, generator=gen)[:t].sort().values
                new_k.append(keys[0, g, sel])
                new_v.append(values[0, g, sel])
                beta.append(torch.zeros(t, device=keys.device))
                continue
            qg = queries[idx][g * group : (g + 1) * group]  # [group, n_tokens, d]
            q = qg.reshape(-1, keys.shape[-1])
            pos = torch.arange(qg.shape[1]).repeat(group)
            if q.shape[0] > max_queries:
                pick = torch.randperm(q.shape[0], generator=gen)[:max_queries]
                q, pos = q[pick], pos[pick]
            bias = None
            if causal:
                assert qg.shape[1] == total, "causal=True expects the prefix's own queries"
                bias = torch.zeros(q.shape[0], total, device=q.device)
                bias[(torch.arange(total)[None, :] > pos[:, None]).to(q.device)] = float("-inf")
            t = keeps[(idx, g)]
            if method == "evict":
                _, _, sel = alg._select_keys_highest_attention(keys[0, g].float(), q.float(), t, bias)
                sel_sorted = torch.tensor(sel).sort().values  # the retained keys in their original order
                new_k.append(keys[0, g, sel_sorted])
                new_v.append(values[0, g, sel_sorted])
                beta.append(torch.zeros(t, device=keys.device))
                continue
            if method in ("am_cv", "am_ridge"):
                c1, b, sel = alg._select_keys_highest_attention(keys[0, g].float(), q.float(), t, bias)
            else:
                c1, b, c2, sel = alg.compute_compacted_cache(keys[0, g].float(), values[0, g].float(), q.float(), t, bias)
            if method in ("am_cv", "am_ridge"):
                kv = (keys[0, g].float(), values[0, g].float(), q.float(), c1.float(), b.float(), sel)
                c2 = _ridge_value_fit(*kv, bias, mask_compact=method == "am_cv", ridge=ridge)
            order = torch.tensor(sel).argsort()  # keep the retained keys in their original order
            sel_sorted = torch.tensor(sel)[order]
            if method in ("evict", "evict_beta"):
                new_k.append(keys[0, g, sel_sorted])
                new_v.append(values[0, g, sel_sorted])
                beta.append(b[order].float() if method == "evict_beta" else torch.zeros(t, device=keys.device))
            else:
                new_k.append(c1[order].to(keys.dtype))
                new_v.append(c2[order].to(values.dtype))
                beta.append(b[order].float())
        longest = max(k.shape[0] for k in new_k)

        def pad(x, fill=0.0, length=longest):
            return torch.cat([x, x.new_full((length - x.shape[0], *x.shape[1:]), fill)])

        layer.keys = torch.cat([old_keys, torch.stack([pad(k) for k in new_k]).unsqueeze(0), suffix_keys], dim=2)
        layer.values = torch.cat([old_values, torch.stack([pad(v) for v in new_v]).unsqueeze(0), suffix_values], dim=2)
        biases[idx] = torch.cat([old_bias, torch.stack([pad(b, PAD_BIAS) for b in beta]).to(keys.device), suffix_bias], dim=1)
    return biases


@contextmanager
def attention_bias(model, biases: dict):
    """Read biased memory through chunked SDPA; one unpadded sequence per call."""
    with biased_attention(model, _attention_layers(model), biases):
        yield


def continue_from(model, cache, biases: dict, input_ids: torch.Tensor, logical_prefix_len: int, **kwargs):
    """Run `input_ids` after a cache whose prefix originally had `logical_prefix_len` tokens.

    The new tokens keep the positions they would have had without compaction.
    """
    n = input_ids.shape[1]
    positions = torch.arange(logical_prefix_len, logical_prefix_len + n, device=input_ids.device).unsqueeze(0)
    with attention_bias(model, biases):
        return model(input_ids=input_ids, past_key_values=cache, position_ids=positions, use_cache=True, **kwargs)


@torch.no_grad()
def compact_segment(model, cache, next_tokens, *, seen, segment_length, memory_length, biases, own_queries, config, summary_span=None):
    """Advance detached memory identically in the reference and checkpointed episode readers."""
    device = cache.layers[_attention_layers(model)[0][0]].keys.device
    compact_length, end = segment_length, None
    if config["queries"] == "summary":
        compact_length = summary_span[0]
        end = memory_length + compact_length
        queries = own_queries
        max_queries = max(q.shape[0] * q.shape[1] for q in queries.values())
    elif config["queries"] == "hindsight":
        extended = {i: torch.cat([b, b.new_zeros(b.shape[0], segment_length)], dim=1) for i, b in biases.items()}
        ids = torch.tensor([next_tokens], device=device)
        queries = probe_queries(model, cache, ids, seen, biases=extended, query_count=config["query_count"])
        # Sampling already happened in token space; retain every GQA head at those positions.
        max_queries = max(q.shape[0] * q.shape[1] for q in queries.values())
    else:
        queries, max_queries = own_queries, 5000
    keep = max(1, round(config["ratio"] * compact_length))
    with torch.autocast(device_type=device.type, enabled=False):
        result = compact_cache(
            model, cache, queries, keep, method=config["method"], max_queries=max_queries,
            causal=config["queries"] == "causal", ridge=config["ridge"], start=memory_length, prev_biases=biases, end=end,
        )
    return result, memory_length + keep + segment_length - compact_length
