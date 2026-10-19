"""KV compaction of a Qwen3.5 (hybrid) cache with Attention Matching, on Hugging Face transformers.

Only the full-attention layers are compacted. The linear-attention layers keep their recurrent and
conv states, which are fixed-size.

The per-head algorithm (which keys to keep, the bias beta, the new values) is the upstream one from
"Fast KV Compaction via Attention Matching" (github.com/adamzweiger/compaction), imported unchanged.
This module is the glue the upstream repo does not have for this model family:

  prefill_with_queries   run a prefix, return its cache and the attention queries of every full-attention layer
  compact_cache          replace each layer's keys/values by the compacted ones, return the biases
  attention_bias         context manager that adds the biases to the attention logits of the compacted keys
  continue_from          run more tokens on top of a (compacted) cache at their original positions

Everything here is inference-side and gradient-free with respect to the compacted prefix.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

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


@contextmanager
def capture_queries(model):
    """Yield a dict that each forward pass inside the block fills with {layer index: queries [heads, T, d]}, detached."""
    queries, hooks = {}, []
    for idx, attn in _attention_layers(model):

        def pre(module, args, kwargs, idx=idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            queries[idx] = _queries(module, hidden, kwargs["position_embeddings"])[0].detach()

        hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield queries
    finally:
        for h in hooks:
            h.remove()


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
def probe_queries(model, cache, probe_ids: torch.Tensor, logical_prefix_len: int) -> dict:
    """Attention queries of tokens that come AFTER the prefix, as {layer index: [heads, n, d]}.

    These are the queries that will actually read the compacted keys: they sit at later positions and
    attend backwards. Queries taken from the prefix's own tokens see the keys at different relative
    positions (including future ones), which makes them poor stand-ins. In an RL rollout the natural
    probe is the continuation of the same trajectory. The cache is not modified.
    """
    import copy

    queries, hooks = {}, []
    for idx, attn in _attention_layers(model):

        def pre(module, args, kwargs, idx=idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            queries[idx] = _queries(module, hidden, kwargs["position_embeddings"])[0].detach()

        hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        n = probe_ids.shape[1]
        positions = torch.arange(logical_prefix_len, logical_prefix_len + n, device=probe_ids.device).unsqueeze(0)
        model(input_ids=probe_ids, past_key_values=copy.deepcopy(cache), position_ids=positions, use_cache=True)
    finally:
        for h in hooks:
            h.remove()
    return queries


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
    gram = x.T @ x
    lam = ridge * gram.diagonal().mean()
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return v0 + torch.linalg.solve(gram + lam * eye, x.T @ (y - x @ v0))


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
    ridge: float = 1e-4, start: int = 0, prev_biases: dict | None = None,
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
        return _compact_cache(model, cache, queries, keep, method, max_queries, causal, ridge, start, prev_biases, alg, gen, seed)


def _compact_cache(model, cache, queries, keep, method, max_queries, causal, ridge, start, prev_biases, alg, gen, seed):
    biases = {}
    for idx, attn in _attention_layers(model):
        layer = cache.layers[idx]
        old_keys, old_values = layer.keys[:, :, :start], layer.values[:, :, :start]
        keys, values = layer.keys[:, :, start:], layer.values[:, :, start:]  # [1, kv_heads, T, d]
        kv_heads, total = keys.shape[1], keys.shape[2]
        old_bias = torch.zeros(kv_heads, start, dtype=torch.float32, device=keys.device)
        if prev_biases and idx in prev_biases:
            old_bias = prev_biases[idx].to(keys.device)
            assert old_bias.shape == (kv_heads, start)
        keeps = head_keeps(model, total, keep)
        if method == "none":
            biases[idx] = torch.cat([old_bias, torch.zeros(kv_heads, total, dtype=torch.float32, device=keys.device)], dim=1)
            continue
        group = attn.num_key_value_groups
        new_k, new_v, beta = [], [], []
        for g in range(kv_heads):
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

        def pad(x, fill=0.0):
            return torch.cat([x, x.new_full((longest - x.shape[0], *x.shape[1:]), fill)])

        layer.keys = torch.cat([old_keys, torch.stack([pad(k) for k in new_k]).unsqueeze(0)], dim=2)
        layer.values = torch.cat([old_values, torch.stack([pad(v) for v in new_v]).unsqueeze(0)], dim=2)
        biases[idx] = torch.cat([old_bias, torch.stack([pad(b, PAD_BIAS) for b in beta]).to(keys.device)], dim=1)
    return biases


@contextmanager
def attention_bias(model, biases: dict):
    """Add `biases[layer]` ([kv_heads, t]) to the attention logits of the first t (compacted) key positions."""
    hooks = []
    for idx, attn in _attention_layers(model):
        if idx not in biases:
            continue

        def pre(module, args, kwargs, idx=idx):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            bias = biases[idx]
            t, new = bias.shape[1], hidden.shape[1]
            per_head = bias.repeat_interleave(module.num_key_value_groups, dim=0)  # [heads, t]
            # Build this layer's own causal mask. The one transformers passes is sized from a single layer's
            # cache, which is wrong for the others once layers keep different numbers of keys; it is also
            # boolean, and a bias needs the additive form. Assumes one unpadded sequence per call.
            mask = torch.zeros(1, 1, new, t + new, dtype=hidden.dtype, device=hidden.device)
            future = torch.triu(torch.ones(new, new, dtype=torch.bool, device=hidden.device), diagonal=1)
            mask[0, 0, :, t:].masked_fill_(future, torch.finfo(hidden.dtype).min)
            mask = mask.expand(-1, per_head.shape[0], -1, -1).clone()
            mask[:, :, :, :t] += per_head.to(mask.dtype)[None, :, None, :]
            kwargs["attention_mask"] = mask
            return args, kwargs

        hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def continue_from(model, cache, biases: dict, input_ids: torch.Tensor, logical_prefix_len: int, **kwargs):
    """Run `input_ids` after a cache whose prefix originally had `logical_prefix_len` tokens.

    The new tokens keep the positions they would have had without compaction.
    """
    n = input_ids.shape[1]
    positions = torch.arange(logical_prefix_len, logical_prefix_len + n, device=input_ids.device).unsqueeze(0)
    with attention_bias(model, biases):
        return model(input_ids=input_ids, past_key_values=cache, position_ids=positions, use_cache=True, **kwargs)
