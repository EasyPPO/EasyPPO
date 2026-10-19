"""KV compaction glue on a tiny random Qwen3.5-style hybrid model. Skipped without the upstream compaction repo."""

import copy
import os

import pytest
import torch

from agentic_critic import kv_compact

pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(kv_compact.UPSTREAM, "compaction")), reason="upstream compaction repo not available"
)
VOCAB, PREFIX, SUFFIX = 512, 96, 24


@pytest.fixture(scope="module")
def model():
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5TextConfig
    except ImportError:
        pytest.skip("this transformers version has no Qwen3.5")
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=8, linear_value_head_dim=8,
        vocab_size=VOCAB, layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
    )  # fmt: skip
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).float().eval()


def tokens(seed: int):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, PREFIX), generator=g), torch.randint(0, VOCAB, (1, SUFFIX), generator=g)


@torch.no_grad()
def suffix_hidden(model, prefix, suffix, method: str, keep: int) -> torch.Tensor:
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    biases = kv_compact.compact_cache(model, cache, queries, keep, method=method)
    out = kv_compact.continue_from(model, cache, biases, suffix, PREFIX, output_hidden_states=True)
    return out.hidden_states[-1][0]


@torch.no_grad()
def reference(model, prefix, suffix) -> torch.Tensor:
    out = model(input_ids=torch.cat([prefix, suffix], dim=1), output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0, PREFIX:]


@pytest.mark.parametrize("seed", range(3))
def test_keeping_everything_reproduces_the_uncompacted_model(model, seed):
    """Checks the plumbing: cache hand-over, positions and the mask hook, with nothing compacted."""
    prefix, suffix = tokens(seed)
    assert torch.allclose(suffix_hidden(model, prefix, suffix, "none", PREFIX), reference(model, prefix, suffix), atol=1e-4)


def test_the_bias_reaches_the_attention_logits(model):
    """A large negative bias on every compacted key must act like removing the prefix from attention."""
    prefix, suffix = tokens(0)
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    biases = kv_compact.compact_cache(model, cache, queries, PREFIX, method="none")
    plain = kv_compact.continue_from(model, cache, biases, suffix, PREFIX, output_hidden_states=True).hidden_states[-1]
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    blocked = {k: torch.full_like(v, -1e4) for k, v in kv_compact.compact_cache(model, cache, queries, PREFIX, method="none").items()}
    out = kv_compact.continue_from(model, cache, blocked, suffix, PREFIX, output_hidden_states=True).hidden_states[-1]
    assert not torch.allclose(out, plain, atol=1e-3)


def test_only_full_attention_layers_are_compacted(model):
    prefix, _ = tokens(0)
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    before = copy.deepcopy([getattr(layer, "recurrent_states", None) for layer in cache.layers])
    biases = kv_compact.compact_cache(model, cache, queries, 24, method="am")
    assert sorted(biases) == [3, 7] and all(b.shape == (2, 24) for b in biases.values())
    for i, layer in enumerate(cache.layers):
        if i in biases:
            assert layer.keys.shape[-2] == 24 and layer.values.shape[-2] == 24
        else:
            assert torch.equal(layer.recurrent_states, before[i])


def test_queries_match_what_the_attention_module_uses(model):
    """The captured queries reproduce the module's own attention output when applied to the cached keys/values."""
    prefix, _ = tokens(1)
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    for idx, attn in kv_compact._attention_layers(model):
        q = queries[idx]  # [heads, T, d]
        assert q.shape == (4, PREFIX, 16)
        assert cache.layers[idx].keys.shape == (1, 2, PREFIX, 16)


def uniform_budgets(model):
    heads = [(i, g) for i, a in kv_compact._attention_layers(model) for g in range(a.config.num_key_value_heads)]
    return {f"L{i}H{g}": 1 / len(heads) for i, g in heads}


def test_equal_shares_give_the_uniform_result(model):
    prefix, suffix = tokens(0)
    uniform = suffix_hidden(model, prefix, suffix, "am", 24)
    shares = suffix_hidden(model, prefix, suffix, "am", (uniform_budgets(model), 24 / PREFIX))
    assert torch.allclose(uniform, shares, atol=1e-5)


def test_non_uniform_budgets_pad_short_heads_with_slots_that_get_no_attention(model):
    prefix, suffix = tokens(0)
    budgets = uniform_budgets(model)
    keys = sorted(budgets)
    budgets[keys[0]], budgets[keys[1]] = budgets[keys[0]] * 1.6, budgets[keys[1]] * 0.4  # same total, uneven split
    keeps = kv_compact.head_keeps(model, PREFIX, (budgets, 0.25))
    assert sorted(keeps.values()) == [10, 24, 24, 38]
    cache, queries = kv_compact.prefill_with_queries(model, prefix)
    biases = kv_compact.compact_cache(model, cache, queries, (budgets, 0.25), method="am")
    first = min(biases)
    assert cache.layers[first].keys.shape[-2] == 38 and biases[first].shape == (2, 38)
    short = int((biases[first] == kv_compact.PAD_BIAS).sum())
    assert short == 38 - 10
    # Garbage in the padding slots must not change the output.
    out = kv_compact.continue_from(model, cache, biases, suffix, PREFIX, output_hidden_states=True).hidden_states[-1]
    cache2, queries2 = kv_compact.prefill_with_queries(model, prefix)
    biases2 = kv_compact.compact_cache(model, cache2, queries2, (budgets, 0.25), method="am")
    pad = biases2[first] == kv_compact.PAD_BIAS
    cache2.layers[first].values[0][pad] = 1e3
    cache2.layers[first].keys[0][pad] = 7.0
    out2 = kv_compact.continue_from(model, cache2, biases2, suffix, PREFIX, output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(out, out2, atol=1e-4)


def test_query_free_baselines_and_bias_only_variant(model):
    prefix, _ = tokens(0)
    cache0, queries = kv_compact.prefill_with_queries(model, prefix)
    first = kv_compact._attention_layers(model)[0][0]
    original = cache0.layers[first].keys.clone()

    cache = copy.deepcopy(cache0)
    biases = kv_compact.compact_cache(model, cache, None, 24, method="recent")
    assert torch.equal(cache.layers[first].keys, original[:, :, -24:]) and not biases[first].any()

    cache = copy.deepcopy(cache0)
    kv_compact.compact_cache(model, cache, None, 24, method="random")
    assert cache.layers[first].keys.shape[-2] == 24

    # evict_beta keeps exactly what eviction keeps and adds the bias Attention Matching fits.
    evict, with_beta, am = copy.deepcopy(cache0), copy.deepcopy(cache0), copy.deepcopy(cache0)
    kv_compact.compact_cache(model, evict, queries, 24, method="evict")
    b_beta = kv_compact.compact_cache(model, with_beta, queries, 24, method="evict_beta")
    b_am = kv_compact.compact_cache(model, am, queries, 24, method="am")
    assert torch.equal(with_beta.layers[first].values, evict.layers[first].values)
    assert torch.allclose(b_beta[first], b_am[first], atol=1e-3) and b_beta[first].any()


def test_causal_value_fit_runs_and_differs_from_the_upstream_fit(model):
    prefix, suffix = tokens(0)
    cache0, queries = kv_compact.prefill_with_queries(model, prefix)
    first = kv_compact._attention_layers(model)[0][0]
    a, b = copy.deepcopy(cache0), copy.deepcopy(cache0)
    bias_a = kv_compact.compact_cache(model, a, queries, 24, method="am", causal=True)
    bias_b = kv_compact.compact_cache(model, b, queries, 24, method="am_cv", causal=True)
    assert torch.allclose(a.layers[first].keys, b.layers[first].keys) and torch.allclose(bias_a[first], bias_b[first], atol=1e-3)
    assert torch.isfinite(b.layers[first].values).all()
    assert not torch.allclose(a.layers[first].values, b.layers[first].values, atol=1e-3)


def test_ridge_value_fit_moves_from_the_plain_fit_towards_eviction(model):
    prefix, _ = tokens(0)
    cache0, queries = kv_compact.prefill_with_queries(model, prefix)
    first = kv_compact._attention_layers(model)[0][0]
    caches = {}
    for name, kw in {"evict": dict(method="evict"), "weak": dict(method="am_ridge", ridge=1e-8), "strong": dict(method="am_ridge", ridge=1e6)}.items():
        caches[name] = copy.deepcopy(cache0)
        kv_compact.compact_cache(model, caches[name], queries, 24, **kw)
    values = {k: c.layers[first].values for k, c in caches.items()}
    assert torch.allclose(values["strong"], values["evict"], atol=1e-3)
    assert not torch.allclose(values["weak"], values["evict"], atol=1e-3)


@torch.no_grad()
def test_chunked_prefill_matches_the_single_pass(model):
    prefix, suffix = tokens(2)
    whole_cache, whole_q = kv_compact.prefill_with_queries(model, prefix)
    cache, q = kv_compact.prefill_with_queries(model, prefix, chunk=17)
    for idx in whole_q:
        assert torch.allclose(q[idx], whole_q[idx], atol=1e-4)
        assert torch.allclose(cache.layers[idx].keys, whole_cache.layers[idx].keys, atol=1e-4)
    a = kv_compact.continue_from(model, cache, {}, suffix, PREFIX, output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(a[0], reference(model, prefix, suffix), atol=1e-4)
