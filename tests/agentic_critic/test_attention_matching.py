"""Attention Matching sampling, cache isolation, mass preservation and training contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from agentic_critic import kv_compact
from agentic_critic.critic_modes import episode_values
from agentic_critic.memory_attention import _biased_sdpa
from agentic_critic.memory_config import resolve_memory_config
from tests.agentic_critic.test_critic_modes import head, make_segments, model  # noqa: F401
from tests.agentic_critic.test_critic_train import collect


@pytest.mark.parametrize("length,count,want", [(11, 4, [0, 3, 6, 10]), (3, 9, [0, 1, 2]), (8, 1, [4])])
def test_positions(length, count, want):
    assert kv_compact.evenly_spaced_positions(length, count).tolist() == want


@pytest.mark.parametrize(
    "cfg",
    [
        {"ratio": 0},
        {"ratio": 1.1},
        {"ratio": float("nan")},
        {"query_count": -1},
        {"query_count": 1.5},
        {"ridge": 0},
        {"ridge": float("inf")},
        {"queries": "causal"},
        {"queries": "hindsight"},
        {"query_count": 400},
        {"method": "evict"},
        {"typo": 1},
    ],
)
def test_reject_invalid_am_configuration(cfg):
    with pytest.raises(ValueError):
        resolve_memory_config("attention_matching", cfg)


def test_resolver_preserves_caller_and_rejects_unused_settings():
    cfg = {"ratio": 0.5}
    assert resolve_memory_config("attention_matching", cfg)["method"] == "am_ridge"
    assert cfg == {"ratio": 0.5}
    with pytest.raises(ValueError, match="not used"):
        resolve_memory_config("shared", cfg)


@torch.no_grad()
def test_probe_queries_are_exact_prefix_queries_and_do_not_mutate_memory(model):  # noqa: F811
    segments = make_segments(42, 2)
    prefix, suffix = [torch.tensor([s.tokens]) for s in segments]
    cache, _ = kv_compact.prefill_with_queries(model, prefix)
    before = copy.deepcopy(cache)
    all_q = kv_compact.probe_queries(model, cache, suffix, prefix.shape[1], query_count=suffix.shape[1])
    sampled = kv_compact.probe_queries(model, cache, suffix, prefix.shape[1], query_count=4)
    positions = torch.arange(4)
    for idx in sampled:
        torch.testing.assert_close(sampled[idx], all_q[idx][:, positions], atol=1e-6, rtol=1e-5)
    for old, new in zip(before.layers, cache.layers, strict=True):
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            if (value := getattr(old, attr, None)) is not None:
                assert torch.equal(value, getattr(new, attr)), attr


@torch.no_grad()
def test_summary_capture_uses_every_position_beyond_400(model):  # noqa: F811
    segment = make_segments(0, 1)[0]
    segment.tokens += [7] * 410
    segment.loss_mask += [1] * 410
    segment.response_length += 410
    segment.summary_span = (len(segment.tokens) - 410, len(segment.tokens))
    config = resolve_memory_config("attention_matching", {})
    assert config["query_count"] == 0 and config["queries"] == "summary"
    ids = torch.tensor([segment.tokens])
    with kv_compact.capture_queries(model) as full:
        model(input_ids=ids)
    with kv_compact.segment_queries(model, segment, config) as sampled:
        model(input_ids=ids)
    for idx in full:
        start = segment.summary_span[0]
        torch.testing.assert_close(sampled[idx], full[idx][:, start:])


def test_summary_mode_never_probes_the_next_segment(model, head, monkeypatch):  # noqa: F811
    def forbidden(*args, **kwargs):
        raise AssertionError("summary queries must come from the existing critic forward")

    monkeypatch.setattr(kv_compact, "probe_queries", forbidden)
    collect(model, head, make_segments(2, 3), "attention_matching", None, train=True)


@torch.no_grad()
def test_summary_cache_suffix_is_preserved_and_excluded_from_key_selection(model):  # noqa: F811
    segment = make_segments(0, 1)[0]
    cache, _ = kv_compact.prefill_with_queries(model, torch.tensor([segment.tokens]))
    start = segment.summary_span[0]
    before = copy.deepcopy(cache)
    with kv_compact.segment_queries(model, segment, resolve_memory_config("attention_matching", None)) as queries:
        model(input_ids=torch.tensor([segment.tokens]))
    biases = kv_compact.compact_cache(model, cache, queries, 2, method="am_ridge", end=start)
    for idx, _ in kv_compact._attention_layers(model):
        layer, original = cache.layers[idx], before.layers[idx]
        assert layer.keys.shape[-2] == 2 + len(segment.tokens) - start
        assert torch.equal(layer.keys[:, :, 2:], original.keys[:, :, start:])
        assert torch.equal(layer.values[:, :, 2:], original.values[:, :, start:])
        assert not biases[idx][:, 2:].any()
        for h in range(layer.keys.shape[1]):
            for key in layer.keys[0, h, :2]:
                assert (original.keys[0, h, :start] == key).all(-1).any()


def test_missing_summary_metadata_fails_before_model_forward(model, head, monkeypatch):  # noqa: F811
    segments = make_segments(1, 2)
    segments[0].summary_span = None

    def forbidden(*args, **kwargs):
        raise AssertionError("validation must run before the model")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="no actor summary_span"):
        episode_values(model, head, segments, "attention_matching", "cpu")


@torch.no_grad()
def test_hindsight_ablation_forwards_only_the_prefix(model):  # noqa: F811
    prefix, suffix = [torch.tensor([s.tokens]) for s in make_segments(1, 2)]
    cache, _ = kv_compact.prefill_with_queries(model, prefix)
    received = []

    def capture(module, args, kwargs):
        received.append(kwargs["input_ids"].clone())

    handle = model.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        kv_compact.probe_queries(model, cache, suffix, prefix.shape[1], query_count=4)
    finally:
        handle.remove()
    assert len(received) == 1 and torch.equal(received[0], suffix[:, :4])


@torch.no_grad()
def test_keep_all_matches_full_detach_even_with_one_probe_position(model, head):  # noqa: F811
    segments = make_segments(7, 3)
    for segment in segments:
        start, _ = segment.summary_span
        segment.summary_span = (start, start + 1)
    actual = episode_values(model, head, segments, "attention_matching", "cpu", kv={"ratio": 1.0})
    expected = episode_values(model, head, segments, "full_detach", "cpu")
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_chunked_attention_matches_dense_outputs_and_gradients():
    torch.manual_seed(0)
    new, memory, groups = 263, 17, 2  # spans three query chunks
    q = torch.randn(1, 4, new, 8, requires_grad=True)
    k = torch.randn(1, 2, memory + new, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    beta = torch.randn(2, memory)
    module = SimpleNamespace(_compaction_bias=beta, num_key_value_groups=groups)
    actual, _ = _biased_sdpa(module, q, k, v, None, scaling=0.25)
    mask = torch.zeros(1, 4, new, memory + new)
    mask[..., :memory] = beta.repeat_interleave(groups, 0)[None, :, None]
    future = torch.arange(memory + new)[None] > memory + torch.arange(new)[:, None]
    mask.masked_fill_(future[None, None], float("-inf"))
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, scale=0.25).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v))
    expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_repeated_keys_preserve_mass_and_values_with_a_short_probe(model):  # noqa: F811
    cache, _ = kv_compact.prefill_with_queries(model, torch.tensor([make_segments(4, 1)[0].tokens]))
    queries = {}
    originals = {}
    for idx, attn in kv_compact._attention_layers(model):
        layer = cache.layers[idx]
        layer.keys = layer.keys[:, :, :1].expand(-1, -1, 12, -1).clone()
        layer.values = torch.randn_like(layer.keys)
        originals[idx] = (layer.keys.clone(), layer.values.clone())
        queries[idx] = torch.randn(4, 1, attn.head_dim)  # two observations per KV head, three unknown values
    biases = kv_compact.compact_cache(model, cache, queries, 3, method="am_ridge")
    for idx, beta in biases.items():
        original_k, original_v = originals[idx]
        layer = cache.layers[idx]
        assert layer.keys.shape[-2] == 3 and torch.isfinite(layer.values).all()
        # All keys are identical: sum(exp(beta)) should still represent all 12 originals.
        torch.testing.assert_close(beta.exp().sum(-1), torch.full((2,), 12.0), atol=1e-3, rtol=1e-3)
        for group in range(2):
            q = queries[idx][group * 2 : (group + 1) * 2].reshape(-1, original_k.shape[-1])
            raw = q @ original_k[0, group].T / original_k.shape[-1] ** 0.5
            compact = q @ layer.keys[0, group].T / original_k.shape[-1] ** 0.5 + beta[group]
            torch.testing.assert_close(compact.logsumexp(-1), raw.logsumexp(-1), atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                compact.softmax(-1) @ layer.values[0, group],
                raw.softmax(-1) @ original_v[0, group],
                atol=2e-3,
                rtol=2e-3,
            )


def test_attention_context_restores_model_on_exception(model):  # noqa: F811
    previous = model.config._attn_implementation
    biases = {i: torch.zeros(2, 3) for i, _ in kv_compact._attention_layers(model)}
    with pytest.raises(RuntimeError, match="probe failure"), kv_compact.attention_bias(model, biases):
        raise RuntimeError("probe failure")
    assert model.config._attn_implementation == previous
    assert all(not hasattr(attn, "_compaction_bias") for _, attn in kv_compact._attention_layers(model))


def test_bfloat16_autocast_training_matches_value_pass(model, head):  # noqa: F811
    segments = make_segments(8, 3)
    config = {}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        predicted, _ = collect(model, head, segments, "attention_matching", config, train=False)
        trained, grads = collect(model, head, segments, "attention_matching", config, train=True)
    for a, b in zip(predicted, trained, strict=True):
        torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)
    assert grads and all(torch.isfinite(g).all() for g in grads.values())


@torch.no_grad()
def test_compression_keeps_earlier_memory_blocks_and_biases(model, head, monkeypatch):  # noqa: F811
    compact = kv_compact.compact_cache
    checked = []

    def check(*args, **kwargs):
        cache, start = args[1], kwargs["start"]
        old = {
            i: (cache.layers[i].keys[:, :, :start].clone(), cache.layers[i].values[:, :, :start].clone())
            for i, _ in kv_compact._attention_layers(model)
        }
        result = compact(*args, **kwargs)
        for i, (keys, values) in old.items():
            assert torch.equal(cache.layers[i].keys[:, :, :start], keys)
            assert torch.equal(cache.layers[i].values[:, :, :start], values)
            if start:
                assert torch.equal(result[i][:, :start], kwargs["prev_biases"][i])
        checked.append(start)
        return result

    monkeypatch.setattr(kv_compact, "compact_cache", check)
    episode_values(model, head, make_segments(11, 3), "attention_matching", "cpu")
    assert len(checked) == 2 and checked[0] == 0 and checked[1] > 0
