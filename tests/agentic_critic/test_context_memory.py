"""Contract tests for native Miles problem / memory / original resume inputs."""

import ast
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from agentic_critic.context_layout import ContextSegment, episode_layout, validate_packed_layout
from agentic_critic.context_memory import Memory, fit_memory, packed_attention, segment_attention, shift_rotary
from agentic_critic.critic_input import Segment
from agentic_critic.miles_context import prepare_context_samples


def segments():
    # 90/91 are continuation instruction + assistant-start, not response tokens.
    return [
        Segment([1, 2, 3, 10, 11, 20, 21, 22, 23], 6, [1] * 6, (6, 9)),
        Segment([1, 2, 3, 30, 21, 22, 90, 91, 40, 41, 50, 51], 4, [1] * 4, (10, 12)),
        Segment([1, 2, 3, 30, 50, 51, 90, 91, 60, 61], 2, [1, 1]),
    ]


def tensors(length):
    return torch.randn(4, length, 8), torch.randn(2, length, 8), torch.randn(2, length, 8)


def dense(q, k, v):
    return F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]


def test_layout_preserves_exact_actor_resume_instruction_and_targets():
    original = segments()
    before = [asdict(s) for s in original]
    layouts = episode_layout(original, episode="e")
    assert [s.problem_prefix for s in layouts] == [3, 3, 3]
    assert original[1].tokens[layouts[1].problem_prefix : original[1].response_start] == [30, 21, 22, 90, 91]
    assert [asdict(s) for s in original] == before
    assert validate_packed_layout([s.to_dict() for s in layouts], [len(s.tokens) for s in original]) == layouts
    with pytest.raises(ValueError, match="complete, ordered"):
        validate_packed_layout([s.to_dict() for s in layouts[1:]], [s.length for s in layouts[1:]])


def test_no_boundary_is_identical_to_shared_attention():
    torch.manual_seed(2)
    layout = ContextSegment("e", 0, 1, 13, 4, None)
    q, k, v = tensors(13)
    actual, fit = segment_attention(q, k, v, layout, None, rotary_dim=4, base=10000.0)
    torch.testing.assert_close(actual, dense(q, k, v), atol=1e-6, rtol=1e-6)
    assert fit is None


def test_problem_and_generated_summary_are_not_persisted_as_raw_kv():
    torch.manual_seed(3)
    layout = episode_layout(segments(), episode="e")[0]
    q, k, v = tensors(layout.length)
    _, fit = segment_attention(q, k, v, layout, None, rotary_dim=4, base=10000.0)
    memory = fit_memory(fit, ratio=1.0)
    a, b = layout.summary_span
    assert torch.equal(memory.keys, k[:, layout.problem_prefix : a])
    assert torch.equal(memory.values, v[:, layout.problem_prefix : a])
    assert fit[3].shape[1] == b - a  # Original summary is a probe, not raw persistent memory.
    assert memory.logical_end == layout.length


def test_fresh_problem_does_not_attend_to_old_memory():
    torch.manual_seed(4)
    layout = episode_layout(segments(), episode="e")[1]
    q, k, v = tensors(layout.length)
    old_k, old_v = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    mem = Memory(old_k, old_v, torch.randn(2, 5), 19)
    changed = Memory(old_k + 4, old_v - 4, mem.bias, 19)
    out, _ = segment_attention(q, k, v, layout, mem, rotary_dim=4, base=10000.0)
    other, _ = segment_attention(q, k, v, layout, changed, rotary_dim=4, base=10000.0)
    p = layout.problem_prefix
    torch.testing.assert_close(out[:, :p], dense(q[:, :p], k[:, :p], v[:, :p]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out[:, :p], other[:, :p])
    assert not torch.allclose(out[:, p:], other[:, p:])


def test_common_rope_offset_preserves_actor_segment_internal_attention():
    torch.manual_seed(5)
    q, k, v = tensors(17)
    shifted_q = shift_rotary(q, offset=8000, rotary_dim=4, base=10000.0)
    shifted_k = shift_rotary(k, offset=8000, rotary_dim=4, base=10000.0)
    torch.testing.assert_close(dense(shifted_q, shifted_k, v), dense(q, k, v), atol=1e-6, rtol=1e-6)


def test_repeated_compaction_keeps_only_history_and_current_resume_once():
    torch.manual_seed(6)
    layouts = episode_layout(segments(), episode="e")
    memory = None
    lengths = []
    for layout in layouts:
        q, k, v = tensors(layout.length)
        previous = None if memory is None else memory.keys.clone()
        _, fit = segment_attention(q, k, v, layout, memory, rotary_dim=4, base=10000.0)
        if fit is not None:
            expected_source = (0 if previous is None else previous.shape[1]) + layout.summary_span[0] - layout.problem_prefix
            assert fit[0].shape[1] == expected_source
            memory = fit_memory(fit, ratio=0.5)
            lengths.append(memory.keys.shape[1])
            assert memory.keys.shape[1] == max(1, round(0.5 * expected_source))
            assert torch.isfinite(memory.values).all() and torch.isfinite(memory.bias).all()
        if previous is not None:
            # segment_attention never mutates its incoming memory.
            assert fit is None or torch.equal(fit[0][:, : previous.shape[1]], previous)
    assert len(lengths) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_recompute_matches_forward_and_gradients_with_real_fits(dtype):
    torch.manual_seed(7)
    layouts = episode_layout(segments(), episode="e")
    boundaries = [0]
    for s in layouts:
        boundaries.append(boundaries[-1] + s.length)
    n = boundaries[-1] + 3
    q = torch.randn(n, 4, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(n, 2, 8, dtype=dtype, requires_grad=True)
    v = torch.randn(n, 2, 8, dtype=dtype, requires_grad=True)

    def run(q, k, v):
        return packed_attention(q, k, v, layouts, boundaries + [n], ratio=0.5, rotary_dim=4, base=10000.0)

    out = run(q, k, v)
    gradients = torch.autograd.grad(out.square().sum(), (q, k, v))
    repeated = checkpoint(run, q, k, v, use_reentrant=False)
    repeated_gradients = torch.autograd.grad(repeated.square().sum(), (q, k, v))
    torch.testing.assert_close(out, repeated)
    for a, b in zip(gradients, repeated_gradients, strict=True):
        torch.testing.assert_close(a, b)
    assert all(torch.isfinite(g).all() for g in gradients)
    # A loss only on the final segment cannot backprop through detached AM history.
    out = run(q, k, v)
    grad = torch.autograd.grad(out[boundaries[-2] : boundaries[-1]].square().sum(), v)[0]
    assert not grad[: boundaries[-2]].any()


def test_miles_metadata_does_not_rewrite_actor_samples():
    original = segments()
    samples = [
        SimpleNamespace(
            tokens=list(s.tokens),
            response_length=s.response_length,
            loss_mask=list(s.loss_mask),
            reward=0.25,
            rollout_id=17,
            metadata={"segment": i, "n_compactions": 2, "summary_span": s.summary_span},
            train_metadata=None,
        )
        for i, s in enumerate(original)
    ]
    before = [(s.tokens[:], s.loss_mask[:], s.response_length, s.reward) for s in samples]
    prepare_context_samples(samples)
    assert [(s.tokens, s.loss_mask, s.response_length, s.reward) for s in samples] == before
    assert [s.train_metadata["critic_context"]["index"] for s in samples] == [0, 1, 2]
    samples[0].metadata.pop("summary_span")
    with pytest.raises(ValueError, match="original episodes.jsonl"):
        prepare_context_samples(samples)


def test_hybrid_reference_resets_linear_states_and_retains_instruction():
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    from agentic_critic.context_reference import read_reference

    torch.manual_seed(8)
    cfg = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        vocab_size=128,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").eval()
    backbone = model.model
    head = torch.nn.Linear(32, 1)
    original = segments()
    captured, hooks = [], []
    for layer in backbone.layers[:3]:

        def check(module, args, kwargs):
            assert kwargs.get("cache_params") is None
            captured.append(True)

        hooks.append(layer.linear_attn.register_forward_pre_hook(check, with_kwargs=True))
    values = []
    try:
        read_reference(model, lambda h: head(h).squeeze(-1), original, "cpu", lambda pairs: values.extend(v for _, v in pairs), ratio=0.5)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == 3 * len(original)
    assert [len(v) for v in values] == [s.response_length for s in original]
    assert all(torch.isfinite(v).all() for v in values)
    with torch.no_grad():
        shared = backbone(input_ids=torch.tensor([original[0].tokens]), use_cache=False).last_hidden_state[0]
        p = original[0].response_start - 1
        expected = head(shared[p : p + original[0].response_length]).squeeze(-1)
    torch.testing.assert_close(values[0], expected, atol=1e-5, rtol=1e-5)


def test_legacy_offsets_recovered_only_from_matching_original_events(tmp_path):
    original = segments()
    events = []
    samples = []
    for i, s in enumerate(original):
        events.append(dict(tokens=s.tokens[: s.response_start], actor=False, segment=i, kind="task" if i == 0 else "resume"))
        end = s.summary_span[0] if s.summary_span else len(s.tokens)
        events.append(dict(tokens=s.tokens[s.response_start : end], actor=True, segment=i, kind="action"))
        if s.summary_span:
            events.append(dict(tokens=s.tokens[end:], actor=True, segment=i, kind="summary"))
        samples.append(SimpleNamespace(tokens=list(s.tokens), response_length=s.response_length, loss_mask=list(s.loss_mask), rollout_id=2, metadata={"segment": i, "sample": "original"}, train_metadata=None))
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps(dict(sample="original", events=events, n_compactions=2)) + "\n")
    prepare_context_samples(samples, events=source)
    assert samples[0].train_metadata["critic_context"]["summary_span"] == (6, 9)
    samples[1].tokens[-1] = 99
    with pytest.raises(ValueError, match="tokens/masks"):
        prepare_context_samples(samples, events=source)


def _definition(path, name, scope):
    # Exercise actual GPU-wrapper ingress without importing unavailable TE/Megatron.
    source = Path(__file__).resolve().parents[2] / path
    node = next(n for n in ast.parse(source.read_text()).body if getattr(n, "name", None) == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), scope)
    return scope[name]


def test_native_packed_metadata_is_enabled_only_for_critic_calls():
    pack = _definition(
        "miles/backends/megatron_utils/parallel.py",
        "get_packed_seq_params",
        {
            "PackedSeqParamsWithHostCuSeqlens": SimpleNamespace,
        },
    )
    original = segments()
    layouts = episode_layout(original, episode="native")
    bounds = [0]
    for s in original:
        bounds.append(bounds[-1] + len(s.tokens))
    batch = dict(cu_seqlens=torch.tensor(bounds), cu_seqlens_host=tuple(bounds), max_seqlen=12, unconcat_tokens=[torch.tensor(s.tokens) for s in original], metadata=[dict(critic_context=s.to_dict()) for s in layouts])
    args = SimpleNamespace(qkv_format="thd", critic_context_mode="attention_matching", critic_am_ratio=0.1)
    assert not hasattr(pack(batch, args), "critic_context")
    critic = pack(batch, args, critic_context=True)
    assert critic.critic_context["layouts"] == tuple(layouts)
    batch["metadata"] = None
    with pytest.raises(ValueError, match="missing"):
        pack(batch, args, critic_context=True)


def test_native_core_preserves_actor_fallback_and_accepts_te_call_signature():
    class Base(torch.nn.Module):
        def __init__(self, config, *args, **kwargs):
            super().__init__()
            self.calls = 0

        def forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
            self.calls += 1
            return "original TE actor kernel"

    core_type = _definition(
        "miles_plugins/models/critic_memory_attention.py",
        "CriticMemoryAttention",
        {
            "TEDotProductAttention": Base,
            "packed_attention": packed_attention,
        },
    )
    core = core_type(None, memory_rotary_dim=4, memory_rotary_base=10000.0)
    q, k, v = tensors(13)
    q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
    assert core(q, k, v, None, "causal") == "original TE actor kernel"
    layout = ContextSegment("one", 0, 1, 13, 4, None)
    packed = SimpleNamespace(critic_context={"layouts": (layout,), "ratio": 0.1}, cu_seqlens_host=(0, 13))
    result = core(q, k, v, None, "causal", packed_seq_params=packed)
    assert result.shape == (13, 32) and core.calls == 1


def test_hf_optimizer_refuses_am_instead_of_silently_training_another_backend():
    validate = _definition("agentic_critic_pipeline/train_am_comparison.py", "_validate_args", {})
    with pytest.raises(ValueError, match="native Miles"):
        validate(SimpleNamespace(modes=["attention_matching"]))


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_am_sharding_transports_exact_layouts_with_complete_episodes(dp_size):
    scope = {}
    _definition("miles/ray/rollout/train_data_conversion.py", "_package_shards", scope)
    shard = _definition("miles/ray/rollout/train_data_conversion.py", "split_episode_aligned_shards", scope)
    samples = []
    for episode in range(8):
        for segment, layout in zip(segments(), episode_layout(segments(), episode=episode), strict=True):
            samples.append((segment, layout))
    data = {
        "tokens": [s.tokens for s, _ in samples],
        "metadata": [{"critic_context": layout.to_dict()} for _, layout in samples],
        "rollout_ids": [layout.episode for _, layout in samples],
    }
    args = SimpleNamespace(global_batch_size=8, balance_data=False)
    ranks = shard(args, data, dp_size=dp_size)
    assert sorted(i for rank in ranks for i in rank["partition"]) == list(range(len(samples)))
    for rank in ranks:
        assert rank["metadata"] == [data["metadata"][i] for i in rank["partition"]]
        for indices in rank["micro_batch_indices"]:
            layouts = validate_packed_layout(
                [rank["metadata"][i]["critic_context"] for i in indices],
                [len(rank["tokens"][i]) for i in indices],
            )
            assert [layout.index for layout in layouts] == [0, 1, 2]


def test_am_optimizer_schedule_shuffles_episodes_without_splitting_segments():
    parallel = SimpleNamespace(effective_dp=SimpleNamespace(size=2, rank=0), cp=SimpleNamespace(size=1), vpp_size=1)
    scope = {"torch": torch, "get_parallel_state": lambda: parallel}
    _definition("miles/backends/training_utils/data/rollout.py", "_shuffled_step_indices", scope)
    _definition("miles/backends/training_utils/data/rollout.py", "_episode_step_groups", scope)
    _definition("miles/backends/training_utils/data/rollout.py", "DataIterator", scope)
    schedule = _definition("miles/backends/training_utils/data/rollout.py", "get_critic_train_schedule", scope)
    data = {"rollout_ids": [0, 0, 1, 1, 1, 2, 3, 3], "total_lengths": [5] * 8}
    args = SimpleNamespace(critic_context_mode="attention_matching", critic_num_steps_per_rollout=2,
                           seed=42, use_dynamic_batch_size=True, max_tokens_per_gpu=30)
    iterators, counts, rollouts = schedule(args, data, rollout_id=3, num_rollouts=[8])
    assert counts == [2, 2] and rollouts == [4, 4]
    indices = iterators[0].micro_batch_indices
    assert sorted(i for group in indices for i in group) == list(range(8))
    for group in indices:
        episode = data["rollout_ids"][group[0]]
        assert group == [i for i, identity in enumerate(data["rollout_ids"]) if identity == episode]
    args.max_tokens_per_gpu = 10
    with pytest.raises(ValueError, match="complete episode"):
        schedule(args, data, rollout_id=3, num_rollouts=[8])
