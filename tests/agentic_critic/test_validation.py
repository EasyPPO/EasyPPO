"""Frozen-cohort, full/segment alignment and compaction-metric regressions."""

import io
import json
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agentic_critic_pipeline.cohort import balanced_selection
from agentic_critic_pipeline.data import digest, training_quotas
from agentic_critic_pipeline.records import batch_schedule
from agentic_critic_pipeline.validation import _chunk_values, load_plan, prepare
from agentic_critic_pipeline.validation_metrics import ValidationMetrics, _metrics, _moments
from agentic_critic_pipeline.validation_storage import compact_episode, compact_rollout


def test_128_problem_quotas_work_with_unchanged_training_schedule():
    problems = [{"problem_id": f"frontiersmith:{i}"} for i in range(128)]
    quotas = training_quotas(problems, 1234)
    assert Counter(quotas.values()) == {208: 64, 192: 64}
    episodes = [dict(problem_id=p, episode_id=f"{p}:{i}") for p, n in quotas.items() for i in range(n)]
    selected, unused, _ = balanced_selection(episodes, 32, 16, 1234)
    assert len(selected) == 25600 and not unused
    schedule = batch_schedule(selected, 32, 16, 1, 1234, False)
    assert len(schedule) == 50
    for batch in schedule:
        assert len(batch) == 512
        assert Counter(Counter(r["problem_id"] for r in batch).values()) == {16: 32}
    assert len({r["episode_id"] for batch in schedule for r in batch}) == 25600


def _episode(identity="validation:0", *, split="validation", problem="frontiersmith:0", lengths=(1024, 32)):
    events, segments = [], []
    for index, length in enumerate(lengths):
        prompt, response = [1, 2], [3] * length
        events.extend(
            [
                dict(tokens=prompt, actor=False, segment=index, kind="task" if index == 0 else "resume"),
                dict(tokens=response, actor=True, segment=index, kind="action"),
            ]
        )
        segments.append(dict(input_ids=prompt + response, response_start=2, loss_mask=[1] * length))
    return dict(
        episode_id=identity,
        problem_id=problem,
        split=split,
        reward=0.5,
        scores=[50.0],
        submission_segments=[0],
        finish_reason="max_turns",
        n_compactions=len(lengths) - 1,
        truncated_turns=0,
        events=events,
        segments=segments,
    )


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_selection_is_frozen_before_predictions_and_checks_exact_group(tmp_path):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    _write(train, [_episode("train:0", split="train", lengths=(2,))])
    _write(val, [_episode(f"validation:{i}", lengths=(2,)) for i in range(32)])
    plan = prepare(val, train, tmp_path / "replay")
    assert len(plan["selected_episode_ids"]) == 20
    assert plan["selected_episode_ids"] == random.Random(1234).sample(sorted(plan["episodes"]), 20)
    assert load_plan(tmp_path / "replay/validation_plan.json", verify_files=True) == plan
    # A complete source cohort is fixed; no undersized validation groups are accepted.
    _write(val, [_episode(f"validation:{i}", lengths=(2,)) for i in range(31)])
    with pytest.raises(ValueError, match="exact per-prompt"):
        prepare(val, train, tmp_path / "bad")
    with pytest.raises(ValueError, match="source changed"):
        load_plan(tmp_path / "replay/validation_plan.json", verify_files=True)


def _prediction(lengths=(4000, 4000), values=(0.2, 0.7)):
    return dict(
        episode_id="e",
        problem_id="frontiersmith:0",
        reward=1.0,
        n_compactions=len(lengths) - 1,
        segments=[
            dict(segment_index=i, values=[v] * length, response_positions=list(range(length)))
            for i, (length, v) in enumerate(zip(lengths, values, strict=True))
        ],
    )


def test_profile_baseline_1024_offsets_3076_and_complete_trace():
    plan = dict(selected_episode_ids=["e"], baseline_tokens=1024, profile_span=3076)
    stats = ValidationMetrics(plan, mode="full_history", batch=5)
    events, traces = io.StringIO(), io.StringIO()
    stats.add(_prediction(), events=events, traces=traces)
    event = json.loads(events.getvalue())
    assert event["baseline"] == pytest.approx(0.2)
    assert event["offsets"] == list(range(-3076, 3076))
    assert event["l1_from_baseline"][:3076] == pytest.approx([0.0] * 3076)
    assert event["l1_from_baseline"][3076:] == pytest.approx([0.5] * 3076)
    assert event["first_token_jump"] == pytest.approx(0.5)
    assert event["mean_shift"] == pytest.approx(0.5)
    trace = json.loads(traces.getvalue())
    assert len(trace["value_prediction"]) == len(trace["value_target"]) == 8000
    assert trace["compaction_boundaries"] == [4000]
    assert stats.summary()["ev"]["all"]["ev"] is None  # one constant return
    assert stats.summary()["jumps"]["total/1/first_token"]["mean"] == pytest.approx(0.5)


def test_short_baselines_are_excluded_and_profiles_do_not_pad_missing_tokens():
    plan = dict(selected_episode_ids=[], baseline_tokens=1024, profile_span=3076)
    stats = ValidationMetrics(plan, mode="actor_summary", batch=10)
    events = io.StringIO()
    stats.add(_prediction((100, 100), (0.1, 0.2)), events=events, traces=io.StringIO())
    assert json.loads(events.getvalue())["eligible"] is False
    assert stats.summary()["profiles"] == {}
    stats.add(_prediction((1024, 2), (0.1, 0.2)), events=io.StringIO(), traces=io.StringIO())
    profile = stats.summary()["profiles"]["all"]
    assert profile["mean_l1"][0] is None
    assert profile["event_counts"][-1] == 0
    assert profile["mean_l1"][3076:3078] == pytest.approx([0.1, 0.1])
    assert "all/mean_shift_1024" not in stats.summary()["jumps"]


def test_ev_is_computed_globally_not_averaged_over_constant_prompt_groups():
    predictions = [np.array([0.1, 0.2]), np.array([0.8, 0.9, 0.7])]
    targets = [0.0, 1.0]
    moment = sum((_moments(v, y) for v, y in zip(predictions, targets, strict=True)), np.zeros(6))
    v, g = np.concatenate(predictions), np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    assert _metrics(moment)["ev"] == pytest.approx(1 - np.var(g - v) / np.var(g))


def test_online_summary_matches_offline_reconstruction_without_collecting_curves():
    plan = dict(selected_episode_ids=[], baseline_tokens=1024, profile_span=3076)
    online = ValidationMetrics(plan, mode="full_history", batch=5, collect_profiles=False)
    offline = ValidationMetrics(plan, mode="full_history", batch=5)
    for prediction in (_prediction(), _prediction((100, 100), (0.1, 0.2)), _prediction((1024, 2), (0.3, 0.9))):
        online.add(prediction)
        offline.add(prediction)
    online_summary, offline_summary = online.summary(), offline.summary()
    assert online_summary.pop("profiles") == {}
    assert offline_summary.pop("profiles")
    assert online_summary == offline_summary


@pytest.mark.parametrize("mode", ["actor_summary", "full_history"])
def test_native_dump_alignment_with_masked_feedback_and_duplicate_detection(tmp_path, mode):
    samples = [
        dict(
            index=10,
            rollout_id=7,
            response_length=3,
            loss_mask=[1, 0, 1],
            reward=0.5,
            critic_span=[0, 3],
            metadata=dict(sample="e", segment=0, problem="frontiersmith:0"),
        ),
        dict(
            index=11,
            rollout_id=7,
            response_length=2,
            loss_mask=[1, 1],
            reward=0.5,
            critic_span=[4, 2],
            metadata=dict(sample="e", segment=1, problem="frontiersmith:0"),
        ),
    ]
    torch.save(dict(samples=samples), tmp_path / "replay.pt")
    frozen = dict(
        problem_id="frontiersmith:0",
        reward=0.5,
        n_compactions=1,
        segments=[dict(tokens=2, positions_sha256=digest([0, 2])), dict(tokens=2, positions_sha256=digest([0, 1]))],
    )
    plan = dict(replay=[dict(path=str(tmp_path / "replay.pt"))], episodes={"e": frozen})
    dump = dict(rollout_id=5)
    if mode == "full_history":
        dump.update(
            value_slices=[(0, 0, 3), (0, 4, 2)],
            rollout_ids=[7],
            values=[torch.tensor([0.1, 99.0, 0.2, 98.0, 0.3, 0.4])],
            loss_masks=[[1, 0, 1, 0, 1, 1]],
        )
    else:
        dump.update(
            sample_indices=[10, 11],
            values=[torch.tensor([0.1, 99.0, 0.2]), torch.tensor([0.3, 0.4])],
            loss_masks=[[1, 0, 1], [1, 1]],
        )
    if mode != "full_history":
        dump["rollout_ids"] = [7, 7]
    dump["raw_reward"] = [0.5] * len(dump["values"])
    dump["response_lengths"] = [len(v) for v in dump["values"]]
    dump = dict(
        schema_version=2,
        value_bin_size=100,
        rollout_id=5,
        full_history=mode == "full_history",
        episodes=compact_rollout(dump),
    )
    torch.save(dump, tmp_path / "0.critic.pt")
    rows = list(_chunk_values(plan, 0, tmp_path, mode, 5))
    assert len(rows) == 1
    assert [v for s in rows[0]["segments"] for v in s["values"]] == pytest.approx([0.15, 0.35])
    torch.save(dump, tmp_path / "1.critic.pt")
    with pytest.raises(ValueError, match="Duplicated"):
        list(_chunk_values(plan, 0, tmp_path, mode, 5))


@pytest.mark.parametrize("mode", ["shared", "full_history"])
def test_validation_packs_keep_whole_episodes_across_eight_ranks(monkeypatch, mode):
    import ray

    from miles.ray.rollout import critic_validation
    from miles.utils.object_store import _RayStoreObjectRef
    from tests.fast.ray.rollout.test_critic_shards import _episode as samples_of

    samples = []
    for rollout in range(32):
        first, second, _ = samples_of(rollout, rollout * 100)
        first.group_index = second.group_index = 0
        samples.extend([first, second])
    args = SimpleNamespace(
        critic_context_mode=mode,
        global_batch_size=512,
        n_samples_per_prompt=16,
        rollout_batch_size=32,
        reward_key=None,
        advantage_estimator="ppo",
        rewards_normalization=False,
        critic_return_std_floor=0.075,
        rollout_top_logprobs_num=0,
        use_dynamic_global_batch_size=False,
        balance_data=True,
        filter_truncated_in_actor_loss=True,
    )
    original = vars(args).copy()
    shards = []

    def put(*, value, value_spec):
        shards.append(value)
        return _RayStoreObjectRef(payload=ray.ObjectRef.from_random())

    monkeypatch.setattr(critic_validation.object_store, "get_instance", lambda: SimpleNamespace(put=put))
    monkeypatch.setattr(critic_validation, "load_debug_rollout_data", lambda a, i: (samples, {}))
    pack = critic_validation.validation_pack(args, replay_path="validation.pt", group_size=32, parallel={"dp_size": 8})
    assert vars(args) == original
    assert len(pack.data_refs) == 8
    assert len({rid for shard in shards for rid in shard["rollout_ids"]}) == 32
    for shard in shards:
        assert len(set(shard["rollout_ids"])) == 4
        assert shard["num_microbatches"] == [4]
        assert shard["num_rollouts"] == [32]
        assert len(shard["tokens"]) == (4 if mode == "full_history" else 8)
        if mode == "full_history":
            assert len(shard["value_slices"]) == 8


def test_validation_driver_finishes_all_chunks_before_logging(monkeypatch, tmp_path):
    import asyncio

    from agentic_critic_pipeline import validation_driver

    plan = dict(replay=[dict(path="first.pt"), dict(path="second.pt")], group_size=32)
    calls = []
    packs = []

    async def get_pack(path, size):
        assert size == 32
        pack = object()
        packs.append(pack)
        calls.append(("get", path))
        return pack

    async def evaluate(**kwargs):
        assert kwargs["rollout_data_pack"] is packs[-1]
        assert kwargs["batch"] == 5
        calls.append(("evaluate", kwargs["batch"]))

    def finalize(plan, output, *, mode, batch):
        calls.append(("finalize", batch))
        return dict(ev={"all": {"ev": 0.8}}, jumps={})

    monkeypatch.setattr(validation_driver, "load_plan", lambda path: plan)
    monkeypatch.setattr(validation_driver, "finalize", finalize)
    monkeypatch.setattr(
        validation_driver, "remove_rollout_data_refs", lambda a, pack: calls.append(("free", packs.index(pack)))
    )
    monkeypatch.setattr(validation_driver.tracking, "define_step_key_metric_group", lambda *a: None)
    monkeypatch.setattr(
        validation_driver.tracking,
        "log",
        lambda a, metrics, **kw: calls.append(("log", metrics["critic_validation/batch"])),
    )
    args = SimpleNamespace(
        critic_validation_plan="plan", critic_validation_output=str(tmp_path), critic_context_mode="shared"
    )
    asyncio.run(
        validation_driver.evaluate(
            args,
            SimpleNamespace(evaluate_critic=evaluate),
            SimpleNamespace(get_critic_validation=get_pack),
            rollout_id=4,
        )
    )
    assert calls == [
        ("get", "first.pt"),
        ("evaluate", 5),
        ("free", 0),
        ("get", "second.pt"),
        ("evaluate", 5),
        ("free", 1),
        ("finalize", 5),
        ("log", 5),
    ]


def test_compact_values_preserve_exact_ev_jumps_and_segment_local_partial_bins():
    plan = dict(selected_episode_ids=["e"], baseline_tokens=1024, profile_span=3076)
    exact = ValidationMetrics(plan, mode="full_history", batch=5, collect_profiles=False)
    compact = ValidationMetrics(plan, mode="full_history", batch=5, collect_profiles=False)
    for reward in (0.0, 1.0):
        episode = _prediction((1103, 50003), (0.0, 0.0))
        episode["reward"] = reward
        for segment in episode["segments"]:
            segment["values"] = (np.arange(len(segment["values"])) % 17 / 17).tolist()
        binned = compact_episode(episode)
        exact.add(episode)
        traces = io.StringIO()
        compact.add(binned, traces=traces)
        assert [len(s["values"]) for s in binned["segments"]] == [12, 501]
        assert binned["segments"][0]["token_counts"] == [100] * 11 + [3]
        assert binned["segments"][1]["token_counts"] == [100] * 500 + [3]
        for original, reduced in zip(episode["segments"], binned["segments"], strict=True):
            assert reduced["values"] == pytest.approx(
                [np.mean(original["values"][i : i + 100]) for i in range(0, len(original["values"]), 100)]
            )
            assert "response_positions" not in reduced
        assert all("offsets" not in e and "l1_from_baseline" not in e for e in binned["boundaries"])
        trace = json.loads(traces.getvalue())
        assert len(trace["value_prediction"]) == len(trace["value_target"]) == 513
        assert trace["compaction_boundaries"] == [1103]
        assert trace["actor_token_ranges"][11:13] == [[1100, 1103], [1103, 1203]]
    actual, expected = compact.summary(), exact.summary()
    assert actual["jumps"] == expected["jumps"]
    for key in expected["ev"]:
        assert actual["ev"][key] == pytest.approx(expected["ev"][key])
    # Within-bin variation is retained in the sufficient statistics, not lost by averaging.
    assert actual["ev"]["all"]["mse"] > 0.3
