"""Frozen-checkpoint AM evaluation preserves cohort, tokens and checkpoint weights."""

import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from agentic_critic.critic_modes import episode_values
from agentic_critic.fcs_env import Episode, Event
from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.frozen_am_figures import load_results
from agentic_critic_pipeline.frozen_am_validation import _checkpoints, predict, summary_segments
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_runtime.fcs_env import Episode as RuntimeEpisode
from agentic_critic_runtime.miles_fcs import episode_samples
from tests.agentic_critic.test_critic_modes import head, model  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def fixture_episode(index=0):
    # Token IDs are the only source of truth. Summary text deliberately does not match.
    events = [
        dict(tokens=[1, 2, 3], actor=False, segment=0, kind="task"),
        dict(tokens=[4, 5, 6, 7] * 3, actor=True, segment=0, kind="action"),
        dict(tokens=[8, 9], actor=False, segment=0, kind="compact_request"),
        dict(tokens=[10, 11, 12, 13], actor=True, segment=0, kind="summary"),
        dict(tokens=[14, 15], actor=False, segment=1, kind="resume"),
        dict(tokens=[16, 17, 18, 19], actor=True, segment=1, kind="action"),
    ]
    original = Episode(events=[Event(**e) for e in events], n_compactions=1)
    return dict(
        episode_id=f"validation:{index}",
        problem_id=f"frontiersmith:{index}",
        split="validation",
        reward=float(index),
        n_compactions=1,
        events=events,
        summary_text="not the token representation",
        segments=[
            dict(input_ids=s.tokens, response_start=s.response_start, loss_mask=s.loss_mask)
            for s in original.to_segments()
        ],
    )


def test_summary_spans_use_exact_events_and_reject_mismatch():
    episode = fixture_episode()
    segments = summary_segments(episode)
    start, end = segments[0].summary_span
    assert segments[0].tokens[start:end] == [10, 11, 12, 13]
    episode["events"][3]["tokens"][0] = 44
    with pytest.raises(ValueError, match="event tokens/masks"):
        summary_segments(episode)


def test_no_text_fallback_for_missing_summary():
    episode = fixture_episode()
    del episode["events"]
    with pytest.raises(ValueError, match="no actor summary_span"):
        summary_segments(episode)


def test_runtime_export_keeps_original_summary_span():
    source = fixture_episode()
    episode = RuntimeEpisode(events=[Event(**e) for e in source["events"]], n_compactions=1)
    logprobs = [[-0.5] * len(e.tokens) for e in episode.events if e.actor]
    fields = episode_samples(episode, logprobs)
    start, end = fields[0]["metadata"]["summary_span"]
    assert fields[0]["tokens"][start:end] == source["events"][3]["tokens"]
    assert fields[1]["metadata"]["summary_span"] is None


@torch.no_grad()
def test_frozen_predictions_align_and_keep_all_matches_full(model, head):  # noqa: F811
    segments = summary_segments(fixture_episode())
    expected = episode_values(model, head, segments, "full_detach", "cpu")
    actual = predict(model, head, segments, "attention_matching", "cpu", ratio=1)
    full = predict(model, head, segments, "full_history", "cpu", chunk=128)
    for s, a, b, c in zip(segments, actual, expected, full, strict=True):
        torch.testing.assert_close(a, b[torch.tensor(s.loss_mask).bool()], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-5)
        assert not a.requires_grad


def test_native_iterations_map_to_completed_batches_and_require_full_history(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    for iteration in (4, 9, 14):
        (root / f"iter_{iteration:07d}").mkdir()
    (root / "latest_checkpointed_iteration.txt").write_text("9")
    run = dict(mode="full_history", validation_plan_hash="hash", checkpoints=str(root), logical_batches=10)
    (tmp_path / "miles_run.json").write_text(json.dumps(run))
    args = SimpleNamespace(training_output=tmp_path, out=tmp_path / "eval")
    assert [r["batch"] for r in _checkpoints(args, {"plan_hash": "hash"})] == [5, 10]
    run["mode"] = "actor_summary"
    (tmp_path / "miles_run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="full_history"):
        _checkpoints(args, {"plan_hash": "hash"})


def test_frozen_cli_three_modes_and_figures(tmp_path, model, head):  # noqa: F811
    base, checkpoint = tmp_path / "base", tmp_path / "checkpoint"
    model.save_pretrained(base)
    checkpoint.mkdir()
    state = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("model.")}
    safetensors.torch.save_file(state, str(checkpoint / "critic.safetensors"))
    torch.save(dict(weight=head.linear.weight.detach(), bias=head.linear.bias.detach()), checkpoint / "value_head.pt")
    (checkpoint / "meta.json").write_text(json.dumps(dict(converted_parameters=len(state))))
    before = {p.name: data_sha256(p) for p in checkpoint.iterdir()}
    source = tmp_path / "validation.jsonl"
    episodes = [fixture_episode(i) for i in range(2)]
    source.write_text("".join(json.dumps(e) + "\n" for e in episodes))
    plan = dict(
        schema_version=1,
        source=str(source),
        source_sha256=data_sha256(source),
        selected_episode_ids=[e["episode_id"] for e in episodes],
        baseline_tokens=1024,
        profile_span=3076,
        episodes={
            e["episode_id"]: dict(
                problem_id=e["problem_id"],
                reward=e["reward"],
                n_compactions=1,
                segments=[
                    dict(
                        tokens=sum(s["loss_mask"]),
                        positions_sha256=digest([i for i, bit in enumerate(s["loss_mask"]) if bit]),
                    )
                    for s in e["segments"]
                ],
            )
            for e in episodes
        },
    )
    plan["plan_hash"] = digest(plan)
    plan_path = tmp_path / "validation_plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_critic_pipeline.frozen_am_validation",
            "--base-model",
            str(base),
            "--plan",
            str(plan_path),
            "--checkpoint",
            f"5={checkpoint}",
            "--out",
            str(output),
            "--dtype",
            "float32",
            "--chunk",
            "128",
            "--workers",
            "2",
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-6000:]
    assert before == {p.name: data_sha256(p) for p in checkpoint.iterdir()}
    _, run, results = load_results(output)
    assert run["status"] == "complete" and len(results) == 3
    assert all(r[0]["episodes"] == 2 and r[0]["timing"]["sum_rank_seconds"] > 0 for r in results.values())
    report = output / "validation_report"
    assert (report / "validation_ev.png").is_file() and (report / "evaluation_cost.pdf").is_file()
    assert len(list((report / "trajectories").glob("*.png"))) == 2
    with gzip.open(output / "attention_matching/validation/batch_0005/values.jsonl.gz", "rt") as stream:
        rows = list(map(json.loads, stream))
    assert {r["episode_id"] for r in rows} == set(plan["episodes"])
    # Changing the stored checkpoint identity must make comparisons fail, not silently mix weights.
    metrics = output / "attention_matching/validation/batch_0005/metrics.json"
    bad = copy.deepcopy(results["attention_matching"][0])
    bad["checkpoint_sha256"] = {}
    metrics.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Unpaired"):
        load_results(output)
