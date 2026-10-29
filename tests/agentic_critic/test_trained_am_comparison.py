"""Offline reuse and replicated HF full/shared/AM training end to end."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch

from agentic_critic.fcs_env import Episode, Event
from agentic_critic.miles_fcs import episode_samples
from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.trained_am_figures import load_results
from agentic_critic_pipeline.training_replay import index_events, load_batch, prepare_replay
from tests.agentic_critic.test_critic_modes import model  # noqa: F401
from tests.agentic_critic.test_frozen_am_validation import fixture_episode

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reuse_inputs(tmp_path_factory, model):  # noqa: F811
    root = tmp_path_factory.mktemp("trained-am")
    base, replay = root / "base", root / "replay"
    model.save_pretrained(base)
    replay.mkdir()
    records, samples = [], []
    for i, reward in enumerate((0.0, 0.8, 0.25, 1.0)):
        original = fixture_episode()
        original["events"][1]["tokens"] *= i + 1  # uneven rank token counts
        record = dict(
            episode_id=f"train:{i}",
            sample=f"train:{i}",
            problem=f"frontiersmith:{i // 2}",
            events=original["events"],
            scores=[100 * reward],
            n_compactions=1,
            finish_reason="max_turns",
        )
        records.append(record)
        ep = Episode(events=[Event(**e) for e in record["events"]], scores=record["scores"], n_compactions=1)
        fields = episode_samples(ep, [[0.0] * len(e.tokens) for e in ep.events if e.actor])
        for segment, f in enumerate(fields):
            samples.append(
                dict(
                    index=len(samples),
                    rollout_id=i,
                    tokens=f["tokens"],
                    response_length=f["response_length"],
                    loss_mask=f["loss_mask"],
                    reward=reward,
                    metadata=dict(problem=record["problem"], sample=record["sample"], segment=segment),
                )
            )
    source = replay / "episodes.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    torch.save(dict(samples=samples), replay / "rollout_0.pt")
    (replay / "manifest.json").write_text(
        json.dumps(
            dict(
                args=dict(problems_per_step=2, episodes_per_problem=2),
                steps=[dict(step=0, episodes=4, problems=["frontiersmith:0", "frontiersmith:1"])],
            )
        )
    )
    identities = dict(
        source_sha256=data_sha256(source),
        episodes=[dict(episode_id=r["episode_id"], sample=r["sample"], problem=r["problem"]) for r in records],
    )
    (replay / "episodes.identities.json").write_text(json.dumps(identities))
    validation = root / "validation.jsonl"
    val = [fixture_episode(i) for i in range(2)]
    validation.write_text("".join(json.dumps(e) + "\n" for e in val))
    plan = dict(
        schema_version=1,
        source=str(validation),
        source_sha256=data_sha256(validation),
        training_source_sha256=identities["source_sha256"],
        selected_episode_ids=[e["episode_id"] for e in val],
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
            for e in val
        },
    )
    plan["plan_hash"] = digest(plan)
    (root / "plan.json").write_text(json.dumps(plan))
    return root


def test_old_replay_restores_summary_spans_without_changing_tokens(reuse_inputs):
    root = reuse_inputs
    directory = root / "replay"
    plan = json.loads((root / "plan.json").read_text())
    spec = prepare_replay(directory, plan, steps=None, mini_batches=2, max_tokens=1000)
    episodes = load_batch(directory / "rollout_0.pt", spec["source"], spec["offsets"])
    assert spec["steps"] == 1 and len(episodes) == 4
    for episode in episodes:
        segment = episode["segments"][0]
        start, end = segment.summary_span
        assert segment.tokens[start:end] == [10, 11, 12, 13]


def test_changed_token_ids_are_rejected(reuse_inputs, tmp_path):
    directory = reuse_inputs / "replay"
    data = torch.load(directory / "rollout_0.pt", weights_only=False)
    data["samples"][0]["tokens"][4] += 1
    path = tmp_path / "changed.pt"
    torch.save(data, path)
    source = directory / "episodes.jsonl"
    with pytest.raises(ValueError, match="no retokenization"):
        load_batch(path, source, index_events(source))


def _run(root, output, workers, modes):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentic_critic_pipeline.train_am_comparison",
            "--model",
            str(root / "base"),
            "--rollouts",
            str(root / "replay"),
            "--validation-plan",
            str(root / "plan.json"),
            "--output",
            str(output),
            "--workers",
            str(workers),
            "--modes",
            *modes,
            "--mini-batches",
            "2",
            "--validation-interval",
            "1",
            "--save-interval",
            "1",
            "--autocast",
            "none",
            "--moment-dtype",
            "float32",
            "--lr",
            "0.001",
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-8000:]


def test_training_three_conditions_and_distributed_gradient_equivalence(reuse_inputs, model):  # noqa: F811
    root = reuse_inputs
    before = {str(p): data_sha256(p) for p in root.rglob("*") if p.is_file()}
    output = root / "replicated"
    modes = ["full_history", "actor_summary", "attention_matching"]
    _run(root, output, 2, modes)
    assert before == {name: data_sha256(name) for name in before}
    _, run, results, training = load_results(output)
    assert run["status"] == "complete" and set(results) == set(modes)
    for mode in modes:
        row = training[mode][0]
        assert row["optimizer_updates"] == 2 and len(row["updates"]) == 2
        assert all(u["grad_norm"] > 0 and u["value_loss"] > 0 for u in row["updates"])
        assert results[mode][0]["episodes"] == 2
        checkpoint = output / mode / "checkpoints/batch_0001/critic.safetensors"
        state = safetensors.torch.load_file(str(checkpoint))
        assert any(not torch.equal(v, model.state_dict()[name]) for name, v in state.items())
    report = output / "training_report"
    for name in (
        "training_metrics.png",
        "training_cost.pdf",
        "validation_ev.png",
        "compaction_l1_profiles.pdf",
        "plot_data.json",
    ):
        assert (report / name).is_file()
    assert len(list((report / "trajectories").glob("*.png"))) == 2
    # The summed, globally token-normalized two-rank update must equal one rank.
    single = root / "single"
    _run(root, single, 1, ["attention_matching"])
    for name in ("critic.safetensors",):
        a = safetensors.torch.load_file(str(output / "attention_matching/checkpoints/batch_0001" / name))
        b = safetensors.torch.load_file(str(single / "attention_matching/checkpoints/batch_0001" / name))
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
    a = torch.load(output / "attention_matching/checkpoints/batch_0001/value_head.pt", weights_only=True)
    b = torch.load(single / "attention_matching/checkpoints/batch_0001/value_head.pt", weights_only=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
    bad = copy.deepcopy(results["attention_matching"][0])
    bad["trained_mode"] = "full_history"
    (output / "attention_matching/validation/batch_0001/metrics.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="trained mode"):
        load_results(output)
