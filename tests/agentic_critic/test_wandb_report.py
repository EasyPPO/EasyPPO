"""The wandb dump reports every analysis value exactly and never fails the pipeline.

A synthetic comparison (episodes, pre-update and held-out value logs for both critics and
both input swaps) runs through the real ``visualize.sh``; the stored offline wandb summary
is then read back and checked leaf by leaf against the analysis JSON files.
"""

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
wandb = pytest.importorskip("wandb")

from agentic_critic_pipeline import wandb_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODES = ("actor_summary", "full_history")


def make_episode(rng, episode_id, problem, split, reward):
    segments, events, submissions = [], [], []
    n_segments = rng.choice([1, 2, 3, 4, 5])
    previous_len = 0
    for k in range(n_segments):
        prompt = [rng.randrange(1000) for _ in range(rng.randrange(20, 40) + (60 if k else 0))]
        actor = rng.randrange(300, 700) if rng.random() < 0.8 else 2200
        env = rng.randrange(10, 40)
        response_mask = [1] * actor + [0] * env
        tokens = prompt + [rng.randrange(1000) for _ in response_mask]
        segments.append(dict(input_ids=tokens, response_start=len(prompt), loss_mask=response_mask))
        if k:
            events.append(dict(kind="resume", actor=False, tokens=[], segment=k, text=""))
        score = round(100 * reward * (k + 1) / n_segments, 1)
        if k < n_segments - 1:
            split_at = actor - 50
            events.append(dict(kind="action", actor=True, tokens=list(range(split_at)), segment=k))
            events.append(
                dict(
                    kind="compact_request",
                    actor=False,
                    tokens=[0] * env,
                    segment=k,
                    text=f"Best score so far: {score}",
                )
            )
            events.append(
                dict(
                    kind="summary",
                    actor=True,
                    tokens=list(range(50)),
                    segment=k,
                    text=f"<think>x</think>best {score:.2f}",
                )
            )
        else:
            events.append(dict(kind="action", actor=True, tokens=list(range(actor)), segment=k))
            events.append(
                dict(kind="feedback", actor=False, tokens=[0] * env, segment=k, text=f"Best score so far: {score}")
            )
        submissions.append(dict(segment=k, score=score))
        previous_len = len(tokens)
    del previous_len
    return dict(
        episode_id=episode_id,
        problem_id=problem,
        split=split,
        reward=reward,
        segments=segments,
        events=events,
        submissions=submissions,
    )


def value_rows(rng, episode, *, input_mode, trained_mode, phase, batch, heldout):
    rows = []
    level = rng.random()
    for k, segment in enumerate(episode["segments"]):
        positions = [i for i, bit in enumerate(segment["loss_mask"]) if bit]
        drift = (episode["reward"] - level) * (k + 1) / len(episode["segments"])
        jump = (rng.random() - 0.5) * (0.4 if input_mode == "actor_summary" else 0.1)
        values = [level + drift + jump + 0.02 * (rng.random() - 0.5) for _ in positions]
        row = dict(
            episode_id=episode["episode_id"],
            problem_id=episode["problem_id"],
            split=episode["split"],
            segment_index=k,
            reward=episode["reward"],
            response_positions=positions,
            values=values,
            mode=input_mode,
            input_mode=input_mode,
            trained_mode=trained_mode,
            phase=phase,
            batch=batch,
        )
        if heldout:
            row.update(
                checkpoint_run_id=f"ckpt-{trained_mode}",
                checkpoint_batch=batch,
                head_gradient=[rng.gauss(0, 1) for _ in range(8)],
                head_gradient_definition="mean((V-G)*h); value-head weights only",
                evaluation_dtype="float32",
            )
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def build_comparison(root):
    rng = random.Random(7)
    data = root / "data"
    train, batch_of = [], {}
    for batch in range(4):
        for p in rng.sample(range(6), 3):
            for e in range(4):
                eid = f"train-b{batch}-p{p}-e{e}"
                train.append(make_episode(rng, eid, f"P{p}", "train", round(rng.random(), 3)))
                batch_of[eid] = batch
    test = [
        make_episode(rng, f"test-p{p}-e{e}", f"T{p}", "test", round(rng.random(), 3))
        for p in range(4)
        for e in range(4)
    ]
    validation = [
        make_episode(rng, f"val-p{p}-e{e}", f"P{p}", "validation", round(rng.random(), 3))
        for p in range(4)
        for e in range(4)
    ]
    for name, rows in (("train", train), ("test", test), ("validation", validation)):
        write_jsonl(data / f"{name}.jsonl", rows)
    out = root / "out"
    for trained in MODES:
        rows = [
            r
            for ep in train
            for r in value_rows(
                rng,
                ep,
                input_mode=trained,
                trained_mode=trained,
                phase="pre_update",
                batch=batch_of[ep["episode_id"]],
                heldout=False,
            )
        ]
        write_jsonl(out / trained / "train" / "values.rank0.jsonl", rows)
        (out / trained / "train" / "miles_run.json").write_text(
            json.dumps(dict(backend="miles", mode=trained, wandb_run_name=f"g-train-{trained}", logical_batches=4))
        )
        for input_mode in MODES:
            for prefix, episodes in (("eval", test), ("validation", validation)):
                rows = [
                    r
                    for ep in episodes
                    for r in value_rows(
                        rng, ep, input_mode=input_mode, trained_mode=trained, phase="heldout", batch=40, heldout=True
                    )
                ]
                write_jsonl(out / trained / f"{prefix}_{input_mode}" / "values.rank0.jsonl", rows)
    return data, out


def run_visualize(data, out, env_extra=None, script=ROOT / "scripts/agentic_critic/visualize.sh"):
    env = dict(os.environ)
    env.pop("WANDB_API_KEY", None)
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        PYTHONPATH=str(ROOT),
        TRAIN_DATA=str(data / "train.jsonl"),
        EVAL_DATA=str(data / "test.jsonl"),
        VALIDATION_DATA=str(data / "validation.jsonl"),
        OUTPUT_ROOT=str(out),
        BOOTSTRAP="20",
        WANDB_MODE="offline",
        WANDB_GROUP="unit-group",
        WANDB_SILENT="true",
        MPLBACKEND="Agg",
    )
    env.update(env_extra or {})
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=1800)


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison")
    data, out = build_comparison(root)
    result = run_visualize(data, out)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return data, out, result


def stored_summary(manifest):
    return wandb_report.read_transaction_log(manifest["status"]["transaction_log"])


def test_dump_uploads_and_verifies_everything(comparison):
    _, out, _ = comparison
    manifest = json.loads((out / "wandb_dump" / "manifest.json").read_text())
    status = manifest["status"]
    assert status["upload"] == "ok" and status["mode"] == "offline"
    assert status["verified"] is True, manifest["issues"][:20]
    assert status["coverage"]["wrong"] == 0 and status["coverage"]["checked"] > 1000
    assert manifest["issues"] == [], manifest["issues"][:20]
    assert manifest["headline"]["missing"] == []
    assert all(s["status"] == "ok" for s in manifest["sources"].values()), manifest["sources"]
    assert len(manifest["sources"]) == 18  # six suites x analysis/figures/diagnostics
    assert manifest["counts"]["images"] >= 6 * 10
    assert status["run_name"] == "unit-group-analysis" and status["group"] == "unit-group"


def leaves(node, key):
    if isinstance(node, dict):
        for name, child in node.items():
            yield from leaves(child, f"{key}/{name}")
    elif isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
        yield key, ("table", len(node))
    elif isinstance(node, list):
        yield key, ("json", json.dumps(node))
    else:
        yield key, ("scalar", node)


def test_every_json_leaf_is_stored_exactly(comparison):
    """Independent of the dumper's own coverage check: walk the files, read the wandb store."""
    _, out, _ = comparison
    manifest = json.loads((out / "wandb_dump" / "manifest.json").read_text())
    stored = stored_summary(manifest)
    checked = 0
    for suite, directories in wandb_report.suites(out):
        for module, directory in directories.items():
            data = json.loads((directory / wandb_report.MODULES[module]).read_text())
            for key, (kind, value) in leaves(data, f"{suite}/{module}"):
                checked += 1
                if kind == "table":
                    table_key = key.replace(f"{suite}/{module}", f"tables/{suite}/{module}", 1)
                    assert stored[table_key]["_type"] == "table-file", table_key
                    assert stored[table_key]["nrows"] == value, table_key
                elif kind == "json":
                    assert json.loads(stored[key]) == json.loads(value), key
                else:
                    assert stored[key] == value, (key, stored.get(key), value)
    assert checked > 5000


def test_headline_matches_sources(comparison):
    _, out, _ = comparison
    stored = stored_summary(json.loads((out / "wandb_dump" / "manifest.json").read_text()))
    analysis = json.loads((out / "analysis" / "analysis.json").read_text())
    heldout, train = analysis["phases"]["heldout"], analysis["phases"]["pre_update"]
    assert stored["headline/test/main/ev/all/full_history"] == heldout["by_compactions"]["all"]["full_history"]["ev"]
    assert (
        stored["headline/train_online/main/ev/all/full_minus_actor/low"]
        == train["by_compactions"]["all"]["full_minus_actor"]["ev"]["low"]
    )
    # True error on both sides of a compaction, with its paired interval.
    for field in ("sq_error_improvement", "abs_error_after", "moved_closer"):
        assert (
            stored[f"headline/test/main/at_compaction/{field}/actor_summary"]
            == heldout["at_compaction"][field]["actor_summary"]
        )
        assert (
            stored[f"headline/test/main/at_compaction/{field}/full_minus_actor/value"]
            == heldout["at_compaction"][field]["full_minus_actor"]["value"]
        )
    swap = json.loads(
        (out / "validation" / "input_swap_full_history" / "diagnostics" / "diagnostics.json").read_text()
    )
    assert (
        stored["headline/validation/swap_trained_full_history/zero_compaction_check/passed"]
        == swap["phases"]["heldout"]["zero_compaction_check"]["passed"]
    )
    assert stored["data/train/episodes"] == 48 and stored["data/test/episodes"] == 16
    assert stored["training/full_history/wandb_run_name"] == "g-train-full_history"
    # Training curves are logged per batch against their own step metric.
    curve = [k for k in stored if k.startswith("curve/main/pre_update/full_history/")]
    assert "curve/main/pre_update/full_history/ev" in curve


def test_new_jump_error_fields_are_consistent(comparison):
    _, out, _ = comparison
    heldout = json.loads((out / "analysis" / "analysis.json").read_text())["phases"]["heldout"]["at_compaction"]
    for mode in MODES:
        before, after = heldout["abs_error_before"][mode], heldout["abs_error_after"][mode]
        assert heldout["error_improvement"][mode] == pytest.approx(before - after)
        sq = heldout["sq_error_before"][mode] - heldout["sq_error_after"][mode]
        assert heldout["sq_error_improvement"][mode] == pytest.approx(sq)
        assert 0.0 <= heldout["moved_closer"][mode] <= 1.0
    figures = json.loads((out / "figures" / "figures.json").read_text())["phases"]["heldout"]
    row = figures["jump_by_compaction_index"][0]
    assert {"error_improvement", "sq_error_improvement", "abs_error_before", "abs_error_after"} <= row.keys()


def test_broken_inputs_and_tracking_never_fail(comparison, tmp_path, monkeypatch):
    _, out, _ = comparison
    broken = tmp_path / "broken"
    shutil.copytree(out, broken, ignore=shutil.ignore_patterns("wandb_dump", "wandb"))
    (broken / "diagnostics" / "diagnostics.json").unlink()
    (broken / "input_swap_actor_summary" / "analysis" / "analysis.json").write_text("{not json")
    analysis = json.loads((broken / "analysis" / "analysis.json").read_text())
    del analysis["phases"]["heldout"]["at_compaction"]
    (broken / "analysis" / "analysis.json").write_text(json.dumps(analysis))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_SILENT", "true")

    def refuse(*args, **kwargs):
        raise RuntimeError("tracking backend unavailable")

    monkeypatch.setattr(wandb, "init", refuse)
    code = wandb_report.main(
        [
            "--output-root",
            str(broken),
            "--mode",
            "online",
            "--episodes",
            f"train={broken / 'missing.jsonl'}",
            "nonsense",
        ]
    )
    assert code == 0
    manifest = json.loads((broken / "wandb_dump" / "manifest.json").read_text())
    kinds = {i["kind"] for i in manifest["issues"]}
    assert {"missing_file", "invalid_json", "missing_field", "tracking", "argument"} <= kinds
    assert manifest["status"]["upload"] == "failed"
    summary = json.loads((broken / "wandb_dump" / "summary.json").read_text())
    # Everything that was readable is still reported.
    assert "headline/test/main/ev/all/full_history" in summary
    assert not any(k.startswith("input_swap_actor_summary/analysis/") for k in summary)
    # --strict turns recorded issues into a failing exit code for CI only.
    assert wandb_report.main(["--output-root", str(broken), "--mode", "disabled", "--strict"]) == 1


def test_dump_crash_does_not_fail_visualize(comparison, tmp_path):
    """A dump that dies outright still leaves visualize.sh (and so run_comparison.sh) successful."""
    data, out, _ = comparison
    copy = tmp_path / "copy"
    shutil.copytree(out, copy, ignore=shutil.ignore_patterns("wandb_dump", "wandb"))
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "agentic_critic_pipeline", repo / "agentic_critic_pipeline")
    (repo / "scripts" / "agentic_critic").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/agentic_critic/visualize.sh", repo / "scripts/agentic_critic/visualize.sh")
    (repo / "agentic_critic_pipeline" / "wandb_report.py").write_text("raise RuntimeError('dump crashed')\n")
    result = run_visualize(data, copy, dict(PYTHONPATH=str(repo)), script=repo / "scripts/agentic_critic/visualize.sh")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "dump crashed" in result.stderr and "wandb report dump failed" in result.stderr
    assert (copy / "analysis" / "analysis.json").is_file()
