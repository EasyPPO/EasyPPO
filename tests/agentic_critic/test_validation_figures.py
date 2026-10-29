"""The focused offline report renders independently and preserves its raw inputs."""

import gzip
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_critic_pipeline import validation_figures
from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.validation_metrics import ValidationMetrics
from agentic_critic_pipeline.validation_storage import compact_episode

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("focused-report")
    ids = [f"validation:{i}" for i in range(20)]
    plan = dict(
        schema_version=1,
        selected_episode_ids=ids,
        baseline_tokens=1024,
        profile_span=3076,
        episodes={
            identity: dict(
                problem_id=f"frontiersmith:{i}",
                reward=(i % 5) / 4,
                n_compactions=i % 3,
                segments=[dict(tokens=1100, positions_sha256=digest(list(range(1100)))) for _ in range(1 + i % 3)],
            )
            for i, identity in enumerate(ids)
        },
    )
    plan["plan_hash"] = digest(plan)
    (root / "validation_replay").mkdir()
    (root / "validation_replay/validation_plan.json").write_text(json.dumps(plan))
    for mode in validation_figures.MODES:
        for batch in (5, 10):
            directory = root / mode / f"train/validation/batch_{batch:04d}"
            directory.mkdir(parents=True)
            metrics = ValidationMetrics(plan, mode=mode, batch=batch, collect_profiles=False)
            with gzip.open(directory / "selected_trajectories.jsonl.gz", "wt") as traces, gzip.open(
                directory / "values.jsonl.gz", "wt"
            ) as raw:
                for i, identity in enumerate(ids):
                    count = 1 + i % 3
                    episode = dict(
                        episode_id=identity,
                        mode=mode,
                        batch=batch,
                        problem_id=f"frontiersmith:{i}",
                        reward=(i % 5) / 4,
                        n_compactions=count - 1,
                        segments=[
                            dict(
                                segment_index=j,
                                response_positions=list(range(1100)),
                                values=[0.1 + j * (0.1 if mode == "full_history" else 0.2)] * 1100,
                            )
                            for j in range(count)
                        ],
                    )
                    episode = compact_episode(episode)
                    metrics.add(episode, traces=traces)
                    raw.write(json.dumps(episode) + "\n")
            summary = metrics.summary()
            summary["plan_hash"] = plan["plan_hash"]
            (directory / "metrics.json").write_text(json.dumps(summary))
            (directory / "chunk_0000").mkdir()
            # Rendering must never open, overwrite or delete the native dump.
            (directory / f"chunk_0000/{batch}_0.critic.pt").write_bytes(b"native dump preservation sentinel")
        (root / mode / "train/miles_run.json").write_text(json.dumps(dict(logical_batches=10, validation_interval=5)))
    return root


def _raw_hashes(root):
    return {
        str(path.relative_to(root)): data_sha256(path)
        for path in root.rglob("*")
        if path.is_file() and "validation_report" not in path.parts
    }


def test_manual_report_renders_only_focused_figures_and_preserves_raw_data(report_source, tmp_path):
    root = report_source
    before = _raw_hashes(root)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A symlink outside a venv can lose pyvenv.cfg and import system packages.
    python = bin_dir / "python"
    python.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    # Fail even if a future change only imports W&B without uploading anything.
    (tmp_path / "wandb.py").write_text("raise RuntimeError('Visualization must not import wandb')\n")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/agentic_critic/visualize.sh")],
        env={
            **os.environ,
            "OUTPUT_ROOT": str(root),
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "WANDB_MODE": "online",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Visualization must not import wandb" not in result.stdout + result.stderr
    output = root / "validation_report"
    assert not (output / "wandb_status.json").exists()
    names = {
        "validation_ev",
        "ev_by_prior_compaction_count",
        "ev_by_total_compaction_count",
        "compaction_l1_profiles",
        "compaction_jump_by_count",
    }
    names.update(f"trajectories/trajectory_{i:02d}" for i in range(1, 21))
    for suffix in (".png", ".pdf"):
        files = list(output.rglob("*" + suffix))
        assert {str(path.relative_to(output).with_suffix("")) for path in files} == names
        assert all(path.stat().st_size > 1000 for path in files)
    _, results = validation_figures._load(root)
    assert json.loads((output / "plot_data.json").read_text()) == results
    for mode, jump in (("full_history", 0.1), ("actor_summary", 0.2)):
        profile = results[mode][-1]["profiles"]["all"]
        assert profile["mean_l1"][profile["offsets"].index(0)] == pytest.approx(jump)
        assert profile["event_counts"][profile["offsets"].index(0)] == 19
        stored = json.loads((root / mode / "train/validation/batch_0010/metrics.json").read_text())
        assert stored["profiles"] == {}
    assert not list(root.rglob("compaction_events.jsonl.gz"))
    assert _raw_hashes(root) == before


@pytest.mark.parametrize("fault", ["duplicate", "missing", "position", "target", "value"])
def test_reconstructed_curves_require_complete_aligned_values(report_source, tmp_path, fault):
    plan = json.loads((report_source / "validation_replay/validation_plan.json").read_text())
    source = report_source / "full_history/train/validation/batch_0010"
    metrics = json.loads((source / "metrics.json").read_text())
    with gzip.open(source / "values.jsonl.gz", "rt") as stream:
        rows = list(map(json.loads, stream))
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "missing":
        rows.pop()
    elif fault == "position":
        rows[0]["segments"][0]["response_position_ranges"][0][0] = 999
    elif fault == "target":
        rows[0]["reward"] = 999
    else:
        rows[0]["segments"][0]["values"][0] = 999
    target = tmp_path / "full_history/train/validation/batch_0010"
    target.mkdir(parents=True)
    with gzip.open(target / "values.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        validation_figures._rebuild_profiles(tmp_path, plan, mode="full_history", metrics=metrics)
