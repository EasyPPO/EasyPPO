"""Sampling, eight-worker scheduling, and real subprocess integration on CPU."""

import json
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from agentic_critic_pipeline.compact_grad import main as single_main
from agentic_critic_pipeline.compact_grad_parallel import main, merge_results, run_jobs, worker_devices
from agentic_critic_pipeline.gradient_cohort import load_selection, sample_segments
from agentic_critic_pipeline.records import data_sha256
from tests.agentic_critic.test_compact_grad_cli import artifacts  # noqa: F401


def test_exactly_100_long_segments_sampled_once_and_balanced_across_eight_workers():
    inventory = [
        dict(
            episode_id=f"e{i:03d}",
            segment_index=j,
            loss_tokens=1500 if i < 140 else 100,
            context_tokens=1600,
            cost=1600**2,
        )
        for i in range(160)
        for j in range(2)
    ]
    kwargs = dict(count=100, minimum=1024, window_size=64, workers=8, seed=1234)
    rows, eligible = sample_segments(inventory, **kwargs)
    assert eligible == 140 and len(rows) == 100
    assert len({(row["episode_id"], row["segment_index"]) for row in rows}) == 100
    assert all(row["segment_index"] == 1 and row["loss_tokens"] >= 1024 for row in rows)
    counts = [sum(row["worker"] == i for row in rows) for i in range(8)]
    assert sorted(counts) == [12, 12, 12, 12, 13, 13, 13, 13]
    assert (rows, eligible) == sample_segments(list(reversed(inventory)), **kwargs)
    assert rows != sample_segments(inventory, **dict(kwargs, seed=5))[0]
    with pytest.raises(ValueError, match="exactly 150"):
        sample_segments(inventory, **dict(kwargs, count=150))
    with pytest.raises(ValueError, match="disjoint"):
        sample_segments(inventory, **dict(kwargs, minimum=100))


def test_visible_gpu_mapping_is_respected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    args = SimpleNamespace(device="cuda", workers=8, gpus="7,6,5,4,3,2,1,0")
    assert worker_devices(args) == [f"GPU-{letter}" for letter in "hgfedcba"]
    with pytest.raises(ValueError, match="distinct"):
        worker_devices(SimpleNamespace(device="cuda", workers=2, gpus="0,0"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="unavailable"):
        worker_devices(args)


def test_all_eight_jobs_get_distinct_gpu_environments(tmp_path):
    devices = [str(i) for i in range(8)]
    jobs = [(i, [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"]) for i in range(8)]
    run_jobs(jobs, devices, tmp_path, "probe")
    assert [(tmp_path / f"probe_{i}.log").read_text().strip() for i in range(8)] == devices


def test_failed_worker_terminates_siblings(tmp_path):
    jobs = [
        (0, [sys.executable, "-c", "import time; time.sleep(30)"]),
        (1, [sys.executable, "-c", "raise SystemExit(2)"]),
    ]
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="worker exited 2"):
        run_jobs(jobs, [None, None], tmp_path, "failing")
    assert time.monotonic() - start < 15


@pytest.mark.parametrize("mode", ["actor_summary", "full_history"])
def test_parallel_real_checkpoints_match_serial_with_global_whitening(artifacts, tmp_path, mode):  # noqa: F811
    base, critic, source = artifacts
    root = tmp_path / "parallel"
    main(
        [
            "--actor",
            str(base),
            "--critic",
            str(critic),
            "--episodes",
            str(source),
            "--output",
            str(root),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--workers",
            "2",
            "--segments",
            "2",
            "--min-segment-tokens",
            "128",
            "--controls",
            "1",
            "--bootstrap",
            "20",
            "--critic-mode",
            mode,
        ]
    )
    plan = json.loads((root / "selection.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["measured_boundaries"] == 2 and summary["skipped_boundaries"] == 0
    metadata = json.loads((root / "manifest.json").read_text())
    assert metadata["schema_version"] == 2
    selections = [
        load_selection(root / "selection.json", episodes=root / "cohort.jsonl", shard=i, window_size=64, seed=1234)
        for i in range(2)
    ]
    assert len(selections[0]) == len(selections[1]) == 1 and not selections[0] & selections[1]
    serial_cache = tmp_path / "serial_cache"
    single_main(
        [
            "cache-values",
            "--base-model",
            str(base),
            "--critic",
            str(critic),
            "--episodes",
            str(root / "cohort.jsonl"),
            "--output",
            str(serial_cache),
            "--device",
            "cpu",
            "--critic-mode",
            mode,
        ]
    )
    serial_manifest = json.loads((serial_cache / "manifest.json").read_text())
    merged = json.loads((root / "cache/manifest.json").read_text())
    assert merged["advantage_moments"] == pytest.approx(serial_manifest["advantage_moments"])
    assert (root / "cache/values.jsonl").read_text() == (serial_cache / "values.jsonl").read_text()
    serial = tmp_path / "serial"
    single_main(
        [
            "measure",
            "--actor",
            str(base),
            "--episodes",
            str(root / "cohort.jsonl"),
            "--values",
            str(serial_cache),
            "--output",
            str(serial),
            "--device",
            "cpu",
            "--controls",
            "1",
            "--bootstrap",
            "20",
        ]
    )
    actual = {row["episode_id"]: row for row in map(json.loads, (root / "boundaries.jsonl").read_text().splitlines())}
    expected = list(map(json.loads, (serial / "boundaries.jsonl").read_text().splitlines()))
    for row in expected:
        got = actual[row["episode_id"]]
        assert got["target"] == pytest.approx(row["target"])
        assert got["ratio_to_control_mean"] == pytest.approx(row["ratio_to_control_mean"])
        assert got["control_positions"] == row["control_positions"]
        assert "cosine" not in got["target"] and "projection_share" not in got["target"]
    # No partial or duplicate result is allowed into the final summary.
    path = root / "measure_0/boundaries.jsonl"
    path.write_text(path.read_text() * 2)
    manifest_path = root / "measure_0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["boundaries_sha256"] = data_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        merge_results(root, plan, seed=1234, bootstrap=20)
    plan["seed"] += 1
    (root / "selection.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksum"):
        load_selection(root / "selection.json", episodes=root / "cohort.jsonl", shard=0, window_size=64, seed=1234)
