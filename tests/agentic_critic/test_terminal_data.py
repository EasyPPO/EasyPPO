from pathlib import Path

import pytest
from scripts.make_miles_terminal_data import task_records


def make_task(root: Path, name="task"):
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Create the requested artifact.")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return task


def test_harbor_records_are_sorted_portable_and_do_not_expose_tests(tmp_path):
    make_task(tmp_path, "b")
    make_task(tmp_path, "a")
    rows = task_records(tmp_path, dataset="tmax", split="train", limit=1)
    assert len(rows) == 1 and rows[0]["metadata"]["task_path"] == "a"
    assert rows[0]["prompt"] == "Create the requested artifact."
    assert rows[0]["metadata"]["agent_name"] == "mini-swe-agent"


def test_missing_task_and_instruction_fail(tmp_path):
    with pytest.raises(ValueError, match="No Harbor"):
        task_records(tmp_path, dataset="tmax", split="train")
    task = make_task(tmp_path)
    (task / "instruction.md").unlink()
    with pytest.raises(ValueError, match="instruction"):
        task_records(tmp_path, dataset="terminal-bench-2.1", split="test")


def test_common_root_keeps_train_and_eval_paths_distinct(tmp_path):
    make_task(tmp_path / "train", "same-name")
    make_task(tmp_path / "eval", "same-name")
    train = task_records(tmp_path / "train", dataset="tmax", split="train", path_root=tmp_path)
    test = task_records(tmp_path / "eval", dataset="terminal-bench-2.1", split="test", path_root=tmp_path)
    assert train[0]["metadata"]["task_path"] == "train/same-name"
    assert test[0]["metadata"]["task_path"] == "eval/same-name"


from scripts.make_miles_terminal_data import training_records


def test_all_eligible_tasks_train_without_holdout():
    rows = [{"prompt": str(i), "metadata": {"task_id": str(i)}} for i in range(7)]
    audits = [
        {"task_id": str(i), "prompt_group": group, "fixture_hash": fixture, "fits_initial_action": True}
        for i, (group, fixture) in enumerate(
            [
                ("group1", "f1"),
                ("group1", "f2"),
                ("group2", "f3"),
                ("group3", "f4"),
                ("group4", "f4"),
                ("benchmark", "f5"),
                ("group5", "benchmark_fixture"),
            ]
        )
    ]
    benchmark = [{"prompt_group": "benchmark", "fixture_hash": "benchmark_fixture"}]
    training, excluded = training_records(rows, audits, benchmark)
    train = {r["metadata"]["task_id"] for r in training}
    assert train == {"0", "1", "2", "3"}
    assert all(r["metadata"]["split"] == "train" for r in training)
    assert {e["task_id"] for e in excluded} == {"4", "5", "6"}
    assert training_records(rows, audits, benchmark) == (training, excluded)
    assert all("split" not in row["metadata"] for row in rows)


import json
import sys

from scripts import make_miles_terminal_data as benchmark


def make_tasks(root, count):
    for i in range(count):
        task = root / "terminal-bench-2-1" / f"task-{i:03}"
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "task.toml").write_text("")
        (task / "instruction.md").write_text(f"Task {i}")


def test_missing_dataset_downloads_pinned_version(monkeypatch, tmp_path):
    calls = []

    def download(argv, *, check):
        calls.append(argv)
        assert check
        make_tasks(tmp_path, 89)

    monkeypatch.setattr(benchmark.subprocess, "run", download)
    benchmark.prepare_tb21(tmp_path, tmp_path / "prepared")
    output = tmp_path / "prepared/tb21.eval.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][:5] == [sys.executable, "-m", "harbor.cli.main", "datasets", "download"]
    assert "@sha256:" in calls[0][5]
    assert len(rows) == 89
    assert all(r["metadata"]["split"] == "test" for r in rows)
    assert all((tmp_path / r["metadata"]["task_path"] / "task.toml").is_file() for r in rows)
    benchmark.prepare_tb21(tmp_path, tmp_path / "prepared")
    assert len(calls) == 1  # Reuse complete local tasks without network access.


def test_incomplete_dataset_is_not_a_full_validation(tmp_path):
    make_tasks(tmp_path, 2)
    with pytest.raises(ValueError, match="all 89 tasks"):
        benchmark.prepare_tb21(tmp_path, tmp_path / "prepared")
    assert not (tmp_path / "prepared/tb21.eval.jsonl").exists()
