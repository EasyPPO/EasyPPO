import json
from pathlib import Path

import pytest

from scripts import make_miles_frontiersmith_data as data


def _task(root: Path, problem_id: str) -> Path:
    task = root / f"frontier-cs-algorithm-{problem_id}"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('schema_version = "1.1"\n')
    (task / "instruction.md").write_text(f"Read /app/statement.txt for {problem_id}.")
    return task


def test_complete_synthetic_manifest_is_required(tmp_path):
    manifest = tmp_path / data._MANIFEST
    manifest.parent.mkdir(parents=True)
    ids = [f"frontiersmith_{i}" for i in range(200)]
    manifest.write_text(json.dumps({"num_problems": 200, "problems": [{"problem_id": value} for value in ids]}))
    for problem_id in ids:
        task = tmp_path / "Frontier-CS/algorithmic/problems" / problem_id
        (task / "testdata").mkdir(parents=True)
        for name in ("statement.txt", "config.yaml", "chk.cc"):
            (task / name).touch()
    assert data._problem_ids(tmp_path) == ids
    (tmp_path / "Frontier-CS/algorithmic/problems/frontiersmith_199/chk.cc").unlink()
    with pytest.raises(ValueError, match="Missing FrontierSmith fixture"):
        data._problem_ids(tmp_path)
    manifest.write_text(json.dumps({"num_problems": 1, "problems": [{"problem_id": "306"}]}))
    with pytest.raises(ValueError, match="200-problem"):
        data._problem_ids(tmp_path)


def test_holdout_is_reproducible_disjoint_and_paths_use_common_root(tmp_path):
    ids = [f"frontiersmith_{i}" for i in range(6)]
    destination = tmp_path / "frontiersmith"
    for problem_id in ids:
        _task(destination, problem_id)
    training, evaluation = data._records(destination, tmp_path, ids, eval_count=2, seed=42)
    assert (training, evaluation) == data._records(destination, tmp_path, ids, eval_count=2, seed=42)
    assert len(training) == 4 and len(evaluation) == 2
    train_ids = {row["metadata"]["problem_id"] for row in training}
    eval_ids = {row["metadata"]["problem_id"] for row in evaluation}
    assert not train_ids & eval_ids and train_ids | eval_ids == set(ids)
    for row in training + evaluation:
        metadata = row["metadata"]
        assert (tmp_path / metadata["task_path"] / "task.toml").is_file()
        assert metadata["source_revision"] == data.SOURCE_REVISION
        assert metadata["dataset"] == "frontiersmith" and metadata["reward_key"] == "reward"
        assert metadata["split"] == ("train" if metadata["problem_id"] in train_ids else "validation")
        assert "checker" not in row["prompt"] and "testdata" not in row["prompt"]
    assert len(data._records(destination, tmp_path, ids, eval_count=0, seed=42)[0]) == 6


def test_incomplete_or_unexpected_tasks_and_invalid_splits_fail(tmp_path):
    destination = tmp_path / "frontiersmith"
    _task(destination, "frontiersmith_0")
    ids = ["frontiersmith_0", "frontiersmith_1"]
    with pytest.raises(ValueError, match="Incomplete"):
        data._records(destination, tmp_path, ids, eval_count=0, seed=42)
    _task(destination, "306")
    with pytest.raises(ValueError, match="Unexpected task"):
        data._records(destination, tmp_path, ids, eval_count=0, seed=42)
    for count in (-1, 2):
        with pytest.raises(ValueError, match="eval-count"):
            data._records(destination, tmp_path, ids, eval_count=count, seed=42)


def test_source_revision_and_tracked_changes_are_checked(monkeypatch, tmp_path):
    answers = iter(["other-revision\n"])
    monkeypatch.setattr(data.subprocess, "check_output", lambda *args, **kwargs: next(answers))
    with pytest.raises(ValueError, match="must be at"):
        data._prepare_source(tmp_path)
    answers = iter([data.SOURCE_REVISION, " M Frontier-CS/algorithmic/problems/frontiersmith_0/chk.cc\n"])
    with pytest.raises(ValueError, match="modified tracked files"):
        data._prepare_source(tmp_path)


def test_prepare_writes_separate_manifests_and_strict_verifier(monkeypatch, tmp_path):
    ids = [f"frontiersmith_{i}" for i in range(3)]
    monkeypatch.setattr(data, "_prepare_source", lambda source, **kwargs: None)
    monkeypatch.setattr(data, "_problem_ids", lambda source: ids)
    monkeypatch.setattr(data, "_parity_aliases", lambda source: [])
    monkeypatch.setattr(data, "_evaluation_ids", lambda source: ["0", "1"])

    def adapt(source, destination, selected, *, overwrite, problems_source=None):
        assert not overwrite
        for problem_id in selected:
            _task(destination, problem_id)

    monkeypatch.setattr(data, "_adapt", adapt)
    output = tmp_path / "prepared"
    data.prepare(source_dir=tmp_path / "source", tasks_dir=tmp_path / "harbor", out=output, eval_count=1)
    train = [json.loads(row) for row in (output / "frontiersmith.train.jsonl").read_text().splitlines()]
    evaluation = [json.loads(row) for row in (output / "frontiersmith.eval.jsonl").read_text().splitlines()]
    manifest = json.loads((output / "frontiersmith.manifest.json").read_text())
    assert manifest["train"] == [row["metadata"]["problem_id"] for row in train]
    assert manifest["validation"] == [row["metadata"]["problem_id"] for row in evaluation]
    benchmark = [json.loads(row) for row in (output / "fcs.eval.jsonl").read_text().splitlines()]
    assert len(benchmark) == 2 and all(row["metadata"]["dataset"] == "frontier-cs" for row in benchmark)
    assert all(row["metadata"]["source_revision"] == data.EVAL_SOURCE_REVISION for row in benchmark)
    for row in train + evaluation:
        tests = tmp_path / "harbor" / row["metadata"]["task_path"] / "tests"
        assert (tests / "evaluate.py").read_text() == Path(data.__file__).with_name(
            "frontiersmith_verifier.py"
        ).read_text()
        assert "pipefail" in (tests / "test.sh").read_text()
        assert "echo" not in (tests / "test.sh").read_text()


def test_duplicate_frontiersmith_problem_ids_are_rejected(tmp_path):
    destination = tmp_path / "frontiersmith"
    _task(destination, "frontiersmith_0")
    _task(destination / "nested", "frontiersmith_0")
    with pytest.raises(ValueError, match="Duplicate FrontierSmith problem ID"):
        data._records(destination, tmp_path, ["frontiersmith_0"], eval_count=0, seed=42)


def test_default_frontiersmith_excludes_known_evaluation_aliases(monkeypatch, tmp_path):
    ids = [f"frontiersmith_{i}" for i in range(200)]
    aliases = [{"problem_id": f"frontiersmith_{i}", "frontiercs_id": str(306 + i)} for i in range(10)]
    monkeypatch.setattr(data, "_prepare_source", lambda source, **kwargs: None)
    monkeypatch.setattr(data, "_problem_ids", lambda source: ids)
    monkeypatch.setattr(data, "_parity_aliases", lambda source: aliases)
    monkeypatch.setattr(data, "_evaluation_ids", lambda source: [str(306 + i) for i in range(10)])

    def adapt(source, destination, selected, *, overwrite, problems_source=None):
        for problem_id in selected:
            _task(destination, problem_id)

    monkeypatch.setattr(data, "_adapt", adapt)
    output = tmp_path / "prepared"
    data.prepare(source_dir=tmp_path / "source", tasks_dir=tmp_path / "harbor", out=output)
    manifest = json.loads((output / "frontiersmith.manifest.json").read_text())
    assert len(manifest["train"]) == 190 and not manifest["validation"]
    assert not set(manifest["train"]) & {row["problem_id"] for row in aliases}
    assert len(manifest["excluded"]) == 10
    assert manifest["evaluation_source_revision"] == data.EVAL_SOURCE_REVISION


def test_official_evaluation_set_excludes_changed_problem_61(tmp_path):
    for i in range(180):
        task = tmp_path / "algorithmic/problems" / str(i)
        (task / "testdata").mkdir(parents=True)
        (task / "statement.txt").write_text("problem")
        (task / "config.yaml").write_text("type: default")
    selected = data._evaluation_ids(tmp_path)
    assert len(selected) == 179 and "61" not in selected


def test_parity_alias_manifest_is_validated(tmp_path):
    manifest = tmp_path / "data/sample_lists/frontiersmith_parity_10.json"
    manifest.parent.mkdir(parents=True)
    aliases = [{"problem_id": f"frontiersmith_{i}", "frontiercs_id": str(306 + i)} for i in range(10)]
    manifest.write_text(json.dumps({"problems": aliases}))
    assert data._parity_aliases(tmp_path) == aliases
    manifest.write_text(json.dumps({"problems": aliases[:9]}))
    with pytest.raises(ValueError, match="10-problem"):
        data._parity_aliases(tmp_path)


def test_generated_mounts_pin_the_correct_source_for_each_corpus(tmp_path):
    task = _task(tmp_path / "tasks", "frontiersmith_0")
    source = tmp_path / "source/Frontier-CS"
    algorithmic = str((source / "algorithmic").resolve())
    expression = "${FRONTIER_CS_ALGORITHMIC_PATH:-" + algorithmic + "}"
    (task / "task.toml").write_text(f'[environment.env]\nFRONTIER_CS_ALGORITHMIC_PATH = "{expression}"\n')
    compose = task / "environment/docker-compose.yaml"
    compose.write_text(f"services:\n  judge:\n    build:\n      context: {expression}\n")
    data._install_verifier(task, problems_source=source)
    assert expression not in compose.read_text() and algorithmic in compose.read_text()
    assert expression not in (task / "task.toml").read_text()
