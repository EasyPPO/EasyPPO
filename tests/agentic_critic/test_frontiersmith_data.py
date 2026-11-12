import json
from pathlib import Path

import pytest
import toml
import yaml

from scripts import make_miles_frontiersmith_data as data


def _task(root: Path, problem_id: str) -> Path:
    task = root / f"frontier-cs-algorithm-{problem_id}"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        f'schema_version = "1.1"\n[verifier.env]\nPROBLEM_ID = "{problem_id}"\nJUDGE_URL = "http://judge:8081"\n'
    )
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on:\n      judge:\n        condition: service_started\n"
        "    volumes: [./statement.txt:/app/statement.txt:ro]\n  judge:\n    privileged: true\n"
    )
    (task / "instruction.md").write_text(f"Read /app/statement.txt for {problem_id}.")
    return task


def test_complete_synthetic_manifest_is_required(tmp_path):
    manifest = tmp_path / data._MANIFEST
    manifest.parent.mkdir(parents=True)
    ids = [f"frontiersmith_{i}" for i in range(200)]
    manifest.write_text(json.dumps({"num_problems": 200, "problems": [{"problem_id": value} for value in ids]}))
    for problem_id in ids:
        task = tmp_path / "algorithmic/problems" / problem_id
        (task / "testdata").mkdir(parents=True)
        for name in ("statement.txt", "config.yaml", "chk.cc"):
            (task / name).touch()
    assert data._problem_ids(tmp_path) == ids
    (tmp_path / "algorithmic/problems/frontiersmith_199/chk.cc").unlink()
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


def test_prepare_writes_separate_manifests_and_strict_verifier(monkeypatch, tmp_path):
    ids = [f"frontiersmith_{i}" for i in range(3)]
    monkeypatch.setattr(data, "_problem_ids", lambda source: ids)
    monkeypatch.setattr(data, "_parity_aliases", lambda source: [])
    monkeypatch.setattr(data, "_evaluation_ids", lambda source: ["0", "1"])

    def adapt(source, destination, selected, *, overwrite):
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


def test_default_frontiersmith_retains_and_reports_known_evaluation_aliases(monkeypatch, tmp_path):
    ids = [f"frontiersmith_{i}" for i in range(200)]
    aliases = [{"problem_id": f"frontiersmith_{i}", "frontiercs_id": str(306 + i)} for i in range(10)]
    monkeypatch.setattr(data, "_problem_ids", lambda source: ids)
    monkeypatch.setattr(data, "_parity_aliases", lambda source: aliases)
    monkeypatch.setattr(data, "_evaluation_ids", lambda source: [str(306 + i) for i in range(10)])

    def adapt(source, destination, selected, *, overwrite):
        for problem_id in selected:
            _task(destination, problem_id)

    monkeypatch.setattr(data, "_adapt", adapt)
    output = tmp_path / "prepared"
    data.prepare(source_dir=tmp_path / "source", tasks_dir=tmp_path / "harbor", out=output)
    manifest = json.loads((output / "frontiersmith.manifest.json").read_text())
    assert len(manifest["train"]) == 200 and not manifest["validation"]
    assert {row["problem_id"] for row in aliases} <= set(manifest["train"])
    assert manifest["excluded"] == manifest["evaluation_excluded"] == []
    assert manifest["known_train_eval_aliases"] == aliases
    assert manifest["evaluation_source_revision"] == data.EVAL_SOURCE_REVISION
    for config_path in (tmp_path / "harbor").rglob("task.toml"):
        services = yaml.safe_load((config_path.parent / "environment/docker-compose.yaml").read_text())["services"]
        assert set(services) == {"main"}


def test_full_evaluation_set_includes_problem_61_for_new_solutions(tmp_path):
    for i in range(180):
        task = tmp_path / "algorithmic/problems" / str(i)
        (task / "testdata").mkdir(parents=True)
        (task / "statement.txt").write_text("problem")
        (task / "config.yaml").write_text("type: default")
    selected = data._evaluation_ids(tmp_path)
    assert len(selected) == 180 and "61" in selected


def test_parity_alias_manifest_is_validated(tmp_path):
    manifest = tmp_path / "data/sample_lists/frontiersmith_parity_10.json"
    manifest.parent.mkdir(parents=True)
    aliases = [{"problem_id": f"frontiersmith_{i}", "frontiercs_id": str(306 + i)} for i in range(10)]
    manifest.write_text(json.dumps({"problems": aliases}))
    assert data._parity_aliases(tmp_path) == aliases
    manifest.write_text(json.dumps({"problems": aliases[:9]}))
    with pytest.raises(ValueError, match="10-problem"):
        data._parity_aliases(tmp_path)


def test_trial_uses_shared_judge_without_sidecar_or_source_mounts(tmp_path):
    task = _task(tmp_path / "tasks", "frontiersmith_0")
    source = tmp_path / "source/Frontier-CS"
    algorithmic = str((source / "algorithmic").resolve())
    expression = "${FRONTIER_CS_ALGORITHMIC_PATH:-" + algorithmic + "}"
    with (task / "task.toml").open("a") as output:
        output.write(f'[environment.env]\nFRONTIER_CS_ALGORITHMIC_PATH = "{expression}"\n')
    compose = task / "environment/docker-compose.yaml"
    for url in ("http://judge-host:8081", "http://other-host:9090"):
        data._install_verifier(task, judge_url=url, judge_timeout=900)
        config = toml.loads((task / "task.toml").read_text())
        assert config["verifier"]["env"] == {
            "PROBLEM_ID": "frontiersmith_0",
            "JUDGE_URL": url,
            "MAX_POLL_TIME": "900",
        }
        assert config["verifier"]["timeout_sec"] == 990
        assert "FRONTIER_CS_ALGORITHMIC_PATH" not in config["environment"]["env"]
        services = yaml.safe_load(compose.read_text())["services"]
        assert set(services) == {"main"}
        assert "depends_on" not in services["main"]
        assert services["main"]["volumes"] == ["./statement.txt:/app/statement.txt:ro"]
        assert services["main"]["extra_hosts"] == ["host.docker.internal:host-gateway"]


def test_bundled_judge_serves_both_corpora_from_one_directory():
    root = data.DEFAULT_SOURCE_DIR
    training_ids = data._problem_ids(root)
    evaluation_ids = data._evaluation_ids(root)
    aliases = data._parity_aliases(root)
    assert len(training_ids) == 200 and len(evaluation_ids) == 180
    assert not set(training_ids) & set(evaluation_ids)
    assert {row["problem_id"] for row in aliases} <= set(training_ids)
    compose = yaml.safe_load((root / "algorithmic/docker-compose.yml").read_text())
    assert set(compose["services"]) == {"lightcpverifier"}
    judge = compose["services"]["lightcpverifier"]
    assert judge["build"] == "."
    assert judge["ports"] == ["${FRONTIERCS_JUDGE_BIND_IP:-0.0.0.0}:8081:8081"]
    assert judge["volumes"] == [
        "${PROBLEMS_DIR:-./problems}:/app/problems",
        "./submissions:/app/submissions",
        "./data:/app/data",
    ]


@pytest.mark.parametrize(
    "overrides", [{"judge_url": "not-a-url"}, {"judge_url": "file:///tmp/judge"}, {"judge_timeout": 0}]
)
def test_invalid_judge_configuration_fails_before_preparation(tmp_path, overrides):
    with pytest.raises(ValueError, match="judge-"):
        data.prepare(source_dir=tmp_path / "missing", tasks_dir=tmp_path / "tasks", out=tmp_path / "out", **overrides)
    assert not (tmp_path / "missing").exists()
