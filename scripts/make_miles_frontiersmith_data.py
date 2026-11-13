"""Prepare bundled FrontierSmith and Frontier-CS problems as Harbor/Miles data.

Install the terminal extra, then prepare the embedded corpus:
    python -m pip install -e '.[terminal]'
    python scripts/make_miles_frontiersmith_data.py \
        --tasks-dir /data/harbor --out /data/prepared --judge-url http://judge-host:8081

Training uses all 200 synthetic problems, including the 10 published evaluation
aliases. Evaluation uses all 180 pinned Frontier-CS algorithmic problems. The shared
judge is started independently with Docker Compose in Frontier-CS/algorithmic.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

import toml
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_miles_terminal_data import task_records, write_jsonl  # noqa: E402

SOURCE_URL = "https://github.com/FrontierCS/FrontierSmith.git"
SOURCE_REVISION = "ac3046cce7a270e3177cbb41376f36e03de1d518"
EVAL_SOURCE_URL = "https://github.com/FrontierCS/Frontier-CS.git"
EVAL_SOURCE_REVISION = "59687e5428266a22a49cf51f981b912103bfa91d"
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "Frontier-CS"
_MANIFEST = Path("data/sample_lists/frontiersmith_train_200.json")
_ADAPTER = Path("harbor/adapters/frontier-cs-algorithm")
DEFAULT_JUDGE_URL = "http://host.docker.internal:8081"
DEFAULT_JUDGE_TIMEOUT = 3600


def _problem_ids(source: Path) -> list[str]:
    manifest = json.loads((source / _MANIFEST).read_text())
    ids = [item["problem_id"] for item in manifest["problems"]]
    expected = {f"frontiersmith_{i}" for i in range(200)}
    if len(ids) != 200 or set(ids) != expected or manifest["num_problems"] != 200:
        raise ValueError("Expected the complete 200-problem FrontierSmith synthetic training manifest")
    for problem_id in ids:
        directory = source / "algorithmic/problems" / problem_id
        for name in ("statement.txt", "config.yaml", "chk.cc", "testdata"):
            if not (directory / name).exists():
                raise ValueError(f"Missing FrontierSmith fixture: {directory / name}")
    return ids


def _adapt(source: Path, destination: Path, ids: list[str], *, overwrite: bool) -> None:
    # Import the bundled adapter rather than another version on PYTHONPATH.
    adapter_source = source / _ADAPTER / "src"
    code = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from frontier_cs_algorithm.adapter import FrontierCSAdapter
FrontierCSAdapter(
    Path(sys.argv[2]), Path(sys.argv[3]), task_ids=sys.argv[5:],
    include_non_numeric=True, overwrite=sys.argv[4] == '1',
).run()
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(adapter_source),
            str(source),
            str(destination),
            str(int(overwrite)),
            *ids,
        ],
        check=True,
    )


def _install_verifier(task: Path, *, judge_url: str, judge_timeout: int) -> None:
    config_path = task / "task.toml"
    config = toml.loads(config_path.read_text())
    verifier = config.setdefault("verifier", {})
    verifier["timeout_sec"] = judge_timeout + 90
    verifier.setdefault("env", {}).update(
        JUDGE_URL=judge_url,
        MAX_POLL_TIME=str(judge_timeout),
        FRONTIERCS_JUDGE_TOKEN="${FRONTIERCS_JUDGE_TOKEN:-}",
    )
    config.get("environment", {}).get("env", {}).pop("FRONTIER_CS_ALGORITHMIC_PATH", None)
    config_path.write_text(toml.dumps(config))

    compose_path = task / "environment/docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    services = compose["services"]
    services.pop("judge", None)
    main = services["main"]
    main.pop("depends_on", None)
    # Resolve the local Docker host on Linux too. Multi-host runs must prepare
    # tasks with the judge host's reachable address instead of this local alias.
    main["extra_hosts"] = ["host.docker.internal:host-gateway"]
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    shutil.copy2(Path(__file__).with_name("frontiersmith_verifier.py"), task / "tests/evaluate.py")
    entrypoint = task / "tests/test.sh"
    entrypoint.write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\npython3 /tests/evaluate.py 2>&1 | tee /logs/verifier/evaluation.log\n"
    )
    entrypoint.chmod(0o755)


def _records(
    destination: Path, root: Path, ids: list[str], *, eval_count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0 <= eval_count < len(ids):
        raise ValueError("--eval-count must be nonnegative and leave at least one training problem")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate problem ID in FrontierSmith source manifest")
    rows = task_records(destination, dataset="frontiersmith", split="train", path_root=root)
    by_id = {}
    for row in rows:
        problem_id = Path(row["metadata"]["task_path"]).name.removeprefix("frontier-cs-algorithm-")
        if not re.fullmatch(r"frontiersmith_\d+", problem_id) or problem_id not in ids:
            raise ValueError(f"Unexpected task in FrontierSmith output: {problem_id}")
        if problem_id in by_id:
            raise ValueError(f"Duplicate FrontierSmith problem ID: {problem_id}")
        by_id[problem_id] = row
    if set(by_id) != set(ids):
        raise ValueError(f"Incomplete FrontierSmith tasks; missing {sorted(set(ids) - set(by_id))}")
    ordered_rows = []
    for problem_id in ids:
        row = by_id[problem_id]
        row["metadata"].update(
            problem_id=problem_id,
            source_url=SOURCE_URL,
            source_revision=SOURCE_REVISION,
            reward_key="reward",
        )
        ordered_rows.append(row)
    ranked = sorted(
        (row["metadata"]["problem_id"] for row in ordered_rows),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    held_out = set(ranked[:eval_count])
    training, evaluation = [], []
    for row in ordered_rows:
        record = deepcopy(row)
        is_eval = record["metadata"]["problem_id"] in held_out
        record["metadata"]["split"] = "validation" if is_eval else "train"
        (evaluation if is_eval else training).append(record)
    return training, evaluation


def _parity_aliases(source: Path) -> list[dict]:
    manifest = json.loads((source / "data/sample_lists/frontiersmith_parity_10.json").read_text())
    aliases = manifest["problems"]
    if len(aliases) != 10 or {row["frontiercs_id"] for row in aliases} != {str(i) for i in range(306, 316)}:
        raise ValueError("Expected the pinned 10-problem FrontierSmith/Frontier-CS parity alias manifest")
    if len({row["problem_id"] for row in aliases}) != 10:
        raise ValueError("Duplicate problem ID in FrontierSmith parity manifest")
    return aliases


def _evaluation_ids(source: Path) -> list[str]:
    ids = sorted(
        (path.name for path in (source / "algorithmic/problems").iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    )
    if len(ids) != 180 or "61" not in ids:
        raise ValueError("Expected all 180 pinned Frontier-CS algorithmic problems, including problem 61")
    for problem_id in ids:
        directory = source / "algorithmic/problems" / problem_id
        for name in ("statement.txt", "config.yaml", "testdata"):
            if not (directory / name).exists():
                raise ValueError(f"Missing Frontier-CS fixture: {directory / name}")
    return ids


def _evaluation_records(destination: Path, root: Path, ids: list[str]) -> list[dict]:
    rows = task_records(destination, dataset="frontier-cs", split="test", path_root=root)
    by_id = {}
    for row in rows:
        problem_id = Path(row["metadata"]["task_path"]).name.removeprefix("frontier-cs-algorithm-")
        if problem_id not in ids:
            raise ValueError(f"Unexpected Frontier-CS evaluation task: {problem_id}")
        if problem_id in by_id:
            raise ValueError(f"Duplicate Frontier-CS evaluation task: {problem_id}")
        row["metadata"].update(
            problem_id=problem_id,
            source_url=EVAL_SOURCE_URL,
            source_revision=EVAL_SOURCE_REVISION,
            reward_key="reward",
        )
        by_id[problem_id] = row
    if set(by_id) != set(ids):
        raise ValueError("Incomplete Frontier-CS evaluation tasks")
    return [by_id[problem_id] for problem_id in ids]


def prepare(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    tasks_dir: Path,
    out: Path,
    eval_count: int = 0,
    seed: int = 42,
    overwrite: bool = False,
    judge_url: str = DEFAULT_JUDGE_URL,
    judge_timeout: int = DEFAULT_JUDGE_TIMEOUT,
) -> None:
    parsed_url = urlsplit(judge_url)
    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.hostname
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("--judge-url must be an HTTP(S) base URL reachable from task containers")
    if judge_timeout <= 0:
        raise ValueError("--judge-timeout must be positive and include queue wait time")
    judge_url = judge_url.rstrip("/")
    source_dir = source_dir.resolve()
    tasks_dir = tasks_dir.resolve()
    ids = _problem_ids(source_dir)
    aliases = _parity_aliases(source_dir)
    if not {row["problem_id"] for row in aliases} <= set(ids):
        raise ValueError("Parity aliases do not belong to the synthetic training corpus")
    evaluation_ids = _evaluation_ids(source_dir)
    if not {row["frontiercs_id"] for row in aliases} <= set(evaluation_ids):
        raise ValueError("Parity alias targets must be in the Frontier-CS evaluation corpus")
    if not 0 <= eval_count < len(ids):
        raise ValueError("--eval-count must leave at least one training problem")
    destination = tasks_dir / "frontiersmith"
    _adapt(source_dir, destination, ids, overwrite=overwrite)
    training, evaluation = _records(destination, tasks_dir, ids, eval_count=eval_count, seed=seed)
    evaluation_destination = tasks_dir / "frontier-cs"
    _adapt(source_dir, evaluation_destination, evaluation_ids, overwrite=overwrite)
    benchmark = _evaluation_records(evaluation_destination, tasks_dir, evaluation_ids)
    # Rerunning preparation updates every generated task to the shared judge.
    for directory in (destination, evaluation_destination):
        for config_path in directory.rglob("task.toml"):
            _install_verifier(config_path.parent, judge_url=judge_url, judge_timeout=judge_timeout)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "frontiersmith.train.jsonl", training)
    write_jsonl(out / "frontiersmith.eval.jsonl", evaluation)
    write_jsonl(out / "fcs.eval.jsonl", benchmark)
    (out / "frontiersmith.manifest.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE_URL,
                "source_revision": SOURCE_REVISION,
                "reward": "upstream bounded score / 100; no per-test count conversion",
                "seed": seed,
                "train": [row["metadata"]["problem_id"] for row in training],
                "validation": [row["metadata"]["problem_id"] for row in evaluation],
                "excluded": [],
                "known_train_eval_aliases": aliases,
                "evaluation_source_url": EVAL_SOURCE_URL,
                "evaluation_source_revision": EVAL_SOURCE_REVISION,
                "judge_source_revision": EVAL_SOURCE_REVISION,
                "judge_url": judge_url,
                "judge_timeout": judge_timeout,
                "evaluation": evaluation_ids,
                "evaluation_excluded": [],
                "deduplication_scope": "None; all published problems retained, including the ten parity aliases",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Prepared {len(training)} FrontierSmith train, {len(evaluation)} optional synthetic holdout, and {len(benchmark)} Frontier-CS evaluation tasks"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--judge-url", default=DEFAULT_JUDGE_URL, help="Shared judge URL reachable from every task container"
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=DEFAULT_JUDGE_TIMEOUT,
        help="Submission timeout in seconds, including queue wait",
    )
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
