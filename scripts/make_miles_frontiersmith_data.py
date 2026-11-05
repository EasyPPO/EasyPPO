"""Prepare pinned FrontierSmith synthetic problems as Harbor/Miles terminal data.

Install the terminal extra (the adapter is loaded from the pinned checkout):
    python -m pip install -e '.[terminal]'
    python scripts/make_miles_frontiersmith_data.py --source-dir /data/FrontierSmith \
        --tasks-dir /data/harbor --out /data/prepared

A missing source checkout is cloned at its pinned revision. Training uses 190
synthetic problems after excluding 10 known aliases of Frontier-CS evaluation
problems. Evaluation uses the official pinned 179-problem algorithmic set.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_miles_terminal_data import task_records, write_jsonl  # noqa: E402

SOURCE_URL = "https://github.com/FrontierCS/FrontierSmith.git"
SOURCE_REVISION = "ac3046cce7a270e3177cbb41376f36e03de1d518"
EVAL_SOURCE_URL = "https://github.com/FrontierCS/Frontier-CS.git"
EVAL_SOURCE_REVISION = "59687e5428266a22a49cf51f981b912103bfa91d"
_MANIFEST = Path("data/sample_lists/frontiersmith_train_200.json")
_ADAPTER = Path("harbor/adapters/frontier-cs-algorithm")


def _prepare_source(source: Path, *, url: str = SOURCE_URL, revision: str = SOURCE_REVISION) -> None:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", revision], check=True)
    actual_revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != revision:
        raise ValueError(f"{source} must be at {revision}, found {actual_revision}; use a separate pinned checkout")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise ValueError(f"{source} has modified tracked files; preparation requires the unmodified pinned source")


def _problem_ids(source: Path) -> list[str]:
    manifest = json.loads((source / _MANIFEST).read_text())
    ids = [item["problem_id"] for item in manifest["problems"]]
    expected = {f"frontiersmith_{i}" for i in range(200)}
    if len(ids) != 200 or set(ids) != expected or manifest["num_problems"] != 200:
        raise ValueError("Expected the complete 200-problem FrontierSmith synthetic training manifest")
    for problem_id in ids:
        directory = source / "Frontier-CS/algorithmic/problems" / problem_id
        for name in ("statement.txt", "config.yaml", "chk.cc", "testdata"):
            if not (directory / name).exists():
                raise ValueError(f"Missing FrontierSmith fixture: {directory / name}")
    return ids


def _adapt(
    source: Path, destination: Path, ids: list[str], *, overwrite: bool, problems_source: Path | None = None
) -> None:
    # Import the adapter from the validated checkout, rather than an unrelated
    # version with the same package name elsewhere on PYTHONPATH.
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
            str(problems_source or source / "Frontier-CS"),
            str(destination),
            str(int(overwrite)),
            *ids,
        ],
        check=True,
    )


def _install_verifier(task: Path, *, problems_source: Path) -> None:
    # Train/eval use two source repositories. Pin each task's mounts explicitly
    # so a shared FRONTIER_CS_ALGORITHMIC_PATH cannot redirect one corpus to the other.
    algorithmic = str((problems_source / "algorithmic").resolve())
    expression = "${FRONTIER_CS_ALGORITHMIC_PATH:-" + algorithmic + "}"
    for relative in ("task.toml", "environment/docker-compose.yaml"):
        path = task / relative
        if path.exists():
            path.write_text(path.read_text().replace(expression, algorithmic))
    shutil.copy2(Path(__file__).with_name("frontiersmith_verifier.py"), task / "tests/evaluate.py")
    entrypoint = task / "tests/test.sh"
    entrypoint.write_text(
        "#!/bin/bash\nset -euo pipefail\nmkdir -p /logs/verifier\npython3 /tests/evaluate.py 2>&1 | tee /logs/verifier/evaluation.log\n"
    )
    entrypoint.chmod(0o755)


def _records(
    destination: Path, root: Path, ids: list[str], *, eval_count: int, seed: int, excluded_ids: set[str] | None = None
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
        if problem_id in (excluded_ids or set()):
            continue
        row = by_id[problem_id]
        row["metadata"].update(
            problem_id=problem_id,
            source_url=SOURCE_URL,
            source_revision=SOURCE_REVISION,
            reward_key="reward",
        )
        ordered_rows.append(row)
    if not 0 <= eval_count < len(ordered_rows):
        raise ValueError("--eval-count must leave at least one training problem after alias exclusions")
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
        raise ValueError(
            "Expected the pinned 180 Frontier-CS algorithmic problems before official problem-61 exclusion"
        )
    selected = [value for value in ids if value != "61"]
    for problem_id in selected:
        directory = source / "algorithmic/problems" / problem_id
        for name in ("statement.txt", "config.yaml", "testdata"):
            if not (directory / name).exists():
                raise ValueError(f"Missing Frontier-CS fixture: {directory / name}")
    return selected


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
    source_dir: Path,
    tasks_dir: Path,
    out: Path,
    frontier_cs_source_dir: Path | None = None,
    eval_count: int = 0,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    tasks_dir = tasks_dir.resolve()
    _prepare_source(source_dir)
    ids = _problem_ids(source_dir)
    aliases = _parity_aliases(source_dir)
    excluded_ids = {row["problem_id"] for row in aliases}
    if not excluded_ids <= set(ids):
        raise ValueError("Parity aliases do not belong to the synthetic training corpus")
    frontier_cs_source_dir = (frontier_cs_source_dir or source_dir.parent / "Frontier-CS-eval").resolve()
    _prepare_source(frontier_cs_source_dir, url=EVAL_SOURCE_URL, revision=EVAL_SOURCE_REVISION)
    evaluation_ids = _evaluation_ids(frontier_cs_source_dir)
    if not {row["frontiercs_id"] for row in aliases} <= set(evaluation_ids):
        raise ValueError("Parity alias targets must be in the Frontier-CS evaluation corpus")
    if not 0 <= eval_count < len(ids) - len(excluded_ids):
        raise ValueError("--eval-count must leave at least one training problem after alias exclusions")
    destination = tasks_dir / "frontiersmith"
    _adapt(source_dir, destination, ids, overwrite=overwrite)
    training, evaluation = _records(
        destination, tasks_dir, ids, eval_count=eval_count, seed=seed, excluded_ids=excluded_ids
    )
    evaluation_destination = tasks_dir / "frontier-cs"
    _adapt(
        source_dir, evaluation_destination, evaluation_ids, overwrite=overwrite, problems_source=frontier_cs_source_dir
    )
    benchmark = _evaluation_records(evaluation_destination, tasks_dir, evaluation_ids)
    for row in training + evaluation:
        _install_verifier(tasks_dir / row["metadata"]["task_path"], problems_source=source_dir / "Frontier-CS")
    for row in benchmark:
        _install_verifier(tasks_dir / row["metadata"]["task_path"], problems_source=frontier_cs_source_dir)
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
                "excluded": [
                    {
                        "problem_id": row["problem_id"],
                        "frontiercs_id": row["frontiercs_id"],
                        "reason": "explicit_alias_of_evaluation_problem",
                    }
                    for row in aliases
                ],
                "evaluation_source_url": EVAL_SOURCE_URL,
                "evaluation_source_revision": EVAL_SOURCE_REVISION,
                "evaluation": evaluation_ids,
                "evaluation_excluded": [
                    {"problem_id": "61", "reason": "official_179_problem_scoring_set_excludes_changed_output_format"}
                ],
                "deduplication_scope": "Known parity aliases only; no claim of exhaustive semantic deduplication",
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
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--frontier-cs-source-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
