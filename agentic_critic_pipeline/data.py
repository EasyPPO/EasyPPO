"""Freeze problem identities and rollout seeds before generating any trajectory."""

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def training_quotas(problems, seed, batches=40, prompts_per_batch=8, trajectories_per_prompt=16):
    """Allocate exactly the episode groups consumed by the fixed training run."""
    identifiers = sorted(problem["problem_id"] for problem in problems)
    groups = batches * prompts_per_batch
    if not identifiers or len(set(identifiers)) != len(identifiers) or len(identifiers) > groups:
        raise ValueError("Training problems must be distinct and fit the available prompt groups")
    order = identifiers[:]
    random.Random(f"{seed}:training-quotas").shuffle(order)
    base, extra = divmod(groups, len(order))
    return {problem: (base + int(index < extra)) * trajectories_per_prompt for index, problem in enumerate(order)}


def collection_quotas(manifest, split):
    """Resolve either an exact per-problem quota map or a legacy uniform quota."""
    identifiers = [problem["problem_id"] for problem in manifest["splits"][split]]
    sampling = manifest["sampling"][split]
    if "quota_by_problem" in sampling:
        quotas = sampling["quota_by_problem"]
        if set(quotas) != set(identifiers):
            raise ValueError("Collection quota map does not cover the exact split")
    else:
        quotas = {problem: sampling["samples_per_problem"] for problem in identifiers}
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ValueError("Collection quotas must be positive integers")
    if "total_episodes" in sampling and sum(quotas.values()) != sampling["total_episodes"]:
        raise ValueError("Collection quota total disagrees with the manifest")
    return dict(quotas)


def inventory(root, dataset):
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "config.yaml").exists():
            continue
        config = yaml.safe_load((path / "config.yaml").read_text())
        if config.get("type", "default") != "default":
            continue  # declared cohort: noninteractive C++/testlib tasks
        statement = path / "statement.txt"
        tests = sorted((path / "testdata").glob("*.in"))
        if not statement.exists() or not tests or any(not p.with_suffix(".ans").exists() for p in tests):
            raise ValueError(f"Incomplete task {path}")
        hashes = {}
        for file in sorted(path.rglob("*")):
            if file.is_file():
                hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
        rows.append(
            {
                "problem_id": f"{dataset}:{path.name}",
                "judge_id": path.name,
                "dataset": dataset,
                "path": str(path),
                "statement_sha256": hashes["statement.txt"],
                "files_sha256": hashes,
                "source_revision": revision,
                "num_cases": len(tests),
            }
        )
    return rows


def verify_problem(problem):
    root = Path(problem["path"])
    for name, expected in problem["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Problem changed since manifest creation: {root / name}")


def qualify_inventory(rows, root, dataset, report_path):
    """Apply an immutable checker-input preflight before selecting any tasks."""
    if report_path is None:
        raise ValueError(f"A complete {dataset} quality preflight report is required")
    path = Path(report_path).resolve()
    payload = path.read_bytes()
    report = json.loads(payload)
    if report.get("schema_version") != 1 or report.get("scope") != "empty_output_checker_preflight":
        raise ValueError("Unsupported task quality report schema or scope")
    if report.get("status") != "complete" or report.get("fatal_errors") != []:
        raise ValueError("Quality preflight did not complete without infrastructure errors")
    if report.get("dataset") != dataset or Path(report["root"]).resolve() != Path(root).resolve():
        raise ValueError("Quality report belongs to a different dataset root")
    tasks = report["tasks"]
    if set(tasks) != {row["judge_id"] for row in rows}:
        raise ValueError("Quality report does not cover the exact noninteractive task inventory")
    eligible, excluded = [], []
    for row in rows:
        task = tasks[row["judge_id"]]
        if report["source_revision"] != row["source_revision"] or task["files_sha256"] != row["files_sha256"]:
            raise ValueError(f"Task changed after quality preflight: {row['problem_id']}")
        status = task["status"]
        if status == "passed":
            eligible.append(row)
        elif status == "failed" and task.get("reasons"):
            excluded.append({"problem_id": row["problem_id"], "reasons": task["reasons"]})
        else:
            raise ValueError(f"Incomplete or ambiguous quality verdict: {row['problem_id']}")
    return eligible, {
        "report_path": str(path),
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "source_revision": report["source_revision"],
        "scope": report["scope"],
        "inventory_count": len(rows),
        "eligible_count": len(eligible),
        "eligible_problem_ids": sorted(row["problem_id"] for row in eligible),
        "excluded": excluded,
        "limitations": report.get("limitations", []),
    }


def prepare(
    smith_root,
    frontier_root,
    train_count=120,
    test_count=20,
    seed=20261021,
    smith_quality_report=None,
    frontier_quality_report=None,
):
    train = inventory(smith_root, "frontiersmith")
    test = inventory(frontier_root, "frontiercs")
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    # Preserve the complete seeded order, then take the first qualifying tasks.
    # Report verdicts were frozen before any actor trajectory is collected.
    train, smith_quality = qualify_inventory(train, smith_root, "frontiersmith", smith_quality_report)
    test, frontier_quality = qualify_inventory(test, frontier_root, "frontiercs", frontier_quality_report)
    if len(train) < train_count or len(test) < test_count:
        raise ValueError(f"Not enough default tasks: train={len(train)}, test={len(test)}")
    train = train[:train_count]
    # Reject verbatim cross-dataset duplicates before test selection.
    train_hashes = {r["statement_sha256"] for r in train}
    test = [r for r in test if r["statement_sha256"] not in train_hashes]
    if len(test) < test_count:
        raise ValueError("Not enough nonduplicate test tasks")
    manifest = {
        "schema_version": 2,
        "seed": seed,
        "cohort": "type: default; complete checker-input preflight passed",
        "selection_order": "Seeded shuffle of full inventory, then quality exclusion, then first requested count",
        "quality_preflight": {"frontiersmith": smith_quality, "frontiercs": frontier_quality},
        "splits": {"train": train, "validation": train, "test": test[:test_count]},
        "sampling_semantics": (
            "quota_by_problem defines exact training collection counts; samples_per_problem is the uniform quota; "
            "sample_start applies only to fixed diagnostic rollouts"
        ),
        "candidate_namespaces": {
            "train": {"offset": 0, "stride": 3},
            "validation": {"offset": 1, "stride": 3},
            "test": {"offset": 2, "stride": 3},
        },
        "sampling": {
            "train": {
                "quota_by_problem": training_quotas(train, seed),
                "total_episodes": 5120,
                "episodes_per_group": 16,
                "training_batches": 40,
                "prompts_per_batch": 8,
                "sample_start": 0,
            },
            "validation": {"samples_per_problem": 8, "sample_start": 64},
            "test": {"samples_per_problem": 8, "sample_start": 0},
        },
        "validation_note": "Held-out rollout seeds on the 120 training problems; test uses a separate dataset.",
    }
    manifest["manifest_hash"] = digest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smith-root", required=True)
    parser.add_argument("--frontier-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smith-quality-report", required=True)
    parser.add_argument("--frontier-quality-report", required=True)
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    output = Path(args.output)
    manifest = prepare(
        args.smith_root,
        args.frontier_root,
        args.train_count,
        args.test_count,
        args.seed,
        args.smith_quality_report,
        args.frontier_quality_report,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError("Existing manifest differs; use a new output path")
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest_hash": manifest["manifest_hash"],
                "split_counts": {k: len(v) for k, v in manifest["splits"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
