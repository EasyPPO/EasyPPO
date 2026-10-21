"""Freeze a common critic cohort without changing or truncating raw trajectories.

Length is the sum of actor segments, including duplicated resume prompts.
Optional strict batching defaults to 8 problems x 16 episodes per batch.
Every episode unused by this schedule is reported separately.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .records import batch_schedule, validate_episode


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(rows):
    if not rows:
        return {"episodes": 0, "problems": 0, "length": {}, "reward_mean": None, "compactions": {}}
    lengths = sorted(row["full_length"] for row in rows)
    return {
        "episodes": len(rows),
        "problems": len({row["problem_id"] for row in rows}),
        "length": {label: lengths[round(q * (len(lengths) - 1))] for label, q in (("min", 0), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1))},
        "reward_mean": sum(row["reward"] for row in rows) / len(rows),
        "compactions": dict(sorted(Counter(row["n_compactions"] for row in rows).items())),
    }


def balanced_selection(rows, problems_per_batch=8, episodes_per_problem=16, seed=1234):
    """Maximal nested complete batches; return all unused IDs with reasons."""
    if min(problems_per_batch, episodes_per_problem) < 1:
        raise ValueError("Batch dimensions must be positive")
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["episode_id"]):
        grouped[row["problem_id"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    capacities = {pid: len(values) // episodes_per_problem for pid, values in grouped.items()}
    problems = [pid for pid in grouped if capacities[pid]]
    rng.shuffle(problems)  # Tie-break equal capacities without lexicographic selection bias.
    problems.sort(key=lambda pid: capacities[pid], reverse=True)
    kept_chunks = dict.fromkeys(grouped, 0)
    groups = []
    for start in range(0, len(problems), problems_per_batch):
        block = problems[start : start + problems_per_batch]
        common = min(capacities[pid] for pid in block) if len(block) == problems_per_batch else 0
        for pid in block:
            kept_chunks[pid] = common
        groups.append(
            {
                "problem_ids": block,
                "original_chunk_counts": {p: capacities[p] for p in block},
                "selected_chunks_per_problem": common,
                "complete_problem_group": len(block) == problems_per_batch,
            }
        )
    selected, unused = [], []
    for pid, values in grouped.items():
        selected_count = kept_chunks[pid] * episodes_per_problem
        selected.extend(values[:selected_count])
        for index, row in enumerate(values[selected_count:], start=selected_count):
            if not capacities[pid]:
                reason = "fewer_than_episodes_per_problem"
            elif index >= capacities[pid] * episodes_per_problem:
                reason = "per_problem_remainder"
            elif not kept_chunks[pid]:
                reason = "incomplete_problem_group"
            else:
                reason = "round_balance"
            unused.append({**row, "reason": reason})
    return sorted(selected, key=lambda r: r["episode_id"]), sorted(unused, key=lambda r: r["episode_id"]), groups


def freeze(
    input_path,
    output_dir,
    max_full_tokens,
    *,
    balance=False,
    seed=1234,
    epochs=1,
    required_problems=None,
    required_episodes_per_problem=None,
    problems_per_batch=8,
    episodes_per_problem=16,
    required_quota_by_problem=None,
):
    if min(problems_per_batch, episodes_per_problem) < 1:
        raise ValueError("Batch dimensions must be positive")
    if required_quota_by_problem is not None and (not required_quota_by_problem or any(not isinstance(value, int) or value < 1 for value in required_quota_by_problem.values())):
        raise ValueError("Per-problem quotas must be positive integers")
    if max_full_tokens < 1 or epochs < 1:
        raise ValueError("Length limit and epoch count must be positive")
    if any(value is not None and value < 1 for value in (required_problems, required_episodes_per_problem)):
        raise ValueError("Required problem and episode counts must be positive")
    source, output = Path(input_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Frozen cohort output already exists: {output}")
    included, excluded, records, splits = [], [], {}, set()
    source_hash = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            source_hash.update(line)
            episode = json.loads(line)
            validate_episode(episode)
            identity = episode["episode_id"]
            if identity in records:
                raise ValueError(f"Duplicate episode ID {identity}")
            splits.add(episode["split"])
            row = {
                "episode_id": identity,
                "problem_id": episode["problem_id"],
                "split": episode["split"],
                "full_length": sum(len(s["input_ids"]) for s in episode["segments"]),
                "n_compactions": episode["n_compactions"],
                "reward": episode["reward"],
            }
            records[identity] = (offset, hashlib.sha256(line).hexdigest())
            if row["full_length"] < max_full_tokens:
                included.append(row)
            else:
                excluded.append({**row, "reason": "full_history_at_or_above_limit"})
    if len(splits) != 1 or not records:
        raise ValueError("Input must contain exactly one nonempty split")
    split = next(iter(splits))
    if balance and split != "train":
        raise ValueError("Training batch balancing is forbidden for held-out splits")
    included.sort(key=lambda row: row["episode_id"])
    excluded.sort(key=lambda row: row["episode_id"])
    selected, unused, groups = balanced_selection(included, problems_per_batch, episodes_per_problem, seed) if balance else (included, [], [])
    included_counts = Counter(row["problem_id"] for row in included)
    quota_errors = []
    if required_quota_by_problem is not None and dict(included_counts) != required_quota_by_problem:
        quota_errors.append("Eligible per-problem counts differ from the exact manifest quotas")
    if required_problems is not None and len(included_counts) != required_problems:
        quota_errors.append(f"Expected {required_problems} eligible problems, found {len(included_counts)}")
    if required_episodes_per_problem is not None:
        wrong_counts = {problem: count for problem, count in included_counts.items() if count != required_episodes_per_problem}
        if wrong_counts:
            quota_errors.append(f"Expected {required_episodes_per_problem} eligible episodes per problem: {wrong_counts}")
    if (required_problems is not None or required_episodes_per_problem is not None or required_quota_by_problem is not None) and unused:
        quota_errors.append("Exact-quota cohorts cannot silently discard episodes during batch balancing")
    output.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    def copy_records(name, rows):
        target = output / name
        temporary = target.with_suffix(".jsonl.tmp")
        with source.open("rb") as src, temporary.open("xb") as dst:
            for row in rows:
                offset, checksum = records[row["episode_id"]]
                src.seek(offset)
                line = src.readline()
                if hashlib.sha256(line).hexdigest() != checksum:
                    raise ValueError("Source changed while freezing the cohort")
                dst.write(line.rstrip(b"\n") + b"\n")
        temporary.replace(target)
        return sha256_file(target)

    filtered_hash = copy_records("length_included.jsonl", included)
    write_json("included.json", included)
    write_json("excluded.json", excluded)
    write_json("batch_unused.json", unused)
    all_rows = included + excluded
    counts = {pid: {"raw": 0, "length_included": 0, "length_excluded": 0, "training_selected": 0, "batch_unused": 0} for pid in sorted({r["problem_id"] for r in all_rows})}
    for label, rows in (
        ("raw", all_rows),
        ("length_included", included),
        ("length_excluded", excluded),
        ("training_selected", selected if balance else []),
        ("batch_unused", unused),
    ):
        for row in rows:
            counts[row["problem_id"]][label] += 1
    report = {
        "schema_version": 1,
        "source_path": str(source),
        "source_sha256": source_hash.hexdigest(),
        "split": split,
        "max_full_tokens": max_full_tokens,
        "length_comparison": "strictly_less_than",
        "required_problems": required_problems,
        "required_episodes_per_problem": required_episodes_per_problem,
        "required_quota_by_problem": required_quota_by_problem,
        "quota_errors": quota_errors,
        "length_definition": "sum(segment.input_ids lengths), including duplicated resume context",
        "raw": describe(all_rows),
        "length_included": describe(included),
        "length_excluded": describe(excluded),
        "length_included_sha256": filtered_hash,
        "per_problem_counts": counts,
        "batch_balancing_enabled": balance,
        "problems_per_batch": problems_per_batch if balance else None,
        "episodes_per_problem": episodes_per_problem if balance else None,
        "trajectories_per_batch": problems_per_batch * episodes_per_problem if balance else None,
        "batch_unused": describe(unused),
        "batch_unused_reasons": dict(Counter(row["reason"] for row in unused)),
        "training_selected": describe(selected) if balance else None,
        "selection_seed": seed if balance else None,
        "balancing_groups": groups,
        "batching_policy": (f"This optional schedule strictly uses {problems_per_batch} problems x {episodes_per_problem} episodes. The same selected IDs and seed must be used by both critic arms."),
        "status": "frozen",
    }
    if quota_errors:
        report["status"] = "failed_incomplete_exact_quota"
    elif balance and selected:
        # Verify against the actual trainer using metadata only, with no token tensors.
        schedule = batch_schedule(selected, problems_per_batch, episodes_per_problem, epochs, seed, allow_incomplete=False)
        frozen_schedule = []
        for index, batch in enumerate(schedule):
            frozen_schedule.append(
                {
                    "batch_index": index,
                    "episode_ids": [row["episode_id"] for row in batch],
                    "per_problem_counts": dict(Counter(row["problem_id"] for row in batch)),
                }
            )
        write_json("schedule.json", {"seed": seed, "epochs": epochs, "batches": frozen_schedule})
        report["schedule_sha256"] = sha256_file(output / "schedule.json")
        report["training_selected_sha256"] = copy_records("train_ready.jsonl", selected)
        report["logical_batches"] = len(schedule)
    elif balance:
        report["status"] = "failed_no_complete_training_batch"
    elif not included:
        report["status"] = "failed_no_length_eligible_episodes"
    if sha256_file(source) != report["source_sha256"]:
        raise ValueError("Source file changed during cohort construction")
    write_json("cohort.json", report)
    if quota_errors:
        raise ValueError(f"Incomplete exact quota; collect additional eligible episodes. Details saved in {output}")
    if balance and not selected:
        raise ValueError(f"No complete {problems_per_batch} x {episodes_per_problem} training batch; length results and all unused IDs saved in {output}")
    if not included:
        raise ValueError(f"No episode meets the length limit; exclusion report saved in {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", help="Require exact per-problem quotas from this frozen manifest")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-full-tokens", type=int, required=True)
    parser.add_argument("--balance-training-batches", action="store_true")
    parser.add_argument("--problems-per-batch", type=int, default=8)
    parser.add_argument("--episodes-per-problem", type=int, default=16)
    parser.add_argument("--required-problems", type=int)
    parser.add_argument("--required-episodes-per-problem", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    quotas = None
    if args.manifest:
        from .data import collection_quotas, digest

        manifest = json.loads(Path(args.manifest).read_text())
        declared = manifest.pop("manifest_hash")
        if digest(manifest) != declared:
            raise ValueError("Manifest checksum mismatch")
        quotas = collection_quotas(manifest, args.split)
    report = freeze(
        args.input,
        args.output,
        args.max_full_tokens,
        balance=args.balance_training_batches,
        required_quota_by_problem=quotas,
        problems_per_batch=args.problems_per_batch,
        episodes_per_problem=args.episodes_per_problem,
        seed=args.seed,
        epochs=args.epochs,
        required_problems=args.required_problems,
        required_episodes_per_problem=args.required_episodes_per_problem,
    )
    print(json.dumps({key: report[key] for key in ("status", "split", "raw", "length_included", "batch_unused")}))


if __name__ == "__main__":
    main()
