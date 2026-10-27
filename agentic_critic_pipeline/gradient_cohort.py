"""Deterministic segment sampling and validated worker assignments."""

import hashlib
import json
import random
from pathlib import Path

from agentic_critic_pipeline.records import data_sha256


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def assign_workers(rows, workers, cost):
    """Greedy longest-first assignment; each row appears on exactly one worker."""
    loads = [0] * workers
    result = []
    for row in sorted(rows, key=lambda row: (-row[cost], row["episode_id"], row.get("segment_index", 0))):
        worker = min(range(workers), key=lambda index: (loads[index], index))
        loads[worker] += row[cost]
        result.append(dict(row, worker=worker))
    return result


def sample_segments(inventory, *, count, minimum, window_size, workers, seed):
    if min(count, minimum, window_size, workers) < 1 or minimum < 2 * window_size:
        raise ValueError("Positive counts required; minimum must allow a target and a disjoint control")
    eligible = [row for row in inventory if row["segment_index"] > 0 and row["loss_tokens"] >= minimum]
    if len(eligible) < count:
        raise ValueError(
            f"Need exactly {count} eligible segments, found {len(eligible)}; do not silently shrink the sample"
        )
    eligible.sort(key=lambda row: (row["episode_id"], row["segment_index"]))
    chosen = random.Random(seed).sample(eligible, count)
    return assign_workers(chosen, workers, "cost"), len(eligible)


def load_selection(path, *, episodes, shard, window_size, seed):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 2 or plan.get("plan_sha256") != plan_digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("Selection plan checksum/schema mismatch")
    if plan["cohort_sha256"] != data_sha256(episodes):
        raise ValueError("Selection plan uses a different cohort")
    if not 0 <= shard < plan["workers"] or window_size != plan["window_size"] or seed != plan["seed"]:
        raise ValueError("Worker shard/window size/seed disagrees with the frozen selection plan")
    keys = [(row["episode_id"], row["segment_index"]) for row in plan["selected"]]
    if len(keys) != plan["segments"] or len(set(keys)) != len(keys):
        raise ValueError("Selection plan has duplicate or missing segments")
    return {key for key, row in zip(keys, plan["selected"], strict=True) if row["worker"] == shard}
