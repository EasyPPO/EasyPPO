"""Freeze validation replay and recover token-aligned native Miles predictions."""

import argparse
import gzip
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.miles_episodes import export_episodes
from agentic_critic_pipeline.records import data_sha256, validate_episode


def _inventory(path, split):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            ep = json.loads(line)
            validate_episode(ep)
            if ep["split"] != split or ep["episode_id"] in rows:
                raise ValueError(f"Duplicate identity or incorrect split in {path}")
            if not ep["problem_id"].startswith("frontiersmith:"):
                raise ValueError("Training and validation must both come from FrontierSmith")
            if ep["n_compactions"] != len(ep["segments"]) - 1:
                raise ValueError("Compaction count must match the complete segment history")
            rows[ep["episode_id"]] = {
                "problem_id": ep["problem_id"],
                "reward": ep["reward"],
                "n_compactions": ep["n_compactions"],
                "segments": [
                    {
                        "tokens": sum(s["loss_mask"]),
                        "positions_sha256": digest([i for i, bit in enumerate(s["loss_mask"]) if bit]),
                    }
                    for s in ep["segments"]
                ],
            }
    if not rows:
        raise ValueError(f"Empty {split} split")
    return rows


def prepare(episodes, training_episodes, out, *, group_size=32, trace_count=20, seed=1234):
    """Select trace IDs before any critic prediction is available."""
    if min(group_size, trace_count) < 1:
        raise ValueError("Group size and trace count must be positive")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a fresh validation directory: {out}")
    validation = _inventory(episodes, "validation")
    training = _inventory(training_episodes, "train")
    counts = Counter(row["problem_id"] for row in validation.values())
    if set(counts) != {row["problem_id"] for row in training.values()}:
        raise ValueError("Validation must cover the same FrontierSmith problems as training")
    if set(validation) & set(training) or set(counts.values()) != {group_size}:
        raise ValueError("Validation needs independent IDs and the exact per-prompt group size")
    if len(validation) < trace_count:
        raise ValueError("Validation cohort is smaller than the requested trace count")
    selected = random.Random(seed).sample(sorted(validation), trace_count)
    out.mkdir(parents=True)
    export_episodes(episodes, out / "episodes.jsonl")
    converter = Path(__file__).resolve().parents[1] / "scripts/agentic_critic/episodes_to_rollouts.py"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--episodes",
            str(out / "episodes.jsonl"),
            "--out",
            str(out),
            "--steps",
            str(len(counts)),
            "--problems-per-step",
            "1",
            "--episodes-per-problem",
            str(group_size),
            "--heldout-problems",
            "0",
            "--seed",
            str(seed),
            "--rounds",
            "--require-exhaustion",
        ],
        check=True,
    )
    plan = {
        "schema_version": 1,
        "split": "validation",
        "group_size": group_size,
        "seed": seed,
        "source": str(Path(episodes).resolve()),
        "source_sha256": data_sha256(episodes),
        "training_source_sha256": data_sha256(training_episodes),
        "selected_episode_ids": selected,
        "episodes": validation,
        "baseline_tokens": 1024,
        "profile_span": 3076,
        "replay": [
            {"path": str(out / f"rollout_{i}.pt"), "sha256": data_sha256(out / f"rollout_{i}.pt")}
            for i in range(len(counts))
        ],
    }
    plan["plan_hash"] = digest(plan)
    (out / "validation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def load_plan(path, *, verify_files=False):
    plan = json.loads(Path(path).read_text())
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if plan.get("schema_version") != 1 or digest(payload) != plan["plan_hash"]:
        raise ValueError("Validation plan checksum/schema mismatch")
    if verify_files:
        for row in plan["replay"]:
            if data_sha256(row["path"]) != row["sha256"]:
                raise ValueError(f'Validation replay changed: {row["path"]}')
        if data_sha256(plan["source"]) != plan["source_sha256"]:
            raise ValueError("Validation source changed")
    return plan


def validate_training_replay(plan, replay_directory):
    """Pair validation with the exact frozen source exported for training."""
    identities = json.loads((Path(replay_directory) / "episodes.identities.json").read_text())
    if identities["source_sha256"] != plan["training_source_sha256"]:
        raise ValueError("Validation plan and training replay use different frozen training sources")
    rows = identities["episodes"]
    episode_ids = {row["episode_id"] for row in rows}
    if len(episode_ids) != len(rows) or episode_ids.intersection(plan["episodes"]):
        raise ValueError("Training replay has duplicate IDs or overlaps validation")
    if {row["problem"] for row in rows} != {row["problem_id"] for row in plan["episodes"].values()}:
        raise ValueError("Validation and training replay must cover the same problems")


def validate_launch(args):
    if args.train_backend != "megatron" or not args.use_critic or args.fully_async or args.entry != "train":
        raise ValueError("Frozen critic validation requires synchronous Megatron critic training")
    if args.context_parallel_size != 1 or args.pipeline_model_parallel_size != 1:
        raise ValueError("Frozen critic value dumps currently require CP=1 and PP=1")
    if args.critic_validation_interval < 1 or not args.critic_validation_output:
        raise ValueError("Critic validation requires a positive interval and an output directory")
    for flag in ("delay_split_train_data_by_dp", "enable_witness", "use_dynamic_global_batch_size"):
        if getattr(args, flag, False):
            raise ValueError(f"Frozen critic validation does not support --{flag.replace('_', '-')}")
    if not getattr(args, "load_debug_rollout_data", None):
        raise ValueError("Frozen critic validation requires --load-debug-rollout-data for token-level dump storage")
    plan = load_plan(args.critic_validation_plan, verify_files=True)
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    tp = args.tensor_model_parallel_size
    if world < 1 or tp < 1 or world % tp:
        raise ValueError("Validation requires a positive world size divisible by tensor parallelism")
    dp = world // tp
    if plan["group_size"] % dp:
        raise ValueError("Validation group size must divide evenly across data-parallel ranks")


def _chunk_values(plan, index, directory, mode, batch):
    # Torch is needed only for native dumps, not for planning/metrics utilities.
    import torch

    from agentic_critic_pipeline.validation_storage import validate_compact_episode

    replay = torch.load(plan["replay"][index]["path"], map_location="cpu", weights_only=False)["samples"]
    lookup = {}
    for path in sorted(directory.glob("*.critic.pt")):
        dump = torch.load(path, map_location="cpu", weights_only=False)
        if (
            dump.get("schema_version") != 2
            or dump.get("value_bin_size") != 100
            or dump["rollout_id"] != batch
            or dump["full_history"] != (mode == "full_history")
        ):
            raise ValueError("Validation dump schema/batch/context mismatch")
        for row in dump["episodes"]:
            key = int(row["rollout_id"])
            if key in lookup:
                raise ValueError("Duplicated validation values across ranks")
            lookup[key] = row["prediction"]
    grouped = defaultdict(list)
    for sample in replay:
        grouped[int(sample["rollout_id"])].append(sample)
    if lookup.keys() != grouped.keys():
        raise ValueError("Missing or unexpected validation rank/sample dumps")
    for key, samples in grouped.items():
        identity = samples[0]["metadata"]["sample"]
        frozen = plan["episodes"][identity]
        episode = lookup[key]
        if episode["episode_id"] != str(key):
            raise ValueError("Validation dump rollout identity mismatch")
        validate_compact_episode(episode, frozen)
        if len(samples) != len(episode["segments"]):
            raise ValueError("Incomplete validation trajectory")
        for sample, segment in zip(samples, episode["segments"], strict=True):
            positions = np.flatnonzero(sample["loss_mask"]).tolist()
            starts = range(0, len(positions), 100)
            ranges = [[positions[start], positions[min(start + 100, len(positions)) - 1] + 1] for start in starts]
            if (
                sample["metadata"]["sample"] != identity
                or sample["metadata"]["segment"] != segment["segment_index"]
                or sample["metadata"]["problem"] != frozen["problem_id"]
                or sample["reward"] != episode["reward"]
                or digest(positions) != segment["positions_sha256"]
                or ranges != segment["response_position_ranges"]
            ):
                raise ValueError("Validation value/mask alignment failed")
        episode = dict(
            episode, episode_id=identity, problem_id=frozen["problem_id"], mode=mode, batch=batch, split="validation"
        )
        episode.pop("data_sha256")
        episode["data_sha256"] = digest(episode)
        yield episode


def finalize(plan, directory, *, mode, batch):
    from agentic_critic_pipeline.validation_metrics import ValidationMetrics

    directory = Path(directory)
    # Persist only 100-token means; exact EV/jumps come from scalar sufficient statistics.
    result = ValidationMetrics(plan, mode=mode, batch=batch, collect_profiles=False)
    seen = set()
    with gzip.open(directory / "values.jsonl.gz", "wt") as values, gzip.open(
        directory / "selected_trajectories.jsonl.gz", "wt"
    ) as traces:
        for index in range(len(plan["replay"])):
            for episode in _chunk_values(plan, index, directory / f"chunk_{index:04d}", mode, batch):
                if episode["episode_id"] in seen:
                    raise ValueError("Repeated validation trajectory across chunks")
                seen.add(episode["episode_id"])
                values.write(json.dumps(episode, separators=(",", ":")) + "\n")
                result.add(episode, traces=traces)
    if seen != plan["episodes"].keys():
        raise ValueError("Validation did not evaluate the entire frozen cohort")
    summary = result.summary()
    summary.update(
        schema_version=2, value_bin_size=100, profile_definition="absolute deviation of 100-token mean values"
    )
    summary["plan_hash"] = plan["plan_hash"]
    temporary = directory / "metrics.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "metrics.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episodes", "training-episodes", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--trace-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    prepare(
        args.episodes,
        args.training_episodes,
        args.out,
        group_size=args.group_size,
        trace_count=args.trace_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
