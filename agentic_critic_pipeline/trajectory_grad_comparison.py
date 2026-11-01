"""Compare actor gradients under two independently pretrained critics.

Run `python -m agentic_critic_pipeline.trajectory_grad_comparison run --help`.
Critic caching and actor measurement are separate, sequential GPU phases.
"""

import argparse
import json
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from agentic_critic_pipeline.compact_grad import (
    _cached_pairs,
    _checkpoint_identity,
    _episodes,
    _json,
    _provenance,
)
from agentic_critic_pipeline.compact_grad_parallel import merge_caches, run_jobs, worker_devices
from agentic_critic_pipeline.gradient_cohort import assign_workers, plan_digest
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.trajectory_grad import measure_trajectory

MODES = ("actor_summary", "full_history")


def prepare(args, root):
    source_hash = data_sha256(args.episodes)
    inventory = []
    for episode in _episodes(args.episodes):
        lengths = [len(s["input_ids"]) for s in episode["segments"]]
        inventory.append(
            dict(
                episode_id=episode["episode_id"],
                problem_id=episode["problem_id"],
                split=episode["split"],
                total_compactions=len(lengths) - 1,
                segments=len(lengths),
                context_tokens=sum(lengths),
                loss_tokens=sum(sum(s["loss_mask"]) for s in episode["segments"]),
                cost=sum(lengths) ** 2,
            )
        )
    inventory.sort(key=lambda row: row["episode_id"])
    if len({row["split"] for row in inventory}) != 1:
        raise ValueError("Supply one frozen split, not a mixture of training and validation")
    if args.trajectories is not None:
        if args.trajectories > len(inventory):
            raise ValueError("Requested more trajectories than available; sample size will not be reduced")
        chosen = random.Random(args.seed).sample(inventory, args.trajectories)
    else:
        chosen = inventory
    if any(row["context_tokens"] > args.max_tokens for row in chosen):
        raise ValueError("Selected full history exceeds --max-tokens; truncation is forbidden")
    assignment = assign_workers(chosen, args.workers, "cost")
    lookup = {row["episode_id"]: row["worker"] for row in assignment}
    root.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        cohort = stack.enter_context((root / "cohort.jsonl").open("w"))
        shards = {
            worker: stack.enter_context((root / f"cache_input_{worker}.jsonl").open("w"))
            for worker in sorted(set(lookup.values()))
        }
        for episode in _episodes(args.episodes):
            if episode["episode_id"] in lookup:
                line = json.dumps(episode, allow_nan=False) + "\n"
                cohort.write(line)
                shards[lookup[episode["episode_id"]]].write(line)
    if data_sha256(args.episodes) != source_hash:
        raise ValueError("Source episodes changed during cohort preparation")
    plan = dict(
        schema_version=1,
        source_sha256=source_hash,
        cohort_sha256=data_sha256(root / "cohort.jsonl"),
        available_trajectories=len(inventory),
        trajectories=len(chosen),
        workers=args.workers,
        seed=args.seed,
        cache_assignment=assignment,
    )
    plan["plan_sha256"] = plan_digest(plan)
    _json(root / "selection.json", plan)
    return plan


def _load_plan(path, episodes, shard):
    plan = json.loads(Path(path).read_text())
    if plan["plan_sha256"] != plan_digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Selection plan checksum mismatch")
    if plan["cohort_sha256"] != data_sha256(episodes) or not 0 <= shard < plan["workers"]:
        raise ValueError("Selection plan cohort/shard mismatch")
    return {row["episode_id"] for row in plan["cache_assignment"] if row["worker"] == shard}


def measure(args):
    cache = Path(args.values)
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["source_sha256"] != data_sha256(args.episodes):
        raise ValueError("Critic cache uses a different cohort")
    if manifest["values_sha256"] != data_sha256(cache / "values.jsonl"):
        raise ValueError("Critic cache checksum mismatch")
    selected = _load_plan(args.selection_plan, args.episodes, args.shard_index)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        _provenance(args),
        actor_checkpoint=_checkpoint_identity(args.actor),
        critic_cache_sha256=data_sha256(cache / "manifest.json"),
        selection_sha256=data_sha256(args.selection_plan),
        device_name=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        cuda_runtime=torch.version.cuda,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.actor,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(args.device)
        .eval()
    )
    seen = set()
    with (output / "trajectories.jsonl").open("w") as trajectories, (output / "segments.jsonl").open("w") as segments:
        for episode, values in _cached_pairs(args.episodes, cache):
            if episode["episode_id"] not in selected:
                continue
            if sum(len(s["input_ids"]) for s in episode["segments"]) > args.max_tokens:
                raise ValueError("Full episode exceeds --max-tokens; truncation is forbidden")
            device = next(model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            trajectory, rows = measure_trajectory(
                model,
                episode,
                values,
                moments=manifest["advantage_moments"],
                normalization=args.advantage_normalization,
                head_chunk_size=args.head_chunk_size,
                checkpoint=args.checkpoint_layers,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            identity = dict(
                episode_id=episode["episode_id"],
                problem_id=episode["problem_id"],
                split=episode["split"],
                critic_mode=manifest["arguments"]["critic_mode"],
                reward=episode["reward"],
            )
            trajectory.update(
                identity,
                elapsed_seconds=time.perf_counter() - started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            )
            trajectories.write(json.dumps(trajectory, allow_nan=False) + "\n")
            trajectories.flush()
            for row in rows:
                segments.write(json.dumps(dict(row, **identity), allow_nan=False) + "\n")
            segments.flush()
            seen.add(episode["episode_id"])
            print(f"Measured {episode['episode_id']}: {trajectory['trajectory_grad_norm']}", flush=True)
    if seen != selected:
        raise ValueError("Missing selected trajectories; refusing partial results")
    for name in ("trajectories", "segments"):
        metadata[f"{name}_sha256"] = data_sha256(output / f"{name}.jsonl")
    _json(output / "manifest.json", metadata)


def merge_results(root, plan, mode, actor_identity, args):
    expected = {row["episode_id"]: row for row in plan["cache_assignment"]}
    trajectories, segments, manifests = {}, {}, []
    for worker in sorted({row["worker"] for row in expected.values()}):
        directory = root / f"measure_{worker}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["actor_checkpoint"] != actor_identity
            or manifest["critic_cache_sha256"] != data_sha256(root / "cache/manifest.json")
            or manifest["selection_sha256"] != data_sha256(root.parent / "selection.json")
        ):
            raise ValueError("Measurement model/cache/selection mismatch")
        for key in ("dtype", "qwen35_fla", "checkpoint_layers", "advantage_normalization", "head_chunk_size"):
            if manifest["arguments"][key] != getattr(args, key):
                raise ValueError("Measurement loss configuration mismatch")
        manifests.append(manifest)
        for name, target in (("trajectories", trajectories), ("segments", segments)):
            path = directory / f"{name}.jsonl"
            if manifest[f"{name}_sha256"] != data_sha256(path):
                raise ValueError("Measurement checksum mismatch")
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    eid = row["episode_id"]
                    key = eid if name == "trajectories" else (eid, row["segment_index"])
                    if (
                        key in target
                        or eid not in expected
                        or expected[eid]["worker"] != worker
                        or row["critic_mode"] != mode
                        or row["problem_id"] != expected[eid]["problem_id"]
                    ):
                        raise ValueError("Duplicate or unexpected measurement")
                    target[key] = row
    expected_segments = {(eid, i) for eid, row in expected.items() for i in range(row["segments"])}
    if trajectories.keys() != expected.keys() or segments.keys() != expected_segments:
        raise ValueError("Incomplete trajectory/segment coverage")
    for name, rows in (("trajectories", trajectories), ("segments", segments)):
        with (root / f"{name}.jsonl").open("w") as stream:
            for key in sorted(rows):
                stream.write(json.dumps(rows[key], allow_nan=False) + "\n")
    return manifests


def _run_condition(args, root, plan, mode, critic, actor_identity, critic_identity, devices):
    directory = root / mode
    directory.mkdir()
    workers = sorted({row["worker"] for row in plan["cache_assignment"]})
    for name in ["cohort.jsonl"] + [f"cache_input_{w}.jsonl" for w in workers]:
        (directory / name).symlink_to(Path("..") / name)
    common = ["--device", args.device, "--dtype", args.dtype, "--max-tokens", str(args.max_tokens)]
    if args.qwen35_fla:
        common.append("--qwen35-fla")
    cache_jobs = [
        (
            w,
            [
                sys.executable,
                "-m",
                "agentic_critic_pipeline.compact_grad",
                "cache-values",
                "--base-model",
                args.base_model or args.actor,
                "--critic",
                critic,
                "--critic-mode",
                mode,
                "--episodes",
                str(directory / f"cache_input_{w}.jsonl"),
                "--output",
                str(directory / f"cache_{w}"),
            ]
            + common,
        )
        for w in workers
    ]
    run_jobs(cache_jobs, devices, directory, "cache")
    merge_caches(directory, plan)
    cache_manifest = json.loads((directory / "cache/manifest.json").read_text())
    if cache_manifest["critic_checkpoint"] != critic_identity:
        raise ValueError("Critic checkpoint changed during the experiment")
    measure_jobs = [
        (
            w,
            [
                sys.executable,
                "-m",
                "agentic_critic_pipeline.trajectory_grad_comparison",
                "measure",
                "--actor",
                args.actor,
                "--episodes",
                str(root / "cohort.jsonl"),
                "--values",
                str(directory / "cache"),
                "--selection-plan",
                str(root / "selection.json"),
                "--shard-index",
                str(w),
                "--output",
                str(directory / f"measure_{w}"),
                "--advantage-normalization",
                args.advantage_normalization,
                "--head-chunk-size",
                str(args.head_chunk_size),
                "--checkpoint-layers" if args.checkpoint_layers else "--no-checkpoint-layers",
            ]
            + common,
        )
        for w in workers
    ]
    run_jobs(measure_jobs, devices, directory, "measure")
    return merge_results(directory, plan, mode, actor_identity, args)


def execute(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    devices = worker_devices(args)
    checkpoints = {mode: _checkpoint_identity(getattr(args, f"{mode}_critic")) for mode in MODES}
    if checkpoints[MODES[0]] == checkpoints[MODES[1]]:
        raise ValueError("Supply two independently pretrained critic exports, not the same checkpoint")
    metadata = dict(_provenance(args), actor_checkpoint=_checkpoint_identity(args.actor), critics=checkpoints)
    metadata["base_checkpoint"] = (
        metadata["actor_checkpoint"] if not args.base_model else _checkpoint_identity(args.base_model)
    )
    plan = prepare(args, root)
    workers = {}
    for mode in MODES:
        workers[mode] = _run_condition(
            args,
            root,
            plan,
            mode,
            getattr(args, f"{mode}_critic"),
            metadata["actor_checkpoint"],
            checkpoints[mode],
            devices,
        )
        cache = json.loads((root / mode / "cache/manifest.json").read_text())
        if cache["base_checkpoint"] != metadata["base_checkpoint"]:
            raise ValueError("Critic base checkpoint changed during the experiment")
    if data_sha256(args.episodes) != metadata["source_sha256"]:
        raise ValueError("Source episodes changed during the experiment")
    # Deferred so workers need no plotting backend.
    from agentic_critic_pipeline.trajectory_grad_figures import render

    render(root, seed=args.seed, bootstrap=args.bootstrap)
    metadata.update(status="complete", selection=plan, worker_manifests=workers)
    metadata["results_sha256"] = {
        f"{mode}/{name}.jsonl": data_sha256(root / mode / f"{name}.jsonl")
        for mode in MODES
        for name in ("trajectories", "segments")
    }
    _json(root / "manifest.json", metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Paired critic caching, gradient measurements, data and figures")
    for name in ("actor-summary-critic", "full-history-critic"):
        run.add_argument(f"--{name}", required=True, help="Separate independently pretrained HF critic export")
    run.add_argument("--base-model", help="HF base of both exported critics; defaults to actor")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--gpus", help="Visible GPU indices, one per worker")
    run.add_argument(
        "--trajectories", type=int, help="Uniform sample without replacement; default: all supplied episodes"
    )
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--bootstrap", type=int, default=2000)
    worker = commands.add_parser("measure", help="Internal actor measurement worker")
    for name in ("values", "selection-plan"):
        worker.add_argument(f"--{name}", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    for command in (run, worker):
        for name in ("actor", "episodes", "output"):
            command.add_argument(f"--{name}", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        command.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
        command.add_argument("--max-tokens", type=int, default=120000)
        command.add_argument("--head-chunk-size", type=int, default=128)
        command.add_argument("--advantage-normalization", choices=("none", "cohort"), default="none")
        command.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--qwen35-fla", action="store_true")
    args = parser.parse_args(argv)
    for name in ("workers", "trajectories", "bootstrap", "max_tokens", "head_chunk_size"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.qwen35_fla and (args.device != "cuda" or args.dtype != "bfloat16"):
        parser.error("--qwen35-fla requires CUDA and bfloat16")
    if args.command == "measure" and args.qwen35_fla:
        # Optional deployment-only dependency.
        from agentic_critic_pipeline.kernels import configure_training_kernels

        configure_training_kernels()
    (execute if args.command == "run" else measure)(args)


if __name__ == "__main__":
    main()
