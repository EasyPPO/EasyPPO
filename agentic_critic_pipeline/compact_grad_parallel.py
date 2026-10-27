"""Sample 100 long post-compaction segments and run independent probes on eight GPUs.

Use --device cpu --workers 2 for a small local integration test. Production
defaults are eight CUDA workers, 100 segments, and at least 1024 actor tokens.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from agentic_critic_pipeline.compact_grad import _cached_pairs, _episodes, _json, _provenance
from agentic_critic_pipeline.gradient_cohort import assign_workers, plan_digest, sample_segments
from agentic_critic_pipeline.gradient_probe import summarize
from agentic_critic_pipeline.records import data_sha256


def prepare(args, root):
    inventory, episode_costs = [], []
    for episode in _episodes(args.episodes):
        lengths = [len(segment["input_ids"]) for segment in episode["segments"]]
        cost = sum(lengths) ** 2 if args.critic_mode == "full_history" else sum(n * n for n in lengths)
        episode_costs.append(dict(episode_id=episode["episode_id"], cost=cost))
        for index, segment in enumerate(episode["segments"]):
            inventory.append(
                dict(
                    episode_id=episode["episode_id"],
                    segment_index=index,
                    loss_tokens=sum(segment["loss_mask"]),
                    context_tokens=lengths[index],
                    cost=lengths[index] ** 2,
                )
            )
    selected, eligible = sample_segments(
        inventory,
        count=args.segments,
        minimum=args.min_segment_tokens,
        window_size=args.window_size,
        workers=args.workers,
        seed=args.seed,
    )
    chosen_ids = {row["episode_id"] for row in selected}
    cache_assignment = assign_workers(
        [row for row in episode_costs if row["episode_id"] in chosen_ids], args.workers, "cost"
    )
    cache_workers = {row["episode_id"]: row["worker"] for row in cache_assignment}
    root.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        cohort = stack.enter_context((root / "cohort.jsonl").open("w"))
        shards = {
            worker: stack.enter_context((root / f"cache_input_{worker}.jsonl").open("w"))
            for worker in sorted(set(cache_workers.values()))
        }
        for episode in _episodes(args.episodes):
            if episode["episode_id"] not in chosen_ids:
                continue
            lengths = [len(segment["input_ids"]) for segment in episode["segments"]]
            limit = sum(lengths) if args.critic_mode == "full_history" else max(lengths)
            if limit > args.max_tokens:
                raise ValueError("Selected episode exceeds --max-tokens; raise the limit or supply a qualified cohort")
            line = json.dumps(episode, allow_nan=False) + "\n"
            cohort.write(line)
            shards[cache_workers[episode["episode_id"]]].write(line)
    plan = dict(
        schema_version=2,
        source_sha256=data_sha256(args.episodes),
        cohort_sha256=data_sha256(root / "cohort.jsonl"),
        segments=args.segments,
        min_segment_tokens=args.min_segment_tokens,
        eligible_segments=eligible,
        workers=args.workers,
        seed=args.seed,
        window_size=args.window_size,
        selected=selected,
        cache_assignment=cache_assignment,
    )
    plan["plan_sha256"] = plan_digest(plan)
    _json(root / "selection.json", plan)
    return plan


def worker_devices(args):
    if args.device == "cpu":
        if args.gpus is not None:
            raise ValueError("--gpus is only valid with --device cuda")
        return [None] * args.workers
    indices = list(range(args.workers)) if args.gpus is None else [int(value) for value in args.gpus.split(",")]
    if len(indices) != args.workers or len(set(indices)) != len(indices):
        raise ValueError("--gpus must contain exactly --workers distinct visible GPU indices")
    if any(index < 0 or index >= torch.cuda.device_count() for index in indices):
        raise ValueError("Requested GPUs are unavailable; no automatic CPU fallback")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [visible.split(",")[index].strip() if visible is not None else str(index) for index in indices]


def run_jobs(jobs, devices, root, phase):
    """Launch all workers concurrently; terminate siblings if any worker fails."""
    processes = []
    with ExitStack() as stack:
        try:
            for worker, command in jobs:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = devices[worker] if devices[worker] is not None else ""
                env.setdefault("OMP_NUM_THREADS", "1")
                log = root / f"{phase}_{worker}.log"
                stream = stack.enter_context(log.open("w"))
                proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                processes.append((proc, log))
            pending = list(processes)
            while pending:
                for proc, log in pending[:]:
                    code = proc.poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"{phase} worker exited {code}; inspect {log}")
                    pending.remove((proc, log))
                    print(f"Completed {log.name}", flush=True)
                if pending:
                    time.sleep(0.2)
        finally:
            for proc, _ in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc, _ in processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def merge_caches(root, plan):
    """Merge in cohort order and compute ONE whitening statistic across all workers."""
    lookup, manifests = {}, []
    for worker in sorted({row["worker"] for row in plan["cache_assignment"]}):
        directory = root / f"cache_{worker}"
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "values.jsonl"
        if manifest["source_sha256"] != data_sha256(root / f"cache_input_{worker}.jsonl") or manifest[
            "values_sha256"
        ] != data_sha256(path):
            raise ValueError("Worker cache checksum mismatch")
        manifests.append(manifest)
        with path.open() as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                identity = json.loads(line)["episode_id"]
                if identity in lookup:
                    raise ValueError("Duplicate episode in cache workers")
                lookup[identity] = (path, offset)
    for manifest in manifests[1:]:
        if any(manifest[key] != manifests[0][key] for key in ("base_checkpoint", "critic_checkpoint")) or any(
            manifest["arguments"][key] != manifests[0]["arguments"][key]
            for key in ("critic_mode", "dtype", "qwen35_fla", "max_tokens")
        ):
            raise ValueError("Cache workers used different critic configurations")
    output = root / "cache"
    output.mkdir()
    with (output / "values.jsonl").open("w") as stream:
        for episode in _episodes(root / "cohort.jsonl"):
            path, offset = lookup.pop(episode["episode_id"])
            with path.open() as source:
                source.seek(offset)
                stream.write(source.readline())
    if lookup:
        raise ValueError("Unexpected episodes in worker caches")
    count, mean, m2, episodes = 0, 0.0, 0.0, 0
    for episode, values in _cached_pairs(root / "cohort.jsonl", output):
        flat = torch.tensor([episode["reward"] - v for segment in values for v in segment], dtype=torch.float64)
        n, current = flat.numel(), float(flat.mean())
        delta = current - mean
        m2 += float(((flat - current) ** 2).sum()) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        episodes += 1
    manifest = dict(
        manifests[0],
        source_sha256=plan["cohort_sha256"],
        episodes=episodes,
        advantage_moments=dict(count=count, mean=mean, variance=m2 / max(1, count - 1)),
        values_sha256=data_sha256(output / "values.jsonl"),
        worker_manifests=manifests,
    )
    manifest["arguments"] = dict(manifest["arguments"], episodes=str(root / "cohort.jsonl"), output=str(output))
    _json(output / "manifest.json", manifest)


def merge_results(root, plan, *, seed, bootstrap):
    expected = {(row["episode_id"], row["segment_index"]): row["worker"] for row in plan["selected"]}
    rows, manifests = {}, []
    for worker in sorted(set(expected.values())):
        directory = root / f"measure_{worker}"
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "boundaries.jsonl"
        if (
            manifest["schema_version"] != 2
            or manifest["boundaries_sha256"] != data_sha256(path)
            or manifest["selection_plan_sha256"] != data_sha256(root / "selection.json")
        ):
            raise ValueError("Worker result schema/checksum/selection mismatch")
        manifests.append(manifest)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["episode_id"], row["segment_index"])
                if key in rows or expected.get(key) != worker or "skip" in row or row.get("schema_version") != 2:
                    raise ValueError("Duplicate, unexpected, skipped, or outdated worker result")
                rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError("Incomplete segment coverage; refusing to report a partial experiment")
    for manifest in manifests[1:]:
        if (
            manifest["actor_checkpoint"] != manifests[0]["actor_checkpoint"]
            or manifest["critic_cache"] != manifests[0]["critic_cache"]
            or any(
                manifest["arguments"][key] != manifests[0]["arguments"][key]
                for key in (
                    "controls",
                    "seed",
                    "window_size",
                    "dtype",
                    "advantage_normalization",
                    "checkpoint_layers",
                    "qwen35_fla",
                )
            )
        ):
            raise ValueError("Measurement workers used different models or loss configurations")
    ordered = [rows[key] for key in sorted(rows)]
    with (root / "boundaries.jsonl").open("w") as stream:
        for row in ordered:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = summarize(ordered, seed=seed, bootstrap=bootstrap)
    _json(root / "summary.json", summary)
    return summary, manifests


def execute(args):
    devices = worker_devices(args)
    root = Path(args.output).resolve()
    plan = prepare(args, root)
    common = ["--device", args.device, "--dtype", args.dtype, "--max-tokens", str(args.max_tokens)]
    if args.qwen35_fla:
        common.append("--qwen35-fla")
    prefix = [sys.executable, "-m", "agentic_critic_pipeline.compact_grad"]
    cache_jobs = [
        (
            worker,
            prefix
            + [
                "cache-values",
                "--base-model",
                args.base_model or args.actor,
                "--critic",
                args.critic,
                "--critic-mode",
                args.critic_mode,
                "--episodes",
                str(root / f"cache_input_{worker}.jsonl"),
                "--output",
                str(root / f"cache_{worker}"),
            ]
            + common,
        )
        for worker in sorted({row["worker"] for row in plan["cache_assignment"]})
    ]
    run_jobs(cache_jobs, devices, root, "cache")
    merge_caches(root, plan)
    measure_jobs = [
        (
            worker,
            prefix
            + [
                "measure",
                "--actor",
                args.actor,
                "--values",
                str(root / "cache"),
                "--episodes",
                str(root / "cohort.jsonl"),
                "--selection-plan",
                str(root / "selection.json"),
                "--shard-index",
                str(worker),
                "--output",
                str(root / f"measure_{worker}"),
                "--window-size",
                str(args.window_size),
                "--controls",
                str(args.controls),
                "--seed",
                str(args.seed),
                "--bootstrap",
                str(args.bootstrap),
                "--head-chunk-size",
                str(args.head_chunk_size),
                "--advantage-normalization",
                args.advantage_normalization,
                "--checkpoint-layers" if args.checkpoint_layers else "--no-checkpoint-layers",
            ]
            + common,
        )
        for worker in sorted({row["worker"] for row in plan["selected"]})
    ]
    run_jobs(measure_jobs, devices, root, "measure")
    summary, manifests = merge_results(root, plan, seed=args.seed, bootstrap=args.bootstrap)
    metadata = dict(
        _provenance(args),
        schema_version=2,
        selection=plan,
        worker_manifests=manifests,
        boundaries_sha256=data_sha256(root / "boundaries.jsonl"),
    )
    _json(root / "manifest.json", metadata)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("actor", "critic", "episodes", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--base-model", help="HF base for the exported critic; defaults to --actor")
    parser.add_argument("--critic-mode", choices=("actor_summary", "full_history"), default="actor_summary")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpus", help="Comma-separated logical indices within the inherited CUDA_VISIBLE_DEVICES")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--min-segment-tokens", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--controls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--head-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--advantage-normalization", choices=("cohort", "none"), default="cohort")
    parser.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qwen35-fla", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "workers",
        "segments",
        "min_segment_tokens",
        "window_size",
        "controls",
        "bootstrap",
        "max_tokens",
        "head_chunk_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.min_segment_tokens < 2 * args.window_size:
        parser.error("--min-segment-tokens must be at least twice --window-size")
    if args.segments < args.workers:
        parser.error("Select at least one segment per worker")
    if args.qwen35_fla and (args.device != "cuda" or args.dtype != "bfloat16"):
        parser.error("--qwen35-fla requires CUDA and bfloat16")
    execute(args)


if __name__ == "__main__":
    main()
