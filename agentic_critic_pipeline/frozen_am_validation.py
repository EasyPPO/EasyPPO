"""Compare frozen full-history checkpoints under full, summary and AM inputs.

No optimizer or actor generation runs. Reuse the exact validation plan, sampled
IDs, raw token IDs, masks and 100-token visualization schema from training.

Example:
  python -m agentic_critic_pipeline.frozen_am_validation \
    --base-model /models/qwen --plan /run/validation_replay/validation_plan.json \
    --training-output /run/full_history/train --out /run/frozen_am --workers 8

Alternatively supply converted checkpoints with repeated --checkpoint BATCH=DIR.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from agentic_critic.critic_input import Segment
from agentic_critic.critic_train import read_episode
from agentic_critic.fcs_env import Episode, Event
from agentic_critic.kv_compact import validate_summary_segments
from agentic_critic.memory_config import resolve_memory_config
from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.records import data_sha256, validate_episode
from agentic_critic_pipeline.validation import load_plan
from agentic_critic_pipeline.validation_metrics import ValidationMetrics
from agentic_critic_pipeline.validation_storage import compact_episode, validate_compact_episode
from agentic_critic_runtime.critic_checkpoint import load_critic
from agentic_critic_runtime.critic_features import episode_features

MODES = ("full_history", "actor_summary", "attention_matching")


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary_segments(episode):
    """Recover spans from original actor events, never from decoded summary text."""
    validate_episode(episode)
    source = episode["segments"]
    if episode["n_compactions"] != len(source) - 1:
        raise ValueError("Incomplete compaction history")
    if "events" in episode:
        rebuilt = Episode(
            events=[Event(**{k: e[k] for k in ("tokens", "actor", "segment", "kind")}) for e in episode["events"]],
            n_compactions=episode["n_compactions"],
        ).to_segments()
        original = [dict(input_ids=s.tokens, response_start=s.response_start, loss_mask=s.loss_mask) for s in rebuilt]
        if original != [{k: s[k] for k in ("input_ids", "response_start", "loss_mask")} for s in source]:
            raise ValueError("Original event tokens/masks differ from stored segments")
    else:
        rebuilt = [Segment(s["input_ids"], len(s["loss_mask"]), s["loss_mask"], s.get("summary_span")) for s in source]
    validate_summary_segments(rebuilt, resolve_memory_config("attention_matching", None))
    return rebuilt


def _episodes(plan, source, max_tokens):
    seen = set()
    with Path(source).open() as stream:
        for line in stream:
            episode = json.loads(line)
            identity = episode["episode_id"]
            if identity in seen or identity not in plan["episodes"] or episode["split"] != "validation":
                raise ValueError("Duplicate, unexpected or non-validation episode")
            seen.add(identity)
            expected = plan["episodes"][identity]
            if any(episode[k] != expected[k] for k in ("problem_id", "reward", "n_compactions")):
                raise ValueError("Validation episode differs from frozen plan")
            segments = summary_segments(episode)
            if sum(len(s.tokens) for s in segments) > max_tokens:
                raise ValueError("Validation exceeds max-tokens; truncation is forbidden")
            identities = [
                dict(tokens=sum(s.loss_mask), positions_sha256=digest([i for i, bit in enumerate(s.loss_mask) if bit]))
                for s in segments
            ]
            if identities != expected["segments"]:
                raise ValueError("Validation actor-token positions differ from frozen plan")
            yield episode, segments
    if seen != plan["episodes"].keys():
        raise ValueError("Validation source is missing frozen episodes")


@torch.no_grad()
def predict(model, head, segments, mode, device, *, ratio=0.1, chunk=4096):
    """One frozen evaluation; AM captures summary Q in its existing segment pass."""
    if mode != "attention_matching":
        features = episode_features(
            model, segments, {"full_history": "full", "actor_summary": "shared"}[mode], device, chunk=chunk
        )
        return [
            torch.cat([head(h[start : start + chunk].to(device)).float().cpu() for start in range(0, len(h), chunk)])
            for h in features
        ]
    values = []

    def collect(pairs):
        for index, value in pairs:
            mask = torch.tensor(segments[index].loss_mask, dtype=torch.bool, device=value.device)
            values.append(value[mask].detach().float().cpu())

    read_episode(model, head, segments, mode, device, collect, train=False, kv={"ratio": ratio})
    return values


def _synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _worker(args):
    run = json.loads((args.out / "frozen_run.json").read_text())
    checkpoint = next(row for row in run["checkpoints"] if row["batch"] == args.worker_batch)
    plan = load_plan(args.plan)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
    model, head = load_critic(
        str(args.base_model), checkpoint["path"], device=device, dtype=getattr(torch, args.dtype)
    )
    model.requires_grad_(False)
    timing = {mode: dict(seconds=0.0, episodes=0, peak_allocated_bytes=0) for mode in MODES}
    streams = {
        mode: gzip.open(args.out / mode / f"validation/batch_{args.worker_batch:04d}/values.rank{rank}.jsonl.gz", "wt")
        for mode in MODES
    }
    try:
        for index, (episode, segments) in enumerate(_episodes(plan, args.episodes or plan["source"], args.max_tokens)):
            if index % world != rank:
                continue
            for mode in MODES:
                _synchronize(device)
                if device != "cpu":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                values = predict(model, head, segments, mode, device, ratio=args.kv_ratio, chunk=args.chunk)
                _synchronize(device)
                timing[mode]["seconds"] += time.perf_counter() - start
                timing[mode]["episodes"] += 1
                if device != "cpu":
                    timing[mode]["peak_allocated_bytes"] = max(
                        timing[mode]["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device)
                    )
                row = compact_episode(
                    dict(
                        episode_id=episode["episode_id"],
                        problem_id=episode["problem_id"],
                        reward=episode["reward"],
                        n_compactions=episode["n_compactions"],
                        mode=mode,
                        batch=args.worker_batch,
                        trained_mode="full_history",
                        checkpoint_sha256=checkpoint["sha256"],
                        segments=[
                            dict(
                                segment_index=i,
                                response_positions=[j for j, bit in enumerate(s.loss_mask) if bit],
                                values=v.tolist(),
                            )
                            for i, (s, v) in enumerate(zip(segments, values, strict=True))
                        ],
                    )
                )
                validate_compact_episode(row, plan["episodes"][episode["episode_id"]])
                streams[mode].write(json.dumps(row, allow_nan=False) + "\n")
                streams[mode].flush()
    finally:
        for stream in streams.values():
            stream.close()
    _json(args.out / f"timing.batch{args.worker_batch:04d}.rank{rank}.json", timing)


def finalize(plan, out, checkpoint, workers):
    """Reject incomplete or mismatched shards before publishing aggregate metrics."""
    batch = checkpoint["batch"]
    for mode in MODES:
        directory = out / mode / f"validation/batch_{batch:04d}"
        stats = ValidationMetrics(plan, mode=mode, batch=batch, collect_profiles=False)
        seen = set()
        with gzip.open(directory / "values.jsonl.gz", "wt") as output, gzip.open(
            directory / "selected_trajectories.jsonl.gz", "wt"
        ) as traces:
            for rank in range(workers):
                with gzip.open(directory / f"values.rank{rank}.jsonl.gz", "rt") as stream:
                    for row in map(json.loads, stream):
                        identity = row["episode_id"]
                        if identity in seen or identity not in plan["episodes"]:
                            raise ValueError("Duplicate or unexpected validation prediction")
                        if (
                            row["mode"] != mode
                            or row["batch"] != batch
                            or row["checkpoint_sha256"] != checkpoint["sha256"]
                            or row["trained_mode"] != "full_history"
                        ):
                            raise ValueError("Validation predictions use different checkpoints or modes")
                        if row["problem_id"] != plan["episodes"][identity]["problem_id"]:
                            raise ValueError("Validation problem mismatch")
                        validate_compact_episode(row, plan["episodes"][identity])
                        seen.add(identity)
                        stats.add(row, traces=traces)
                        output.write(json.dumps(row, allow_nan=False) + "\n")
        if seen != plan["episodes"].keys():
            raise ValueError("Validation predictions do not cover the frozen cohort")
        timing = [
            json.loads((out / f"timing.batch{batch:04d}.rank{rank}.json").read_text())[mode] for rank in range(workers)
        ]
        _json(
            directory / "metrics.json",
            dict(
                **stats.summary(),
                plan_hash=plan["plan_hash"],
                trained_mode="full_history",
                checkpoint_sha256=checkpoint["sha256"],
                timing=dict(
                    sum_rank_seconds=sum(t["seconds"] for t in timing),
                    max_rank_seconds=max(t["seconds"] for t in timing),
                    peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in timing),
                ),
            ),
        )


def _checkpoints(args, plan):
    if args.training_output:
        run = json.loads((args.training_output / "miles_run.json").read_text())
        if run["mode"] != "full_history" or run["validation_plan_hash"] != plan["plan_hash"]:
            raise ValueError("Training run must be full_history with this frozen validation plan")
        root = Path(run["checkpoints"])
        latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
        # Miles saves the zero-based rollout_id; plots use completed logical batches.
        rows = [
            dict(batch=int(p.name.removeprefix("iter_")) + 1, native=str(p.resolve()))
            for p in sorted(root.glob("iter_*"))
            if p.is_dir() and p.name.removeprefix("iter_").isdigit() and int(p.name.removeprefix("iter_")) <= latest
        ]
        if not rows or max(r["batch"] for r in rows) > run["logical_batches"]:
            raise ValueError("No valid completed full-history checkpoints")
        for row in rows:
            row["path"] = str((args.out / "converted" / f"batch_{row['batch']:04d}").resolve())
    else:
        rows = []
        for spec in args.checkpoint:
            batch, path = spec.split("=", 1)
            rows.append(dict(batch=int(batch), path=str(Path(path).resolve())))
    if not rows or any(r["batch"] < 1 for r in rows) or len({r["batch"] for r in rows}) != len(rows):
        raise ValueError("Provide distinct positive checkpoint batches")
    return sorted(rows, key=lambda row: row["batch"])


def _prepare_checkpoint(row, args):
    if "native" in row:
        converter = Path(__file__).resolve().parents[1] / "scripts/agentic_critic/critic_to_hf.py"
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--input-dir",
                row["native"],
                "--base-hf-dir",
                str(args.base_model),
                "--out",
                row["path"],
            ],
            check=True,
        )
    root = Path(row["path"])
    row["sha256"] = {name: data_sha256(root / name) for name in ("critic.safetensors", "value_head.pt", "meta.json")}


def run(args):
    plan = load_plan(args.plan)
    source = args.episodes or Path(plan["source"])
    if data_sha256(source) != plan["source_sha256"]:
        raise ValueError("Validation source changed from frozen plan")
    # Check all spans/tokens before loading checkpoints; no text fallback is allowed.
    for _ in _episodes(plan, source, args.max_tokens):
        pass
    checkpoints = _checkpoints(args, plan)
    args.out.mkdir(parents=True, exist_ok=False)
    _json(args.out / "validation_plan.json", plan)
    manifest = dict(
        schema_version=1,
        experiment="frozen_full_history_am",
        trained_mode="full_history",
        modes=list(MODES),
        plan_hash=plan["plan_hash"],
        checkpoints=checkpoints,
        memory_config=resolve_memory_config("attention_matching", {"ratio": args.kv_ratio}),
        dtype=args.dtype,
        status="running",
        base_model=str(args.base_model.resolve()),
        base_config_sha256=data_sha256(args.base_model / "config.json"),
    )
    for checkpoint in checkpoints:
        _prepare_checkpoint(checkpoint, args)
        _json(args.out / "frozen_run.json", manifest)
        for mode in MODES:
            (args.out / mode / f"validation/batch_{checkpoint['batch']:04d}").mkdir(parents=True)
        command = [sys.executable]
        if args.workers > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.workers}"]
        command += [
            "-m",
            "agentic_critic_pipeline.frozen_am_validation",
            "--worker-batch",
            str(checkpoint["batch"]),
            "--base-model",
            str(args.base_model),
            "--plan",
            str(args.plan),
            "--episodes",
            str(source),
            "--out",
            str(args.out),
            "--dtype",
            args.dtype,
            "--kv-ratio",
            str(args.kv_ratio),
            "--chunk",
            str(args.chunk),
            "--max-tokens",
            str(args.max_tokens),
        ]
        subprocess.run(command, check=True)
        finalize(plan, args.out, checkpoint, args.workers)
    manifest["status"] = "complete"
    _json(args.out / "frozen_run.json", manifest)
    from agentic_critic_pipeline.frozen_am_figures import render

    render(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "plan", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--training-output", type=Path)
    source.add_argument(
        "--checkpoint",
        action="append",
        help="Completed batch=converted critic directory; weights must be full-history-trained",
    )
    parser.add_argument("--episodes", type=Path, help="Relocated validation JSONL; must match frozen source hash")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--kv-ratio", type=float, default=0.1)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.workers, args.chunk, args.max_tokens) < 1:
        parser.error("workers, chunk and max-tokens must be positive")
    resolve_memory_config("attention_matching", {"ratio": args.kv_ratio})
    if args.worker_batch:
        _worker(args)
    elif not (args.training_output or args.checkpoint):
        parser.error("provide --training-output or --checkpoint")
    else:
        run(args)


if __name__ == "__main__":
    main()
