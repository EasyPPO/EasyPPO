"""Train full/shared/AM critics on existing offline trajectories, then visualize.

This HF replicated-data-parallel experiment is separate from Miles training and
from frozen_am_validation. No actor, judge, collection or tokenization is run.

python -m agentic_critic_pipeline.train_am_comparison \
  --model /models/base --rollouts /old-run/replay \
  --validation-plan /old-run/validation_replay/validation_plan.json \
  --output /new-run/trained-am --workers 8
"""

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from agentic_critic.critic_train import AdamWHalfMoments
from agentic_critic.kv_compact import validate_model
from agentic_critic.memory_config import resolve_memory_config
from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.critic_training import (
    INPUT_MODES,
    prediction_row,
    read_values,
    save_checkpoint,
    train_batch,
)
from agentic_critic_pipeline.frozen_am_validation import _episodes
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.training_replay import load_batch, prepare_replay
from agentic_critic_pipeline.validation import load_plan
from agentic_critic_pipeline.validation_metrics import ValidationMetrics
from agentic_critic_pipeline.validation_storage import validate_compact_episode


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _checkpoint(model, linear, directory, *, mode, batch, plan_hash, rank):
    identity = None
    if rank == 0:
        save_checkpoint(model, linear, directory, mode=mode, batch=batch, plan_hash=plan_hash)
        identity = {
            name: data_sha256(directory / name) for name in ("critic.safetensors", "value_head.pt", "meta.json")
        }
    if dist.is_initialized():
        payload = [identity]
        dist.broadcast_object_list(payload, src=0)
        identity = payload[0]
    return identity


def _finalize_validation(plan, directory, *, mode, batch, world, checkpoint):
    stats = ValidationMetrics(plan, mode=mode, batch=batch, collect_profiles=False)
    seen = set()
    with gzip.open(directory / "values.jsonl.gz", "wt") as output, gzip.open(
        directory / "selected_trajectories.jsonl.gz", "wt"
    ) as traces:
        for rank in range(world):
            with gzip.open(directory / f"values.rank{rank}.jsonl.gz", "rt") as source:
                for row in map(json.loads, source):
                    identity = row["episode_id"]
                    if identity in seen or identity not in plan["episodes"]:
                        raise ValueError("Duplicate or unexpected trained-critic validation episode")
                    if (
                        row["mode"] != mode
                        or row["trained_mode"] != mode
                        or row["batch"] != batch
                        or row["checkpoint_sha256"] != checkpoint
                    ):
                        raise ValueError("Trained-critic validation checkpoint/mode mismatch")
                    if row["problem_id"] != plan["episodes"][identity]["problem_id"]:
                        raise ValueError("Validation problem mismatch")
                    validate_compact_episode(row, plan["episodes"][identity])
                    seen.add(identity)
                    stats.add(row, traces=traces)
                    output.write(json.dumps(row, allow_nan=False) + "\n")
    if seen != plan["episodes"].keys():
        raise ValueError("Trained-critic validation cohort is incomplete")
    timings = [json.loads((directory / f"timing.rank{rank}.json").read_text()) for rank in range(world)]
    _json(
        directory / "metrics.json",
        dict(
            **stats.summary(),
            plan_hash=plan["plan_hash"],
            trained_mode=mode,
            checkpoint_sha256=checkpoint,
            backend="hf_replicated",
            timing=dict(
                sum_rank_seconds=sum(t["seconds"] for t in timings),
                peak_allocated_bytes=max(t["peak_allocated_bytes"] for t in timings),
            ),
        ),
    )


def _validate(model, linear, plan, *, mode, batch, rank, world, device, args, output, checkpoint):
    directory = output / f"validation/batch_{batch:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    if device != "cpu":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with gzip.open(directory / f"values.rank{rank}.jsonl.gz", "wt") as stream:
        for index, (episode, segments) in enumerate(
            _episodes(plan, args.validation_episodes or plan["source"], args.max_tokens)
        ):
            if index % world != rank:
                continue
            ep = dict(
                episode_id=episode["episode_id"],
                problem=episode["problem_id"],
                reward=episode["reward"],
                segments=segments,
            )
            values = read_values(model, linear, ep, mode, device, args)
            row = prediction_row(ep, values, mode=mode, batch=batch)
            # Include checkpoint identity before computing the persisted checksum.
            row.pop("data_sha256")
            row["checkpoint_sha256"] = checkpoint
            row["data_sha256"] = digest(row)
            validate_compact_episode(row, plan["episodes"][ep["episode_id"]])
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if device != "cpu":
        torch.cuda.synchronize(device)
    _json(
        directory / f"timing.rank{rank}.json",
        dict(
            seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device != "cpu" else 0,
        ),
    )
    _barrier()
    if rank == 0:
        _finalize_validation(plan, directory, mode=mode, batch=batch, world=world, checkpoint=checkpoint)
    _barrier()
    model.train()


def _worker(config_path, mode):
    from transformers import AutoModelForCausalLM

    config = json.loads(config_path.read_text())
    args = argparse.Namespace(**config["arguments"])
    replay = config["replay"]
    plan = load_plan(Path(args.output) / "validation_plan.json")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if device != "cpu" else "gloo", timeout=timedelta(hours=2))
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(
        device
    )
    validate_model(model)
    if getattr(model.config, "attention_dropout", 0) or any(
        isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()
    ):
        raise ValueError("Use dropout-free critics for deterministic detached cache reconstruction")
    linear = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    torch.nn.init.normal_(linear.weight, std=args.head_init_std)
    torch.nn.init.zeros_(linear.bias)
    parameters = [p for n, p in model.named_parameters() if not n.startswith("lm_head")] + list(linear.parameters())
    optimizer = AdamWHalfMoments(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        moment_dtype=torch.bfloat16 if args.moment_dtype == "bfloat16" else torch.float32,
    )
    output = Path(args.output) / mode
    output.mkdir(exist_ok=True)
    rows = []
    for step, source in enumerate(replay["files"]):
        if data_sha256(source["path"]) != source["sha256"]:
            raise ValueError("Existing replay changed after the experiment was prepared")
        episodes = load_batch(source["path"], replay["source"], replay["offsets"])
        if [e["episode_id"] for e in episodes] != source["episode_ids"]:
            raise ValueError("Replay trajectory order changed")
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)
        row = train_batch(
            model,
            linear,
            optimizer,
            episodes,
            mode=mode,
            batch=step + 1,
            device=device,
            rank=rank,
            world=world,
            args=args,
            output=output,
        )
        rows.append(row)
        if rank == 0:
            _json(output / "training_log.json", dict(mode=mode, backend="hf_replicated", steps=rows))
            print(json.dumps(dict(mode=mode, **row)), flush=True)
        batch = step + 1
        evaluate = batch % args.validation_interval == 0 or batch == replay["steps"]
        if evaluate or batch % args.save_interval == 0:
            checkpoint = _checkpoint(
                model,
                linear,
                output / f"checkpoints/batch_{batch:04d}",
                mode=mode,
                batch=batch,
                plan_hash=plan["plan_hash"],
                rank=rank,
            )
        if evaluate:
            _validate(
                model,
                linear,
                plan,
                mode=mode,
                batch=batch,
                rank=rank,
                world=world,
                device=device,
                args=args,
                output=output,
                checkpoint=checkpoint,
            )
    if rank == 0:
        _json(
            output / "complete.json", dict(mode=mode, steps=len(rows), optimizer_updates=len(rows) * args.mini_batches)
        )
    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def _validate_args(args):
    if any(getattr(args, k) is None for k in ("model", "rollouts", "validation_plan", "output")):
        raise ValueError("model, existing rollouts, validation-plan and fresh output are required")
    if min(args.workers, args.mini_batches, args.validation_interval, args.save_interval, args.max_tokens) < 1:
        raise ValueError("Worker, mini-batch, interval and token counts must be positive")
    for name in ("lr", "clip_grad", "value_clip", "value_loss_coef", "return_std_floor", "head_init_std"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or len(args.modes) != len(set(args.modes)):
        raise ValueError("Invalid weight decay or duplicate modes")
    resolve_memory_config("attention_matching", {"ratio": args.kv_ratio})


def run(args):
    _validate_args(args)
    plan = load_plan(args.validation_plan)
    source = args.validation_episodes or Path(plan["source"])
    if data_sha256(source) != plan["source_sha256"]:
        raise ValueError("Validation JSONL differs from the existing frozen plan")
    for _ in _episodes(plan, source, args.max_tokens):
        pass
    replay = prepare_replay(
        args.rollouts, plan, steps=args.steps, mini_batches=args.mini_batches, max_tokens=args.max_tokens
    )
    args.output.mkdir(parents=True, exist_ok=False)
    _json(args.output / "validation_plan.json", plan)
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    arguments["validation_episodes"] = str(source.resolve())
    config = dict(arguments=arguments, replay=replay)
    _json(args.output / "run_config.json", config)
    validation_batches = [
        b for b in range(1, replay["steps"] + 1) if b % args.validation_interval == 0 or b == replay["steps"]
    ]
    manifest = dict(
        schema_version=1,
        experiment="trained_am_comparison",
        backend="hf_replicated",
        modes=args.modes,
        plan_hash=plan["plan_hash"],
        training_source_sha256=plan["training_source_sha256"],
        replay_source_sha256=replay["source_sha256"],
        validation_batches=validation_batches,
        steps=replay["steps"],
        mini_batches=args.mini_batches,
        memory_config=resolve_memory_config("attention_matching", {"ratio": args.kv_ratio}),
        arguments=arguments,
        status="running",
    )
    _json(args.output / "training_run.json", manifest)
    for mode in args.modes:
        command = [sys.executable]
        if args.workers > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.workers}"]
        command += [
            "-m",
            "agentic_critic_pipeline.train_am_comparison",
            "--worker-config",
            str(args.output / "run_config.json"),
            "--worker-mode",
            mode,
        ]
        subprocess.run(command, check=True)
    manifest["status"] = "complete"
    _json(args.output / "training_run.json", manifest)
    from agentic_critic_pipeline.trained_am_figures import render

    render(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "rollouts", "validation-plan", "validation-episodes", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(INPUT_MODES),
        default=["attention_matching"],
        help="Default: train AM only; completed full/shared baselines are not rerun",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, help="Default: all existing replay batches")
    parser.add_argument("--mini-batches", type=int, default=4)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kv-ratio", type=float, default=0.1)
    parser.add_argument("--autocast", choices=("bfloat16", "none"), default="bfloat16")
    parser.add_argument("--moment-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    for name, default in (
        ("lr", 2e-6),
        ("weight-decay", 0.01),
        ("clip-grad", 1.0),
        ("value-clip", 0.2),
        ("value-loss-coef", 0.5),
        ("return-std-floor", 0.075),
        ("head-init-std", 0.02),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=tuple(INPUT_MODES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        if args.worker_mode is None:
            parser.error("worker-mode is required with worker-config")
        _worker(args.worker_config, args.worker_mode)
    else:
        run(args)


if __name__ == "__main__":
    main()
