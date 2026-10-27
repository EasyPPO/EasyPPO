"""Frozen-model compact-boundary gradient experiment.

Run `python -m agentic_critic_pipeline.compact_grad --help`.
Cache token-aligned critic values once, then reuse them for actor measurements.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from agentic_critic_pipeline.gradient_cohort import load_selection
from agentic_critic_pipeline.gradient_models import (
    actor_loss_function,
    checkpoint_layers,
    critic_values,
    token_positions,
)
from agentic_critic_pipeline.gradient_probe import measure_gradients, probe_mode, select_windows, summarize
from agentic_critic_pipeline.records import data_sha256, validate_episode
from agentic_critic_runtime.critic_checkpoint import load_critic


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _episodes(path):
    seen = set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            episode = json.loads(line)
            validate_episode(episode)
            identity = episode["episode_id"]
            if identity in seen:
                raise ValueError(f"Duplicate episode: {identity}")
            seen.add(identity)
            if episode.get("n_compactions", len(episode["segments"]) - 1) != len(episode["segments"]) - 1:
                raise ValueError("Compaction count disagrees with complete segment history")
            if episode.get("finish_reason", "max_turns") not in {"max_turns", "submissions_exhausted"}:
                raise ValueError("Use naturally completed episodes: R-V assumes terminal Monte Carlo returns")
            yield episode
    if not seen:
        raise ValueError("Empty episode file")


def _identity(episode):
    # Hash exact tokens, masks, reward and identities, not unrelated event text.
    return _digest({key: episode[key] for key in ("episode_id", "problem_id", "reward", "segments")})


def _provenance(args):
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True
    )
    return dict(
        schema_version=1,
        arguments=vars(args),
        source_sha256=data_sha256(args.episodes),
        git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
    )


def _checkpoint_identity(path):
    """Hash local checkpoint artifacts once, so paths alone cannot hide changes."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError("Use a local, immutable HF checkpoint directory")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".pt"})
    if not files:
        raise ValueError(f"No checkpoint artifacts in {root}")
    return {str(p.relative_to(root)): data_sha256(p) for p in files}


def cache_values(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = _provenance(args)
    manifest["base_checkpoint"] = _checkpoint_identity(args.base_model)
    manifest["critic_checkpoint"] = _checkpoint_identity(args.critic)
    model, head = load_critic(args.base_model, args.critic, device=args.device, dtype=getattr(torch, args.dtype))
    model.eval().requires_grad_(False)
    count, mean, m2, episodes = 0, 0.0, 0.0, 0
    with (output / "values.jsonl").open("w") as stream:
        for episode in _episodes(args.episodes):
            values = critic_values(model, head, episode, mode=args.critic_mode, max_tokens=args.max_tokens)
            flat = torch.tensor([episode["reward"] - v for segment in values for v in segment], dtype=torch.float64)
            n, current_mean = flat.numel(), float(flat.mean())
            delta = current_mean - mean
            m2 += float(((flat - current_mean) ** 2).sum()) + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
            row = dict(episode_id=episode["episode_id"], identity=_identity(episode), values=values)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            episodes += 1
            print(f"cached episode {episodes}: {episode['episode_id']}", flush=True)
    manifest.update(
        episodes=episodes,
        advantage_moments=dict(count=count, mean=mean, variance=m2 / max(1, count - 1)),
        values_sha256=data_sha256(output / "values.jsonl"),
    )
    # Written last: interrupted caches have no completion manifest.
    _json(output / "manifest.json", manifest)


def _cached_pairs(source, cache):
    with (cache / "values.jsonl").open() as stream:
        for episode in _episodes(source):
            line = stream.readline()
            if not line:
                raise ValueError("Critic cache has missing episodes")
            row = json.loads(line)
            if row["episode_id"] != episode["episode_id"] or row["identity"] != _identity(episode):
                raise ValueError("Critic cache token/mask/reward identity mismatch")
            values = row["values"]
            if len(values) != len(episode["segments"]) or any(
                len(v) != sum(s["loss_mask"]) or not all(math.isfinite(x) for x in v)
                for s, v in zip(episode["segments"], values, strict=True)
            ):
                raise ValueError("Critic cache values are nonfinite or misaligned")
            yield episode, values
        if stream.read().strip():
            raise ValueError("Critic cache has extra episodes")


def _measure_segment(model, segment, values, reward, windows, args, moments):
    if len(segment["input_ids"]) > args.max_tokens:
        raise ValueError("Actor segment exceeds --max-tokens; truncation is forbidden")
    advantages = torch.tensor([reward - value for value in values], dtype=torch.float32)
    if args.advantage_normalization == "cohort":
        advantages = (advantages - moments["mean"]) / math.sqrt(moments["variance"] + 1e-8)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with probe_mode(model), checkpoint_layers(model, args.checkpoint_layers):
        losses = actor_loss_function(model, segment, advantages, head_chunk_size=args.head_chunk_size)
        result = measure_gradients(losses, model.parameters(), windows)
    positions = token_positions(segment)
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        loss_tokens=len(positions),
        context_tokens=len(segment["input_ids"]),
        # Persist exact absolute TARGET positions (not t-1 predictor positions).
        target_positions=[positions[i] + 1 for i in windows[0]],
        control_positions=[[positions[i] + 1 for i in window] for window in windows[1:]],
        advantage_rms=[float(advantages[window].square().mean().sqrt()) for window in windows],
    )
    return result


def measure(args):
    cache = Path(args.values)
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest["source_sha256"] != data_sha256(args.episodes):
        raise ValueError("Critic cache uses a different episode file/schema")
    if manifest["values_sha256"] != data_sha256(cache / "values.jsonl"):
        raise ValueError("Critic cache checksum mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = _provenance(args)
    metadata["schema_version"] = 2
    metadata.update(critic_cache=manifest, actor_checkpoint=_checkpoint_identity(args.actor))
    selected = None
    if args.selection_plan:
        selected = load_selection(
            args.selection_plan,
            episodes=args.episodes,
            shard=args.shard_index,
            window_size=args.window_size,
            seed=args.seed,
        )
        metadata["selection_plan_sha256"] = data_sha256(args.selection_plan)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.actor, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa", local_files_only=True
        )
        .to(args.device)
        .eval()
    )
    rows = []
    with (output / "boundaries.jsonl").open("w") as stream:
        for episode, values in _cached_pairs(args.episodes, cache):
            for index, segment in enumerate(episode["segments"]):
                if index == 0 and len(episode["segments"]) > 1:
                    continue
                key = (episode["episode_id"], index)
                if selected is not None and key not in selected:
                    continue
                row = dict(
                    schema_version=2,
                    episode_id=episode["episode_id"],
                    problem_id=episode["problem_id"],
                    segment_index=index,
                )
                windows, skip = select_windows(
                    sum(segment["loss_mask"]),
                    size=args.window_size,
                    controls=args.controls,
                    seed=f"{args.seed}:{episode['episode_id']}:{index}",
                )
                if index == 0 or skip:
                    row["skip"] = "no_compaction" if index == 0 else skip
                else:
                    row.update(
                        _measure_segment(
                            model,
                            segment,
                            values[index],
                            episode["reward"],
                            windows,
                            args,
                            manifest["advantage_moments"],
                        )
                    )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(
                    json.dumps(
                        {
                            key: row[key]
                            for key in row
                            if key in {"episode_id", "segment_index", "skip", "ratio_to_control_mean"}
                        }
                    ),
                    flush=True,
                )
    result = summarize(rows, seed=args.seed, bootstrap=args.bootstrap)
    if selected is not None and (
        {(row["episode_id"], row["segment_index"]) for row in rows} != selected or any("skip" in row for row in rows)
    ):
        raise ValueError("Selected segments are missing or ineligible; partial results are not complete")
    _json(output / "summary.json", result)
    metadata["boundaries_sha256"] = data_sha256(output / "boundaries.jsonl")
    _json(output / "manifest.json", metadata)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache-values", help="One no-grad critic pass, reusable for all probes")
    cache.add_argument("--base-model", required=True, help="Local HF base model directory")
    cache.add_argument("--critic", required=True, help="Export containing critic.safetensors and value_head.pt")
    cache.add_argument("--critic-mode", choices=("actor_summary", "full_history"), default="actor_summary")
    probe = commands.add_parser("measure", help="Exact parameter-gradient norms, without optimizer updates")
    probe.add_argument("--actor", required=True, help="Local HF actor checkpoint directory")
    probe.add_argument("--values", required=True, help="Completed cache-values output directory")
    probe.add_argument("--window-size", type=int, default=64)
    probe.add_argument("--controls", type=int, default=8)
    probe.add_argument("--seed", type=int, default=1234)
    probe.add_argument("--bootstrap", type=int, default=2000)
    probe.add_argument("--head-chunk-size", type=int, default=128)
    probe.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True)
    probe.add_argument("--advantage-normalization", choices=("cohort", "none"), default="cohort")
    probe.add_argument("--selection-plan", help="Frozen segment plan produced by compact_grad_parallel")
    probe.add_argument("--shard-index", type=int, default=0)
    for command in (cache, probe):
        command.add_argument(
            "--episodes", required=True, help="Original collected JSONL with segments; not 100-token bins"
        )
        command.add_argument("--output", required=True, help="Fresh output directory")
        command.add_argument("--device", default="cuda")
        command.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        command.add_argument("--max-tokens", type=int, default=120000)
        command.add_argument(
            "--qwen35-fla", action="store_true", help="Explicitly enable the repository's Qwen3.5 FLA kernels"
        )
    args = parser.parse_args(argv)
    if getattr(args, "shard_index", 0) < 0 or (getattr(args, "shard_index", 0) and not args.selection_plan):
        parser.error("A nonzero --shard-index requires --selection-plan; indices must be nonnegative")
    for name in ("max_tokens", "window_size", "controls", "bootstrap", "head_chunk_size"):
        if getattr(args, name, 1) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.qwen35_fla:
        if args.dtype != "bfloat16":
            parser.error("--qwen35-fla requires --dtype bfloat16 for the flash-only SDPA kernel")
        # Optional GPU-only dependency, deliberately deferred for CPU smoke tests.
        from agentic_critic_pipeline.kernels import configure_training_kernels

        configure_training_kernels()
    (cache_values if args.command == "cache-values" else measure)(args)


if __name__ == "__main__":
    main()
