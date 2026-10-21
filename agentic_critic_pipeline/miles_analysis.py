"""Read Miles predictions and evaluate checkpoints for critic visualizations."""

import argparse
import json
import os
from pathlib import Path

import torch


def episodes(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def record(ep, segment, values, *, mode, phase, batch, trained_mode=None, **extra):
    mask = ep["segments"][segment]["loss_mask"]
    positions = [i for i, bit in enumerate(mask) if bit]
    values = torch.as_tensor(values).float().flatten()
    if len(values) != len(positions) or not torch.isfinite(values).all():
        raise ValueError("Miles values do not align with the original actor-token mask")
    return dict(
        episode_id=ep["episode_id"],
        problem_id=ep["problem_id"],
        split=ep["split"],
        segment_index=segment,
        reward=ep["reward"],
        response_positions=positions,
        values=values.tolist(),
        mode=mode,
        input_mode=mode,
        trained_mode=trained_mode or mode,
        phase=phase,
        batch=batch,
        **extra,
    )


def training(args):
    pool = list(episodes(args.episodes))
    by_id = {ep["episode_id"]: ep for ep in pool}
    # The replay adapter assigns original IDs in metadata. Legacy metadata can
    # use problem/sample only if it identifies exactly one source episode.
    legacy = {}
    for ep in pool:
        legacy.setdefault((str(ep["problem_id"]), str(ep.get("sample", ep.get("sample_index", "")))), []).append(ep)

    def source_episode(sample):
        meta = sample["metadata"]
        ep = by_id.get(meta.get("episode_id", meta.get("sample")))
        if ep is None:
            candidates = legacy.get((str(meta["problem"]), str(meta.get("sample", ""))), [])
            if len(candidates) != 1:
                raise ValueError("Replay metadata needs an unambiguous source episode_id")
            ep = candidates[0]
        segment = int(meta["segment"])
        original = ep["segments"][segment]
        if original["loss_mask"] != sample["loss_mask"]:
            raise ValueError("Replay actor-token mask differs from original trajectory")
        if original["input_ids"] != sample["tokens"]:
            raise ValueError("Replay token IDs differ from original trajectory")
        if ep["split"] != "train" or str(meta["problem"]) != str(ep["problem_id"]):
            raise ValueError("Replay problem or split differs from original training trajectory")
        if float(sample["reward"]) != float(ep["reward"]):
            raise ValueError("Replay reward differs from original trajectory")
        return ep, segment

    manifest = json.loads((Path(args.rollouts) / "manifest.json").read_text())
    expected_steps = {int(row["step"]) for row in manifest["steps"]}
    planned_count = int(manifest["args"]["steps"])
    first_step = int(manifest["args"].get("first_step", 0))
    if expected_steps != set(range(first_step, first_step + planned_count)):
        raise ValueError("Replay manifest does not cover every configured training batch")
    expected_rows = set()
    for step in sorted(expected_steps):
        replay = torch.load(Path(args.rollouts) / f"rollout_{step}.pt", map_location="cpu", weights_only=False)
        for sample in replay["samples"]:
            ep, segment = source_episode(sample)
            identity = (step, ep["episode_id"], segment)
            if identity in expected_rows:
                raise ValueError("Duplicate replay episode/segment identity")
            expected_rows.add(identity)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, observed_steps = set(), set()
    with (out / "values.rank0.jsonl").open("w") as stream:
        files = sorted(Path(args.dump).glob("*.critic.pt"))
        if not files:
            raise ValueError("No Miles pre-update critic dumps found")
        for path in files:
            dump = torch.load(path, map_location="cpu", weights_only=False)
            step = int(dump["rollout_id"])
            if step not in expected_steps:
                raise ValueError("Miles dump contains a batch outside the frozen replay manifest")
            observed_steps.add(step)
            replay = torch.load(Path(args.rollouts) / f"rollout_{step}.pt", map_location="cpu", weights_only=False)
            samples = replay["samples"]
            full = "value_slices" in dump
            if full != (args.mode == "full_history"):
                raise ValueError("Miles dump context mode differs from requested analysis mode")
            index_key = "rollout_ids" if full else "sample_indices"
            lookup = {int(key): i for i, key in enumerate(dump[index_key])}
            if len(lookup) != len(dump["values"]) or len(lookup) != len(dump["loss_masks"]):
                raise ValueError("Duplicate or misaligned identities in Miles critic dump")
            sample_keys = {int(s["rollout_id"] if full else s["index"]) for s in samples}
            if not lookup.keys() <= sample_keys:
                raise ValueError("Miles dump includes a sample outside its replay batch")
            for sample in samples:
                key = int(sample["rollout_id"] if full else sample["index"])
                if key not in lookup:
                    continue  # each data-parallel rank owns a subset
                i = lookup[key]
                value = torch.as_tensor(dump["values"][i]).flatten()
                mask = torch.as_tensor(dump["loss_masks"][i]).bool().flatten()
                if full:
                    start, length = sample["critic_span"]
                    if start < 0 or start + length > len(value):
                        raise ValueError("Invalid full-history segment span")
                    value, mask = value[start : start + length], mask[start : start + length]
                expected = torch.as_tensor(sample["loss_mask"]).bool()
                if len(value) != sample["response_length"] or not torch.equal(mask, expected):
                    raise ValueError("Miles response length or mask differs from replay")
                ep, segment = source_episode(sample)
                identity = (step, ep["episode_id"], segment)
                if identity in seen:
                    raise ValueError(f"Duplicate critic values across dump ranks: {identity}")
                seen.add(identity)
                row = record(ep, segment, value[mask], mode=args.mode, phase="pre_update", batch=step)
                stream.write(json.dumps(row) + "\n")

    if observed_steps != expected_steps or seen != expected_rows:
        raise ValueError("Miles critic dumps are missing replay batches or segments")


def heldout(args):
    from agentic_critic_runtime.critic_checkpoint import load_critic
    from agentic_critic_runtime.critic_features import episode_features
    from agentic_critic_runtime.critic_input import Segment

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local)
    model, head = load_critic(args.base_model, args.critic, device=device, dtype=getattr(torch, args.dtype))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    original_mode = {"actor_summary": "shared", "full_history": "full"}[args.input_mode]
    with torch.no_grad(), (out / f"values.rank{rank}.jsonl").open("w") as stream:
        for index, ep in enumerate(episodes(args.episodes)):
            if index % world != rank:
                continue
            if ep["split"] not in ("test", "validation"):
                raise ValueError("Heldout evaluation requires test or validation episodes")
            if sum(len(s["input_ids"]) for s in ep["segments"]) > args.max_tokens:
                raise ValueError("Heldout episode exceeds configured maximum; truncation is forbidden")
            segments = [Segment(s["input_ids"], len(s["loss_mask"]), s["loss_mask"]) for s in ep["segments"]]
            features = episode_features(model, segments, original_mode, device, chunk=args.chunk)
            for segment, hidden in enumerate(features):
                # Original evaluator returns only actor-token features, already
                # aligned to the same causal t-1 positions as Miles.
                if not len(hidden):
                    continue
                value_parts, grad = [], torch.zeros(hidden.shape[-1], dtype=torch.float32)
                for start in range(0, len(hidden), args.chunk):
                    h = hidden[start : start + args.chunk].float()
                    pred = head(h.to(device)).float().cpu()
                    value_parts.append(pred)
                    grad += ((pred - ep["reward"]).unsqueeze(1) * h).sum(0)
                values = torch.cat(value_parts)
                grad /= len(hidden)
                if not torch.isfinite(grad).all():
                    raise ValueError("Nonfinite heldout value-head gradient")
                row = record(
                    ep,
                    segment,
                    values,
                    mode=args.input_mode,
                    phase="heldout",
                    batch=args.checkpoint_batch,
                    trained_mode=args.trained_mode,
                    checkpoint_run_id=args.checkpoint_run_id,
                    checkpoint_batch=args.checkpoint_batch,
                    head_gradient=grad.tolist(),
                    head_gradient_definition="mean((V-G)*h); value-head weights only",
                    evaluation_dtype=args.dtype,
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("training")
    for name in ("episodes", "rollouts", "dump", "out"):
        train.add_argument(f"--{name}", required=True)
    train.add_argument("--mode", choices=("actor_summary", "full_history"), required=True)
    evaluate = sub.add_parser("heldout")
    for name in ("episodes", "critic", "base-model", "out", "checkpoint-run-id"):
        evaluate.add_argument(f"--{name}", required=True)
    for name in ("input-mode", "trained-mode"):
        evaluate.add_argument(f"--{name}", choices=("actor_summary", "full_history"), required=True)
    evaluate.add_argument("--checkpoint-batch", type=int, default=40)
    evaluate.add_argument("--max-tokens", type=int, default=120000)
    evaluate.add_argument("--chunk", type=int, default=4096)
    evaluate.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    args = ap.parse_args()
    if args.command == "heldout" and (args.chunk <= 0 or args.max_tokens <= 0):
        ap.error("chunk and max-tokens must be positive")
    (training if args.command == "training" else heldout)(args)


if __name__ == "__main__":
    main()
