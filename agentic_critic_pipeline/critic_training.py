"""Shared HF critic optimization for full, actor-summary, and AM comparisons."""

import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from agentic_critic.critic_train import clipped_value_loss, problem_weights, read_episode
from agentic_critic_pipeline.validation_metrics import _metrics, _moments
from agentic_critic_pipeline.validation_storage import compact_episode

INPUT_MODES = {"full_history": "full", "actor_summary": "shared", "attention_matching": "attention_matching"}


def autocast(args, device):
    return torch.autocast(
        device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=args.autocast == "bfloat16"
    )


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def sum_tensor(value):
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def synchronize_gradients(parameters):
    """SUM already globally token-normalized gradients, including empty ranks.

    read_episode invokes the backbone and calls backward per segment, so a DDP
    wrapper around the LM would not see its forward. Synchronize once per complete
    optimizer mini-batch instead. All ranks execute the same collectives.
    """
    if not dist.is_initialized():
        return
    present = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int)
    sum_tensor(present)
    for parameter, used in zip(parameters, present.tolist(), strict=True):
        if used:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            sum_tensor(parameter.grad)


def prediction_row(episode, values, *, mode, batch):
    return compact_episode(
        dict(
            episode_id=episode["episode_id"],
            problem_id=episode["problem"],
            reward=episode["reward"],
            n_compactions=len(episode["segments"]) - 1,
            mode=mode,
            trained_mode=mode,
            batch=batch,
            segments=[
                dict(
                    segment_index=i,
                    response_positions=[j for j, bit in enumerate(s.loss_mask) if bit],
                    values=v[torch.tensor(s.loss_mask).bool()].tolist(),
                )
                for i, (s, v) in enumerate(zip(episode["segments"], values, strict=True))
            ],
        )
    )


def read_values(model, linear, episode, mode, device, args):
    values = []
    with torch.no_grad(), autocast(args, device):
        read_episode(
            model,
            lambda h: linear(h.float()).squeeze(-1),
            episode["segments"],
            INPUT_MODES[mode],
            device,
            lambda pairs: values.extend(v.detach().float().cpu() for _, v in pairs),
            train=False,
            kv={"ratio": args.kv_ratio} if mode == "attention_matching" else None,
        )
    return values


def _backward_episode(model, linear, episode, old, *, mode, device, args, weight, denominator):
    stats = [0.0, 0.0]

    def backward(pairs):
        loss = 0
        for index, value in pairs:
            mask = torch.tensor(episode["segments"][index].loss_mask, dtype=torch.bool, device=device)
            reference = old[index].to(device)
            raw, clipped = clipped_value_loss(value[mask], reference[mask], episode["reward"], args.value_clip)
            loss = loss + args.value_loss_coef * weight * raw.sum() / denominator
            stats[1] += float(clipped.sum())
        stats[0] += float(loss.detach())
        loss.backward()

    with autocast(args, device):
        read_episode(
            model,
            lambda h: linear(h.float()).squeeze(-1),
            episode["segments"],
            INPUT_MODES[mode],
            device,
            backward,
            train=True,
            kv={"ratio": args.kv_ratio} if mode == "attention_matching" else None,
        )
    return stats


def train_batch(model, linear, optimizer, episodes, *, mode, batch, device, rank, world, args, output):
    """One existing replay batch: pre-update values, then fixed mini-batch updates."""
    model.train()
    parameters = list(optimizer.param_groups[0]["params"])
    old, moments = {}, np.zeros(6, dtype=np.float64)
    local = [(i, e) for i, e in enumerate(episodes) if i % world == rank]
    _sync(device)
    start = time.perf_counter()
    for index, episode in local:
        values = read_values(model, linear, episode, mode, device, args)
        old[index] = values
        for segment, value in zip(episode["segments"], values, strict=True):
            moments += _moments(
                value[torch.tensor(segment.loss_mask).bool()].numpy().astype(np.float64), episode["reward"]
            )
    _sync(device)
    value_seconds = time.perf_counter() - start
    pre_update = _metrics(sum_tensor(torch.tensor(moments, device=device)).cpu().numpy())
    directory = output / "pre_update" / f"batch_{batch:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / f"values.rank{rank}.jsonl.gz", "wt") as stream:
        for index, episode in local:
            stream.write(
                json.dumps(prediction_row(episode, old[index], mode=mode, batch=batch), allow_nan=False) + "\n"
            )
    rewards = {}
    for episode in episodes:
        rewards.setdefault(episode["problem"], []).append(episode["reward"])
    weights = problem_weights(rewards, args.return_std_floor)
    size = len(episodes) // args.mini_batches
    updates = []
    _sync(device)
    start = time.perf_counter()
    for first in range(0, len(episodes), size):
        n_tokens = sum(sum(s.loss_mask) for ep in episodes[first : first + size] for s in ep["segments"])
        stats = [0.0, 0.0]
        optimizer.zero_grad(set_to_none=True)
        for index, episode in local:
            if first <= index < first + size:
                loss, clipped = _backward_episode(
                    model,
                    linear,
                    episode,
                    old[index],
                    mode=mode,
                    device=device,
                    args=args,
                    weight=weights[episode["problem"]],
                    denominator=n_tokens,
                )
                stats[0] += loss
                stats[1] += clipped
        synchronize_gradients(parameters)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad, error_if_nonfinite=True))
        optimizer.step()
        stats = sum_tensor(torch.tensor(stats, dtype=torch.float64, device=device)).tolist()
        updates.append(
            dict(value_loss=stats[0], value_clipfrac=stats[1] / n_tokens, grad_norm=norm, actor_tokens=n_tokens)
        )
    _sync(device)
    train_seconds = time.perf_counter() - start
    timing = torch.tensor(
        [
            value_seconds,
            train_seconds,
            torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    return dict(
        batch=batch,
        episodes=len(episodes),
        pre_update=pre_update,
        updates=updates,
        value_seconds=timing[0].item(),
        train_seconds=timing[1].item(),
        peak_allocated_bytes=int(timing[2]),
        optimizer_updates=batch * args.mini_batches,
    )


def save_checkpoint(model, linear, output: Path, *, mode, batch, plan_hash):
    """Weights-only FP32 export compatible with the frozen evaluation loader."""
    import safetensors.torch

    output.mkdir(parents=True, exist_ok=False)
    state = {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if name.startswith("model.")}
    safetensors.torch.save_file(state, str(output / "critic.safetensors"))
    torch.save(dict(weight=linear.weight.detach().cpu(), bias=linear.bias.detach().cpu()), output / "value_head.pt")
    (output / "meta.json").write_text(
        json.dumps(
            dict(
                converted_parameters=len(state),
                trained_mode=mode,
                batch=batch,
                validation_plan_hash=plan_hash,
                dtype="float32",
                optimizer_saved=False,
                backend="hf_replicated",
            ),
            indent=2,
        )
        + "\n"
    )
