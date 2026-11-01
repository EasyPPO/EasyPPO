"""Whole-trajectory policy gradients with bounded GPU and host storage."""

import math

import torch

from agentic_critic_pipeline.gradient_models import actor_loss_function, checkpoint_layers
from agentic_critic_pipeline.gradient_probe import probe_mode


def gradient_products(gradient, reference=None):
    """FP64 reductions in bounded chunks; missing gradients represent zeros."""
    device = next((value.device for value in gradient if value is not None), torch.device("cpu"))
    norm2 = torch.zeros((), dtype=torch.float64, device=device)
    dot = torch.zeros_like(norm2)
    for index, value in enumerate(gradient):
        if value is None:
            continue
        flat = value.detach().reshape(-1)
        other = None if reference is None else reference[index]
        for start in range(0, flat.numel(), 1 << 20):
            chunk = flat[start : start + (1 << 20)].double()
            norm2 += chunk.square().sum()
            if other is not None:
                ref = other.reshape(-1)[start : start + (1 << 20)].to(chunk)
                dot += (chunk * ref).sum()
    norm2, dot = float(norm2), float(dot)
    if not math.isfinite(norm2) or not math.isfinite(dot):
        raise ValueError("Nonfinite actor gradients; inspect the checkpoint or use float32")
    return norm2, dot


def measure_trajectory(model, episode, values, *, moments, normalization="none", head_chunk_size=128, checkpoint=True):
    """Two passes: sum segment contributions on CPU, then compare each with the sum.

    The actor always sees its original segment context, for either critic.
    Every segment contribution uses the trajectory's generated-token denominator.
    No optimizer, parameter .grad mutation, concatenated actor context, or retained
    multi-segment computation graph. Tied parameters are counted once.
    """
    segments = episode["segments"]
    counts = [sum(segment["loss_mask"]) for segment in segments]
    total = sum(counts)
    if len(values) != len(segments) or any(len(v) != n for v, n in zip(values, counts, strict=True)):
        raise ValueError("Critic values must align with every segment's generated tokens")
    if normalization not in {"none", "cohort"}:
        raise ValueError("Unknown advantage normalization")
    advantages = [torch.tensor([episode["reward"] - v for v in row], dtype=torch.float32) for row in values]
    if normalization == "cohort":
        advantages = [(a - moments["mean"]) / math.sqrt(moments["variance"] + 1e-8) for a in advantages]
    parameters = tuple(model.parameters())
    reference = [None] * len(parameters)

    def differentiate(index):
        if not counts[index]:
            return (None,) * len(parameters)
        losses = actor_loss_function(model, segments[index], advantages[index], head_chunk_size=head_chunk_size)
        return torch.autograd.grad(losses().sum() / total, parameters, allow_unused=True)

    with probe_mode(model), checkpoint_layers(model, checkpoint):
        for index in range(len(segments)):
            gradient = differentiate(index)
            for i, value in enumerate(gradient):
                if value is None:
                    continue
                # FP32 host accumulation avoids repeated BF16 rounding of the sum.
                if reference[i] is None:
                    reference[i] = value.detach().to(device="cpu", dtype=torch.float32, copy=True)
                else:
                    reference[i].add_(value.detach().to(device="cpu", dtype=torch.float32))
            del gradient
        total_norm2, _ = gradient_products(reference)
        total_norm = math.sqrt(total_norm2)
        rows = []
        for index, count in enumerate(counts):
            gradient = differentiate(index)
            norm2, dot = gradient_products(gradient, reference)
            del gradient
            norm = math.sqrt(norm2)
            cosine = dot / math.sqrt(norm2 * total_norm2) if norm2 and total_norm2 else None
            rows.append(
                dict(
                    segment_index=index,
                    prior_compactions=index,
                    total_compactions=len(segments) - 1,
                    context_tokens=len(segments[index]["input_ids"]),
                    loss_tokens=count,
                    token_fraction=count / total,
                    segment_grad_norm=norm,
                    segment_mean_grad_norm=norm * total / count if count else None,
                    trajectory_grad_norm=total_norm,
                    cosine_with_trajectory=max(-1.0, min(1.0, cosine)) if cosine is not None else None,
                    norm_ratio=norm / total_norm if total_norm else None,
                    segment_mean_norm_ratio=norm * total / count / total_norm if count and total_norm else None,
                    projection_on_trajectory=dot / total_norm2 if total_norm2 else None,
                    advantage_rms=float(advantages[index].square().mean().sqrt()) if count else None,
                    raw_advantage_rms=(
                        math.sqrt(sum((episode["reward"] - v) ** 2 for v in values[index]) / count) if count else None
                    ),
                )
            )
    trajectory = dict(
        total_compactions=len(segments) - 1,
        segments=len(segments),
        loss_tokens=total,
        context_tokens=sum(len(s["input_ids"]) for s in segments),
        trajectory_grad_norm=total_norm,
        parameter_count=sum(p.numel() for p in parameters),
        projection_sum=sum(row["projection_on_trajectory"] for row in rows) if total_norm else None,
        # This need not equal one: cancellation can make it arbitrarily large.
        sum_segment_norm_over_trajectory=(
            sum(row["segment_grad_norm"] for row in rows) / total_norm if total_norm else None
        ),
    )
    return trajectory, rows
