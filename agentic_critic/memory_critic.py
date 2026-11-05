"""Value regression on independently remembered terminal episodes.

This module has no Miles/Ray dependency. The distributed adapter owns optimizer,
checkpoint and device lifetimes; each read here recomputes memory with the current
critic weights. Neither an actor cache nor a cache from an earlier update is reused.
"""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from agentic_critic.critic_input import Segment
from agentic_critic.critic_modes import ValueHead
from agentic_critic.critic_train import read_episode
from agentic_critic.miles_samples import critic_episode
from torch import nn

MEMORY_MODES = ("full_detach", "actor-summary")


@dataclass(frozen=True)
class MemoryEpisode:
    segments: list[Segment]
    reward: float


class MemoryCritic(nn.Module):
    """An independently optimized text backbone and scalar value head."""

    def __init__(self, backbone: nn.Module, head: ValueHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def values(self, episode: MemoryEpisode, *, mode: str, device) -> list[torch.Tensor]:
        self._check_mode(mode)
        output = []

        def collect(items):
            output.extend(value.detach().float().cpu() for _, value in items)

        with torch.no_grad():
            read_episode(
                self.backbone,
                self.head,
                episode.segments,
                self._reader_mode(mode),
                device,
                collect,
                train=False,
            )
        return output

    def backward_episode(
        self,
        episode: MemoryEpisode,
        old_values: list[torch.Tensor],
        *,
        mode: str,
        device,
        value_clip: float,
        loss_coef: float,
        normalizer: float,
        loss_weight: float = 1.0,
        checkpoint: bool = True,
    ) -> float:
        """Accumulate a token-sum value loss divided by the supplied global count.

        gamma=lambda=1 and zero reward KL make every trainable token's target
        exactly the terminal reward. Clipping always uses the rollout's fixed
        pre-update values. Prompt weights scale the numerator, never the count.
        """
        self._check_mode(mode)
        if normalizer <= 0:
            raise ValueError("Memory critic requires a positive global loss-token count")
        if not math.isfinite(loss_weight) or loss_weight < 0:
            raise ValueError("Memory critic loss weights must be finite and nonnegative")
        loss_sum = 0.0

        def backward(items):
            nonlocal loss_sum
            for index, value in items:
                mask = torch.as_tensor(episode.segments[index].loss_mask, device=device, dtype=torch.bool)
                old = old_values[index].to(device=device)
                clipped = old + (value.float() - old).clamp(-value_clip, value_clip)
                errors = torch.maximum((value.float() - episode.reward).square(), (clipped - episode.reward).square())
                loss = loss_coef * loss_weight * errors[mask].sum()
                loss_sum += float(loss.detach())
                (loss / normalizer).backward()

        read_episode(
            self.backbone,
            self.head,
            episode.segments,
            self._reader_mode(mode),
            device,
            backward,
            train=True,
            use_checkpoint=checkpoint,
        )
        return loss_sum

    @staticmethod
    def _check_mode(mode):
        if mode not in MEMORY_MODES:
            raise ValueError(f"Unknown memory critic mode: {mode}")

    @staticmethod
    def _reader_mode(mode):
        return "shared" if mode == "actor-summary" else "full_detach"


@dataclass(frozen=True)
class MemoryTrainResult:
    values: list[list[torch.Tensor]]
    loss_sum: float
    token_count: float
    grad_norms: list[float]
    learning_rates: list[float]
    optimizer_steps: int


def train_memory_minibatches(
    model: MemoryCritic,
    optimizer,
    episodes: list[MemoryEpisode],
    *,
    minibatch_indices: list[list[int]],
    loss_weights: list[float],
    mode: str,
    device,
    value_clip: float,
    loss_coef: float,
    learning_rate: float,
    warmup_steps: int,
    optimizer_steps: int,
    clip_grad: float,
    checkpoint: bool = True,
    on_episode=None,
) -> MemoryTrainResult:
    """Train one rollout epoch, preserving its pre-update values.

    The producer owns prompt grouping and minibatch order. Each episode appears
    exactly once locally. All ranks execute the same number of optimizer steps;
    every step reduces its own *unweighted* token count. Each backward call
    rebuilds the episode memory with the current weights, so neither forward
    caches nor old graphs survive an optimizer update.
    """
    _validate_minibatches(episodes, minibatch_indices, loss_weights)
    device = torch.device(device)
    values = []
    with _autocast(device):
        for episode in episodes:
            values.append(model.values(episode, mode=mode, device=device))
            if on_episode is not None:
                on_episode()
    loss_sum, token_count = 0.0, 0.0
    norms, rates = [], []
    for indices in minibatch_indices:
        count = torch.tensor(
            sum(sum(segment.loss_mask) for i in indices for segment in episodes[i].segments),
            device=device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
            dist.all_reduce(count)
        if count.item() <= 0:
            raise ValueError("Memory critic minibatch received no trainable tokens")
        token_count += count.item()
        optimizer.zero_grad()
        for i in indices:
            with _autocast(device):
                loss_sum += model.backward_episode(
                    episodes[i],
                    values[i],
                    mode=mode,
                    device=device,
                    value_clip=value_clip,
                    loss_coef=loss_coef,
                    normalizer=count.item(),
                    loss_weight=float(loss_weights[i]),
                    checkpoint=checkpoint,
                )
            if on_episode is not None:
                on_episode()
        optimizer_steps += 1
        lr = learning_rate * min(1.0, optimizer_steps / warmup_steps) if warmup_steps else learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr
        rates.append(lr)
        norms.append(optimizer.step(clip_grad=clip_grad))
    return MemoryTrainResult(values, loss_sum, token_count, norms, rates, optimizer_steps)


def _validate_minibatches(episodes, minibatch_indices, loss_weights):
    if not minibatch_indices or sorted(i for indices in minibatch_indices for i in indices) != list(
        range(len(episodes))
    ):
        raise ValueError("Memory critic minibatches must cover every local episode exactly once")
    if len(loss_weights) != len(episodes) or any(not math.isfinite(w) or w < 0 for w in loss_weights):
        raise ValueError("Memory critic requires one finite nonnegative loss weight per episode")


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def flatten_episode_values(episode: MemoryEpisode, values: list[torch.Tensor]) -> torch.Tensor:
    """Recover the episode response layout consumed by Miles' value_slices."""
    sample, spans = critic_episode(episode.segments)
    output = torch.zeros(sample.response_length, dtype=torch.float32)
    for (start, length), value in zip(spans, values, strict=True):
        if len(value) != length:
            raise ValueError("Memory critic values do not match their actor response span")
        output[start : start + length] = value
    return output
