"""HF forwards for frozen-critic values and fixed-policy gradient diagnostics."""

from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from agentic_critic_pipeline.records import work_items


def token_positions(segment):
    return [segment["response_start"] + i - 1 for i, flag in enumerate(segment["loss_mask"]) if flag]


@torch.no_grad()
def critic_values(model, head, episode, *, mode, max_tokens):
    """Evaluate V(s_t) at t-1, never at the already-generated target token."""
    result = [[] for _ in episode["segments"]]
    device = next(model.parameters()).device
    for item in work_items(episode, mode, max_tokens):
        ids = torch.tensor([item.input_ids], device=device, dtype=torch.long)
        hidden = model.base_model(input_ids=ids, use_cache=False).last_hidden_state[0]
        values = head(hidden[item.positions]).float()
        if not torch.isfinite(values).all():
            raise ValueError("Critic produced nonfinite values")
        for (segment, _), value in zip(item.locations, values.cpu().tolist(), strict=True):
            result[segment].append(value)
    return result


@contextmanager
def checkpoint_layers(model, enabled):
    """Non-reentrant checkpointing works in eval mode, preserving dropout off."""
    originals = []
    try:
        if enabled:
            layers = getattr(model.base_model, "layers", None)
            if layers is None:
                raise ValueError("Layer checkpointing requires base_model.layers; use --no-checkpoint-layers")
            for layer in layers:
                originals.append((layer, layer.__dict__.get("forward")))
                layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        yield
    finally:
        for layer, original in originals:
            if original is None:
                del layer.forward
            else:
                layer.forward = original


def actor_loss_function(model, segment, advantages, *, head_chunk_size=128):
    """Weighted NLL: gradient of unclipped PPO at theta=theta_old (ratio=1).

    Advantages are constants. Compute vocabulary logits only at supervised
    positions and checkpoint each head chunk, avoiding a persistent [T,V]
    activation. Decoder context is complete, including masked tool tokens.
    """
    if head_chunk_size < 1:
        raise ValueError("Head chunk size must be positive")
    if model.config.model_type not in {"llama", "qwen3_5_text"}:
        raise ValueError("Validated head decomposition supports Llama and Qwen3.5 text models only")
    device = next(model.parameters()).device
    positions = token_positions(segment)
    ids = torch.tensor([segment["input_ids"]], device=device, dtype=torch.long)
    hidden = model.base_model(input_ids=ids, use_cache=False).last_hidden_state[0]
    indices = torch.tensor(positions, device=device)
    selected = hidden[indices]
    targets = ids[0, indices + 1]
    advantages = torch.as_tensor(advantages, device=device, dtype=torch.float32).detach()
    if advantages.shape != targets.shape or not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite and aligned to generated tokens")

    def nll(states, labels):
        return F.cross_entropy(model.get_output_embeddings()(states).float(), labels, reduction="none")

    def losses(indices=None):
        states = selected if indices is None else selected[indices]
        labels = targets if indices is None else targets[indices]
        weights = advantages if indices is None else advantages[indices]
        chunks = [
            checkpoint(
                nll,
                states[start : start + head_chunk_size],
                labels[start : start + head_chunk_size],
                use_reentrant=False,
            )
            for start in range(0, len(labels), head_chunk_size)
        ]
        return torch.cat(chunks) * weights

    return losses
