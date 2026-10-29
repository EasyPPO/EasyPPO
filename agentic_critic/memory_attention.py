"""SDPA reads of biased KV memory, without allocating a full segment-sized mask."""

from contextlib import contextmanager

import torch
from torch.nn import functional as F


def _biased_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """One unpadded sequence: physical memory followed by a causal current segment.

    Rebuild the mask from physical sizes, independently of logical RoPE positions or other layers'
    cache lengths. Chunking bounds the explicit mask during prefill/probing. Decoder-layer checkpointing
    releases the masks between the training forward and backward recomputation.
    """
    bias = module._compaction_bias
    new, total = query.shape[-2], key.shape[-2]
    memory = total - new
    if bias.shape != (key.shape[1], memory):
        raise ValueError(f"memory bias shape {tuple(bias.shape)} does not match KV shape {tuple(key.shape)}")
    per_head = bias.repeat_interleave(module.num_key_value_groups, dim=0).to(query.dtype)
    outputs = []
    key_positions = torch.arange(total, device=query.device)
    for start in range(0, new, 128):
        end = min(start + 128, new)
        positions = torch.arange(start, end, device=query.device) + memory
        future = key_positions[None, :] > positions[:, None]
        mask = query.new_zeros(1, query.shape[1], end - start, total)
        mask[:, :, :, :memory] = per_head[None, :, None, :]
        mask.masked_fill_(future[None, None], float("-inf"))
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout,
                scale=scaling,
                enable_gqa=key.shape[1] != query.shape[1],
            )
        )
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


@contextmanager
def biased_attention(model, layers, biases):
    # Optional transformers dependency is loaded only for model execution.
    from transformers import AttentionInterface

    if not biases:
        yield
        return
    AttentionInterface.register("compaction_sdpa", _biased_sdpa)
    previous = model.config._attn_implementation
    saved = []
    try:
        for idx, attn in layers:
            if idx not in biases:
                raise ValueError(f"missing memory bias for full-attention layer {idx}")
            saved.append((attn, getattr(attn, "_compaction_bias", None)))
            attn._compaction_bias = biases[idx]
        model.config._attn_implementation = "compaction_sdpa"
        yield
    finally:
        model.config._attn_implementation = previous
        for attn, old in saved:
            if old is None:
                del attn._compaction_bias
            else:
                attn._compaction_bias = old
