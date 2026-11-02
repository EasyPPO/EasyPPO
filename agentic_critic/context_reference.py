"""HF frozen/reference evaluation of the same stateless native Miles AM reads.

This is not a second training backend. Each original actor segment is forwarded
without past_key_values, resetting all linear states exactly as actor-summary.
Only full-attention operators receive the explicit detached AM memory.
"""

from contextlib import contextmanager

import torch

from agentic_critic.context_layout import episode_layout
from agentic_critic.context_memory import fit_memory, segment_attention


@contextmanager
def _full_attention(model, layout, memories, fits):
    # Transformers is optional for native Miles and pure tensor tests.
    from transformers import AttentionInterface

    def attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
        rotary = getattr(model.config, "partial_rotary_factor", 0.25)
        rope = getattr(model.config, "rope_parameters", None) or {}
        rotary = rope.get("partial_rotary_factor", rotary)
        base = rope.get("rope_theta", getattr(model.config, "rope_theta", 10000000.0))
        out, fit = segment_attention(
            query[0],
            key[0],
            value[0],
            layout,
            memories.get(module.layer_idx),
            rotary_dim=int(query.shape[-1] * rotary),
            base=base,
            scale=scaling,
        )
        if fit is not None:
            fits[module.layer_idx] = fit
        return out.transpose(0, 1)[None], None

    AttentionInterface.register("critic_context_am", attention)
    previous = model.config._attn_implementation
    model.config._attn_implementation = "critic_context_am"
    try:
        yield
    finally:
        model.config._attn_implementation = previous


def read_reference(model, head, segments, device, on_values, *, ratio, ridge=1e-4):
    """Frozen evaluation only, retaining the original response/value alignment."""
    model = getattr(model, "model", model)
    if model.config.model_type != "qwen3_5_text":
        raise ValueError("AM reference requires a Qwen3.5 text backbone")
    layouts = episode_layout(segments, episode="reference")
    memories = {}
    with torch.no_grad():
        for index, (seg, layout) in enumerate(zip(segments, layouts, strict=True)):
            fits = {}
            with _full_attention(model, layout, memories, fits):
                result = model(input_ids=torch.tensor([seg.tokens], device=device), use_cache=False)
            start = seg.response_start - 1
            on_values([(index, head(result.last_hidden_state[0, start : start + seg.response_length]))])
            memories = {layer: fit_memory(fit, ratio=ratio, ridge=ridge) for layer, fit in fits.items()}
