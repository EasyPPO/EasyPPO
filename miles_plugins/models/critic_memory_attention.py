"""Native Megatron full-attention core for problem / AM memory / actor resume.

The surrounding Megatron attention keeps its original projections, Q/K norms,
output gate and output projection. Actor and non-AM calls use the original TE
kernel. Only critic calls carrying validated episode layouts use AM reads.
"""

from megatron.core.extensions.transformer_engine import TEDotProductAttention

from agentic_critic.context_memory import packed_attention


class CriticMemoryAttention(TEDotProductAttention):
    def __init__(self, config, *args, memory_rotary_dim, memory_rotary_base, **kwargs):
        if getattr(config, "softmax_type", "vanilla") != "vanilla" or getattr(config, "window_size", None):
            raise ValueError("AM context requires vanilla full attention without a sliding window")
        super().__init__(config, *args, **kwargs)
        self.memory_rotary_dim = memory_rotary_dim
        self.memory_rotary_base = memory_rotary_base

    def forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None, **kwargs):
        context = getattr(packed_seq_params, "critic_context", None)
        if context is None:
            return super().forward(query, key, value, attention_mask, attn_mask_type, attention_bias=attention_bias, packed_seq_params=packed_seq_params, **kwargs)
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("AM packed attention accepts only the validated causal actor segments")
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("AM native core requires THD tensors and TP/CP/PP=1")
        return packed_attention(
            query,
            key,
            value,
            context["layouts"],
            packed_seq_params.cu_seqlens_host,
            ratio=context["ratio"],
            rotary_dim=self.memory_rotary_dim,
            base=self.memory_rotary_base,
            scale=query.shape[-1] ** -0.5,
        )
