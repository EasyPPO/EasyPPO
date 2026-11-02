"""Native TE attention with official AM cache/beta adaptation for the critic."""

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

from agentic_critic.context_memory import packed_attention


class CriticMemoryAttention(TEDotProductAttention):
    def __init__(self, config, *args, memory_rotary_dim, memory_rotary_base, **kwargs):
        if getattr(config, "softmax_type", "vanilla") != "vanilla" or getattr(config, "window_size", None):
            raise ValueError("AM context requires vanilla full attention without a sliding window")
        super().__init__(config, *args, **kwargs)
        self.memory_rotary_dim = memory_rotary_dim
        self.memory_rotary_base = memory_rotary_base
        # The outer model remains THD (including GDN segment resets). The core
        # handles one independent Q/KV pair at a time through TE's native BSHD API.
        self._memory_seq_params = PackedSeqParams(qkv_format="bshd")

    def _read_native(self, q, k, v, *, fixed_bias=None, scale=None):
        bias = None
        mask_type = AttnMaskType.causal
        if fixed_bias is not None:
            mask_type = AttnMaskType.causal_bottom_right
            groups = q.shape[0] // k.shape[0]
            # Native post_scale_bias supplies the official per-key beta. The
            # causal pattern is represented by TE, not an explicit Python mask.
            beta = q.new_zeros(q.shape[0], k.shape[1])
            beta[:, :fixed_bias.shape[1]] = fixed_bias.repeat_interleave(groups, 0).to(q.dtype)
            bias = beta[None, :, None].expand(1, q.shape[0], q.shape[1], k.shape[1])
        output = super().forward(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
            None, mask_type, attention_bias=bias, packed_seq_params=self._memory_seq_params,
        )
        return output.reshape(q.shape[1], q.shape[0], v.shape[-1]).transpose(0, 1)

    def forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None, **kwargs):
        context = getattr(packed_seq_params, "critic_context", None)
        if context is None:
            return super().forward(query, key, value, attention_mask, attn_mask_type, attention_bias=attention_bias, packed_seq_params=packed_seq_params, **kwargs)
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("AM packed attention accepts only the validated causal actor segments")
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("AM native core requires THD tensors and TP/CP/PP=1")
        return packed_attention(
            query, key, value, context["layouts"], packed_seq_params.cu_seqlens_host,
            ratio=context["ratio"], rotary_dim=self.memory_rotary_dim,
            base=self.memory_rotary_base, scale=query.shape[-1] ** -0.5,
            attention=self._read_native,
        )
