"""Native TE attention with detached episode memory, using TE's existing kernel."""

import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType

from miles_plugins.models.detached_memory import detached_attention, memory_segments


class DetachedMemoryDotProductAttention(TEDotProductAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_attention.register_forward_pre_hook(self._fa4_causal_window, with_kwargs=True)

    @staticmethod
    def _fa4_causal_window(module, inputs, kwargs):
        # TE passes (-1, 0) for unlimited causal attention. FA4 interprets
        # that pair as a local window and its SM100 HD256 backward rejects it.
        query = inputs[0]
        if (
            kwargs.get("attn_mask_type") == "causal_bottom_right"
            and kwargs.get("window_size") == (-1, 0)
            and getattr(kwargs.get("flash_attention_backend"), "major", None) == 4
            and query.ndim == 4
            and query.shape[-1] == 256
            and torch.cuda.get_device_capability(query.device) == (10, 0)
        ):
            return inputs, {**kwargs, "window_size": (None, None)}

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        segments = memory_segments(packed_seq_params, query.shape[0])
        if segments is None:
            return super().forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                **kwargs,
            )
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("Detached critic attention requires native THD Q/K/V tensors")
        if self.config.context_parallel_size != 1:
            raise ValueError("Detached critic attention currently requires CP=1")
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("Detached critic attention supports the causal mask without an extra bias")

        # TE supports rectangular bottom-right causal SBHD directly. Converting
        # one segment to batch size one avoids adding a THD-specific mask enum
        # to Megatron or allocating a dense attention mask/padded dummy queries.
        native_forward = super().forward

        def attend(own_query, own_key, own_value):
            # Contiguous token slices can retain nonzero storage offsets. TE's
            # contiguous() fallback leaves those views unchanged and rejects them.
            own_query, own_key, own_value = (
                tensor.clone(memory_format=torch.contiguous_format) if tensor.storage_offset() else tensor
                for tensor in (own_query, own_key, own_value)
            )
            output = native_forward(
                own_query.unsqueeze(1),
                own_key.unsqueeze(1),
                own_value.unsqueeze(1),
                None,
                attn_mask_type=AttnMaskType.causal_bottom_right,
                packed_seq_params=None,
                **kwargs,
            )
            return output.reshape(own_query.shape[0], -1)

        return detached_attention(query, key, value, segments, attend)
