"""Cache updates for hybrid attention layers."""

from contextlib import contextmanager

import torch


@contextmanager
def assigning_cache_updates():
    """Make the linear-attention cache store new states by assignment instead of in-place copy.

    transformers writes the recurrent and conv states into the cache with `copy_` (to keep a static
    address for compiled inference). During training that overwrites a tensor the graph still needs,
    and backward fails. Assignment keeps the old tensor intact; the values are identical.
    """
    from transformers.cache_utils import LinearAttentionLayer

    orig_conv, orig_rec = LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state

    def update_conv_state(self, conv_states, **kwargs):
        if not self.is_conv_states_initialized:
            self.dtype, self.device = conv_states.dtype, conv_states.device
            self.max_batch_size, self.conv_kernel_size = conv_states.shape[0], conv_states.shape[-1]
            self.is_conv_states_initialized = True
        k, new = self.conv_kernel_size, conv_states.shape[-1]
        if not self.has_previous_state or new >= k:
            self.conv_states = conv_states[..., -k:]
        else:
            self.conv_states = torch.cat([self.conv_states[..., new:], conv_states], dim=-1)
        self.has_previous_state = True
        return self.conv_states

    def update_recurrent_state(self, recurrent_states, **kwargs):
        self.is_recurrent_states_initialized = True
        self.recurrent_states = recurrent_states
        return self.recurrent_states

    LinearAttentionLayer.update_conv_state = update_conv_state
    LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    try:
        yield
    finally:
        LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state = orig_conv, orig_rec
