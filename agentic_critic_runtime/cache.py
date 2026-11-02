"""Cache updates for hybrid attention layers."""

import inspect
from contextlib import contextmanager

import torch


@contextmanager
def assigning_cache_updates():
    """Make the linear-attention cache store new states by assignment instead of in-place copy.

    transformers writes the recurrent and conv states into the cache with `copy_` (to keep a static
    address for compiled inference). During training that overwrites a tensor the graph still needs,
    and backward fails. Assignment keeps the old tensor intact; the values are identical.

    transformers >= 5.19 keeps several states per layer (`state_idx`), and `update_conv_state` must
    return the cached left context concatenated with the new chunk so that chunked prefill is exact;
    that version is mirrored here with the same semantics. Older versions take a single state.
    """
    from transformers.cache_utils import LinearAttentionLayer

    orig_conv, orig_rec = LinearAttentionLayer.update_conv_state, LinearAttentionLayer.update_recurrent_state
    multi_state = "state_idx" in inspect.signature(orig_conv).parameters

    if multi_state:

        def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
            if not self.is_conv_states_initialized[state_idx]:
                self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
            k = self.conv_kernel_size[state_idx]
            if not self.has_previous_state[state_idx]:
                full = conv_states
                self.has_previous_state[state_idx] = True
                if not self.record_past and full.shape[-1] < k:
                    full = torch.nn.functional.pad(full, (k - full.shape[-1], 0), value=0)
            else:
                full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
            self.conv_states[state_idx] = full if self.record_past else full[..., -k:]
            return full

        def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            self.recurrent_states[state_idx] = recurrent_states
            return self.recurrent_states[state_idx]

    else:

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
