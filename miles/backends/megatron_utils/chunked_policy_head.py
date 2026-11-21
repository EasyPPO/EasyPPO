"""Megatron GPT output processor for packed, single-card PPO training."""

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.data.context_parallel import iter_local_response_rows
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state


def chunked_policy_head(
    *, hidden_states, output_layer, output_weight, runtime_gather_output, scale_logits, context, **kwargs
):
    args, batch = context
    parallel = get_parallel_state()
    with_entropy = args.entropy_coef != 0 or args.observe_training_entropy

    def score(hidden, targets):
        logits, _ = output_layer(hidden, weight=output_weight, runtime_gather_output=runtime_gather_output)
        log_probs, entropy = calculate_log_probs_and_entropy(
            scale_logits(logits).squeeze(1),
            targets,
            parallel.tp.group,
            with_entropy=with_entropy,
            entropy_requires_grad=args.entropy_coef != 0,
            temperature=args.rollout_temperature,
            chunk_size=args.log_probs_chunk_size,
            debug_unified_grad_fused_logprob=args.debug_unified_grad_fused_logprob,
        )
        columns = [log_probs.reshape(-1)]
        if with_entropy:
            columns.append(entropy.reshape(-1))
        return torch.stack(columns, dim=-1)

    scores = []
    layout = iter_local_response_rows(
        hidden_states.size(0),
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        qkv_format=args.qkv_format,
        allgather_cp=False,
        include_response_indices=False,
    )
    for ranges, targets, _ in layout:
        ((start, end),) = ranges
        for offset in range(0, end - start, args.ppo_head_chunk_size):
            n = min(args.ppo_head_chunk_size, end - start - offset)
            # Recompute this projection in backward. Autograd retains the compact
            # scores and decoder states, without saving each chunk's vocabulary logits.
            scores.append(
                checkpoint(
                    score,
                    hidden_states[start + offset : start + offset + n],
                    targets[offset : offset + n],
                    use_reentrant=False,
                )
            )
    if not scores:
        # Keep the head and decoder on the graph for empty-response microbatches.
        dummy = hidden_states.new_zeros((1,), dtype=torch.long)
        return score(hidden_states[:1], dummy)[:0]
    return torch.cat(scores, dim=0)
