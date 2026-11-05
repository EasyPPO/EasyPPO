import copy
import logging
from typing import Any

import torch

from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils import object_store
from miles.utils.dp_schedule import build_dp_schedule, has_full_schedule_config
from miles.utils.flops_utils import calculate_fwd_flops
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.object_store import ValueSpec
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.timer import Timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": "int32",
    "target_tokens": "int32",
    "loss_masks": "int32",
    "rollout_log_probs": "float32",
    "rollout_topk_token_ids": "int32",
    "rollout_topk_lengths": "int32",
    "rollout_topk_log_probs": "float32",
    "rollout_sampling_mask_ids": "int32",
    "rollout_sampling_mask_offsets": "int64",
    "teacher_log_probs": "float32",
    "opd_reverse_kl": "float32",
    "rollout_routed_experts": "int32",
    "rollout_indexer_topk": "int32",
}

ROLLOUT_DATA_VALUE_SPEC: dict[str, ValueSpec] = {
    **{field: ValueSpec(codec="typed_ragged") for field in ROLLOUT_DATA_TENSOR_DTYPES},
    "partition": ValueSpec(codec="ndarray", dtype="int64"),
    "seq_witness_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
    "rewards": ValueSpec(codec="ndarray", dtype="float32"),
    "critic_loss_weights": ValueSpec(codec="ndarray", dtype="float32"),
    "truncated": ValueSpec(codec="ndarray", dtype="int64"),
    "round_number": ValueSpec(codec="ndarray", dtype="int64"),
    "sample_indices": ValueSpec(codec="ndarray", dtype="int64"),
    "prompt_group_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_mask_sums": ValueSpec(codec="ndarray", dtype="int64"),
    "multimodal_train_inputs": ValueSpec(codec="ragged_tensor_dict"),
    "prompt": ValueSpec(codec="msgpack_ragged"),
    "metadata": ValueSpec(codec="msgpack_ragged"),
    "critic_segment_lengths": ValueSpec(codec="msgpack_ragged"),
    "weight_versions": ValueSpec(codec="msgpack_ragged"),
    "raw_reward": ValueSpec(codec="auto"),
    "total_lengths": ValueSpec(codec="auto"),
    "dynamic_global_batch_size": ValueSpec(codec="auto"),
    "num_microbatches": ValueSpec(codec="auto"),
    "micro_batch_indices": ValueSpec(codec="auto"),
    "num_rollouts": ValueSpec(codec="auto"),
}


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        return f(args, samples)

    raw_rewards, rewards = _post_process_rewards(
        args,
        samples,
        custom_reward_post_process_func=custom_reward_post_process_func,
        prompt_group_sizes=metadata.get("prompt_group_sizes"),
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": [s.rollout_id if s.rollout_id is not None else s.index for s in samples],
    }
    if getattr(args, "critic_context_mode", "shared") in ("full_detach", "actor-summary") and all(
        sample.group_index is not None for sample in samples
    ):
        # A prompt's responses can contain different numbers of compacted rows.
        # Preserve their source identity instead of inferring groups from row counts.
        train_data["prompt_group_ids"] = [sample.group_index for sample in samples]

    if getattr(args, "filter_truncated_in_actor_loss", False) and samples and all(train_data["truncated"]):
        # Same refusal as the EasyPPO reference: an actor step without a single loss token is not a no-op
        # for Adam (momentum, weight decay), so it must not happen silently.
        raise ValueError(
            "--filter-truncated-in-actor-loss removed every training sample of the rollout batch; "
            "increase --rollout-max-response-len or the batch size."
        )

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks
    # Only build_critic_samples attaches this private field. Dataset metadata
    # must never activate detached memory on actor or ordinary critic inputs.
    segment_lengths = [getattr(sample, "_critic_segment_lengths", None) for sample in samples]
    if any(lengths is not None for lengths in segment_lengths):
        for sample, lengths in zip(samples, segment_lengths, strict=True):
            if (
                not lengths
                or any(type(length) is not int or length <= 0 for length in lengths)
                or sum(lengths) != len(sample.tokens)
            ):
                raise ValueError("Critic segment lengths must cover every real token of each critic sample")
        train_data["critic_segment_lengths"] = segment_lengths

    train_data["rollout_mask_sums"] = _compute_rollout_mask_sums(train_data["rollout_ids"], loss_masks)

    if (std_floor := getattr(args, "critic_return_std_floor", None)) is not None:
        train_data["critic_loss_weights"] = _compute_critic_loss_weights(
            rewards=rewards,
            loss_masks=loss_masks,
            prompt_groups=_reward_group_segments(args, samples, metadata.get("prompt_group_sizes")),
            std_floor=std_floor,
            rollout_ids=train_data["rollout_ids"],
        )

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if k := args.rollout_top_logprobs_num:
        for sample in samples:
            for field in ("rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"):
                if getattr(sample, field) is None:
                    raise ValueError(
                        f"Rollout top-k logprobs collection requires {field}; "
                        f"sample_index={sample.index}. Custom rollout producers must record probabilities during generation."
                    )
        if args.ci_test:
            for sample in samples:
                validate_rollout_topk_logprobs_sample(sample, k)
        train_data["rollout_topk_token_ids"] = [sample.rollout_topk_token_ids for sample in samples]
        train_data["rollout_topk_log_probs"] = [sample.rollout_topk_log_probs for sample in samples]

    has_sampling_mask = any(sample.rollout_sampling_mask is not None for sample in samples)
    if has_sampling_mask:
        sampling_mask_ids = []
        sampling_mask_offsets = []
        for position, sample in enumerate(samples):
            sample.validate()
            if sample.rollout_sampling_mask is None:
                raise ValueError(
                    "sampling-mask data must be present for every training sample; "
                    f"missing at position={position}, sample_index={sample.index}, status={sample.status}"
                )
            ids, offsets = sample.rollout_sampling_mask._as_tensors()

            sampling_mask_ids.append(ids)
            sampling_mask_offsets.append(offsets)

        train_data["rollout_sampling_mask_ids"] = sampling_mask_ids
        train_data["rollout_sampling_mask_offsets"] = sampling_mask_offsets
        if k and args.loss_type == "score_centering" and args.rollout_sampling_logprobs_mode == "support":
            _compact_candidate_ids(train_data)

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    elif getattr(args, "use_rollout_routing_replay", False):
        raise ValueError(
            "--use-rollout-routing-replay is set but the rollout samples carry no "
            "rollout_routed_experts: the engine response meta_info lacked 'routed_experts'."
        )

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]
    elif getattr(args, "use_rollout_indexer_replay", False):
        raise ValueError(
            "--use-rollout-indexer-replay is set but the rollout samples carry no "
            "rollout_indexer_topk: the engine response meta_info lacked 'indexer_topk'."
        )

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [[call.to_dicts() for call in sample.weight_versions] for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if (prompt_group_sizes := metadata.get("prompt_group_sizes")) is not None:
        train_data["prompt_group_sizes"] = prompt_group_sizes

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _compact_candidate_ids(data: dict) -> None:
    """Share ordered support IDs in training batches; preserve other producer layouts."""
    lengths = []
    for dense, logps, support, offsets in zip(
        data["rollout_topk_token_ids"],
        data["rollout_topk_log_probs"],
        data["rollout_sampling_mask_ids"],
        data["rollout_sampling_mask_offsets"],
        strict=True,
    ):
        dense, logps = torch.as_tensor(dense), torch.as_tensor(logps)
        if dense.ndim != 2 or dense.shape != logps.shape or dense.size(0) != offsets.numel() - 1:
            return
        valid = dense >= 0
        count = valid.sum(-1, dtype=torch.int32)
        columns = torch.arange(dense.size(1))
        prefix = columns < count[:, None]
        if not torch.equal(valid, prefix) or not (dense[~prefix] == -1).all():
            return
        if not (count[count > 0] == offsets.diff()[count > 0]).all():
            return
        positions = offsets[:-1, None] + columns
        if not torch.equal(dense[prefix], support[positions[prefix]]):
            return
        if not torch.isneginf(logps[count == 0]).all():
            return
        lengths.append(count)
    data["rollout_topk_lengths"] = lengths
    del data["rollout_topk_token_ids"]


def _compute_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    """Whole-rollout loss-mask total per sample: every sibling of one rollout carries
    the sum over all of that rollout's samples, so the loss reducer reconstructs one
    token-weighted mean per rollout even when siblings land in different micro-batches."""
    totals: dict[int, int] = {}
    for rid, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rid] = totals.get(rid, 0) + sum(mask)
    return [totals[rid] for rid in rollout_ids]


def _compute_critic_loss_weights(
    *,
    rewards: list[float],
    loss_masks: list[list[int]],
    prompt_groups: list[list[int]],
    std_floor: float,
    rollout_ids: list | None = None,
) -> list[float]:
    """EasyPPO noise-normalized critic weights, one per sample (arXiv 2609.36802).

    Every sample of a prompt gets ``1 / max(std, std_floor)``, with ``std`` the
    population std of that prompt's rewards in this rollout batch; the weights
    are then divided by their mean over prompts, so the unweighted loss scale is
    kept. Samples with an empty loss mask get weight 0 and do not enter the
    statistics. Mirrors ``compute_prompt_variance_loss_weights`` of the EasyPPO
    reference (beta=0.5, no w_max) operation by operation, prompt order included.

    With ``rollout_ids``, samples that share a rollout id (the segments of one compact rollout, which also share
    the reward) count once in a prompt's std. Otherwise an episode with more segments would weigh more, and the
    weights would differ between a critic that trains on segments and one that trains on whole episodes.
    """
    group_of_row = {row: group for group, rows in enumerate(prompt_groups) for row in rows}
    valid_rows_by_group: dict[int, list[int]] = {}
    for row, loss_mask in enumerate(loss_masks):
        if any(loss_mask):
            valid_rows_by_group.setdefault(group_of_row[row], []).append(row)
    if not valid_rows_by_group:
        raise ValueError("--critic-return-std-floor: every sample of the rollout batch has an empty loss mask")

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    weights = torch.zeros_like(rewards_tensor)
    prompt_weights = []
    for rows in valid_rows_by_group.values():
        index = torch.tensor(rows, dtype=torch.long)
        stat_rows = rows
        if rollout_ids is not None:
            first_of_rollout: dict = {}
            for row in rows:
                first_of_rollout.setdefault(rollout_ids[row] if rollout_ids[row] is not None else ("row", row), row)
            stat_rows = list(first_of_rollout.values())
        prompt_std = torch.var(rewards_tensor[torch.tensor(stat_rows, dtype=torch.long)], correction=0).pow(0.5)
        prompt_weight = torch.clamp(prompt_std, min=std_floor).reciprocal()
        prompt_weights.append(prompt_weight)
        weights[index] = prompt_weight

    weights /= torch.stack(prompt_weights).mean()
    return weights.tolist()


def _reward_group_segments(args: Any, samples: list[Sample], prompt_group_sizes: list[int] | None) -> list[list[int]]:
    """Return the flattened row indices for each prompt reward group."""
    # Multi-LoRA records explicit prompt boundaries before flattening.
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(
            samples
        ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(samples)} rewards"
        groups: list[list[int]] = []
        start = 0
        for size in prompt_group_sizes:
            end = start + size
            if size > 0:
                groups.append(list(range(start, end)))
            start = end
        return groups

    # Standard rollout samples carry their prompt identity in `group_index`.
    group_indices = [sample.group_index for sample in samples]
    if all(group_index is not None for group_index in group_indices):
        segments_by_group_index: dict[int, list[int]] = {}
        for segment_index, group_index in enumerate(group_indices):
            segments_by_group_index.setdefault(int(group_index), []).append(segment_index)
        return list(segments_by_group_index.values())

    # Legacy fixed-fanout batches store each prompt's segments contiguously.
    expected_samples = args.n_samples_per_prompt * args.rollout_batch_size
    if len(samples) == expected_samples:
        return [
            list(range(start, start + args.n_samples_per_prompt))
            for start in range(0, len(samples), args.n_samples_per_prompt)
        ]
    # Without prompt identities or a complete fixed layout, use one reward group.
    return [list(range(len(samples)))]


def _normalize_rewards_by_rollout(
    args: Any,
    samples: list[Sample],
    raw_rewards: list[float],
    prompt_group_sizes: list[int] | None,
) -> list[float]:
    """Normalize one shared reward per rollout, then broadcast it to siblings."""
    if not samples:
        return []

    normalized_rewards = torch.empty(len(raw_rewards), dtype=torch.float)
    for prompt_segments in _reward_group_segments(args, samples, prompt_group_sizes):
        segments_by_rollout_key: dict[int | tuple[str, int], list[int]] = {}
        for segment_index in prompt_segments:
            sample = samples[segment_index]
            if sample.rollout_id is not None:
                rollout_key = sample.rollout_id
            elif sample.index is not None:
                rollout_key = sample.index
            else:
                rollout_key = ("row", segment_index)
            segments_by_rollout_key.setdefault(rollout_key, []).append(segment_index)

        rollout_segment_groups = list(segments_by_rollout_key.items())
        shared_rewards: list[float] = []
        for rollout_key, rollout_segments in rollout_segment_groups:
            sibling_rewards = [raw_rewards[segment_index] for segment_index in rollout_segments]
            if any(reward != sibling_rewards[0] for reward in sibling_rewards[1:]):
                raise ValueError(
                    f"all samples in rollout {rollout_key!r} must share one reward; "
                    f"rows {rollout_segments} have rewards {sibling_rewards}"
                )
            shared_rewards.append(sibling_rewards[0])

        rollout_rewards = torch.tensor(shared_rewards, dtype=torch.float)
        normalized_rollout_rewards = rollout_rewards - rollout_rewards.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and len(rollout_rewards) > 1:
            rollout_std = rollout_rewards.std()
            if rollout_std > 0:
                normalized_rollout_rewards = normalized_rollout_rewards / (rollout_std + 1e-6)

        for (_, rollout_segments), normalized_reward in zip(
            rollout_segment_groups, normalized_rollout_rewards.tolist(), strict=True
        ):
            for segment_index in rollout_segments:
                normalized_rewards[segment_index] = normalized_reward

    return normalized_rewards.tolist()


def _post_process_rewards(
    args,
    samples: list[Sample] | list[list[Sample]],
    custom_reward_post_process_func,
    prompt_group_sizes: list[int] | None = None,
):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        normalized_rewards = _normalize_rewards_by_rollout(args, samples, raw_rewards, prompt_group_sizes)
        return raw_rewards, normalized_rewards

    return raw_rewards, raw_rewards


def split_train_data_by_dp(args, data: dict[str, Any], train_parallel_config: dict | None):
    """Split the train data across DP ranks and put the shards into the object store.

    When the training backend can consume a rollout-side schedule, the shards
    also carry the precomputed micro-batch layout; otherwise this falls back to
    the legacy split (the training side schedules locally)."""
    shards = split_train_data_by_dp_shards(args, data, train_parallel_config)
    store = object_store.get_instance()
    return [store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC) for shard in shards]


def split_train_data_by_dp_shards(args, data: dict[str, Any], train_parallel_config: dict | None) -> list[dict]:
    """The per-rank shards of `split_train_data_by_dp`, before they go into the object store."""
    if can_schedule_on_rollout_side(args, data, train_parallel_config):
        return split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=train_parallel_config)
    return split_train_data_by_dp_raw(args, data, dp_size=train_parallel_config["dp_size"])


def split_full_history_shards(
    args,
    data: dict[str, Any],
    critic_data: dict[str, Any],
    spans: list[tuple[int, int, int]],
    parallel: dict,
) -> tuple[list[dict], list[dict]]:
    """Keep each complete critic episode and all its actor segments on the same DP rank.

    Adapted from EasyPPO/EasyPPO's actor-summary-compact branch (72fa1ce).
    The shared-context split remains unchanged. Memory critics keep whole prompt
    groups in each optimizer mini-batch, then reuse the episode planner inside it.
    One episode per forward micro-batch keeps collective counts equal even when
    ranks receive different numbers of actor segments.
    """
    if getattr(args, "critic_context_mode", "shared") not in ("full_history", "full_detach", "actor-summary"):
        raise ValueError("Episode-aligned actor/critic sharding requires --critic-context-mode full_history")
    if not args.use_dynamic_batch_size:
        raise ValueError("full_history episode packing requires --use-dynamic-batch-size")
    if parallel.get("cp_size", 1) != 1:
        # value_slices indexes full response positions, not CP-local value shards.
        raise ValueError("full_history value slicing currently requires context parallel size 1")
    if (parallel.get("vpp_size") or 1) != 1:
        raise ValueError("full_history episode packing currently requires virtual pipeline size 1")
    if "adapter_slots" in data or "multimodal_train_inputs" in data:
        raise ValueError("full_history episode packing does not support multi-LoRA or multimodal inputs")
    dp_size = parallel["dp_size"]
    batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    actor_lengths = [len(tokens) for tokens in data["tokens"]]
    critic_lengths = [len(tokens) for tokens in critic_data["tokens"]]
    critic_rows, siblings = _critic_episode_rows(data, critic_data, spans)
    num_episodes = len(critic_rows)
    if dp_size <= 0 or batch_size <= 0 or batch_size % dp_size:
        raise ValueError("full_history global batch size must be positive and divisible by the data-parallel size")
    if not num_episodes or num_episodes % batch_size:
        raise ValueError("full_history requires complete logical batches of episodes; partial batches are unsupported")
    critic_steps = getattr(args, "critic_num_steps_per_rollout", None)
    critic_steps = 1 if critic_steps is None else critic_steps
    if critic_steps <= 0 or num_episodes % (dp_size * critic_steps):
        raise ValueError("full_history episode count must be divisible by DP size times critic mini-batch steps")
    budget = args.max_tokens_per_gpu
    if budget is None or budget <= 0:
        raise ValueError("full_history requires a positive --max-tokens-per-gpu")
    for episode, indices in enumerate(siblings):
        if not indices:
            raise ValueError(f"full_history critic episode {episode} has no actor samples")
        if max(sum(critic_lengths[i] for i in critic_rows[episode]), sum(actor_lengths[i] for i in indices)) > budget:
            raise ValueError(f"full_history episode {episode} exceeds --max-tokens-per-gpu; increase the token budget")

    # The critic's episode ownership wins over independent actor length balancing:
    # all sibling values must be available locally for the existing value_slices.
    memory_critic = args.critic_context_mode in ("full_detach", "actor-summary")
    critic_batch_size = num_episodes // critic_steps if memory_critic else batch_size
    episode_order = (
        _memory_critic_episode_order(args, critic_data, critic_steps, critic_rows) if memory_critic else None
    )
    episode_lengths = [sum(critic_lengths[row] for row in rows) for rows in critic_rows]
    partitions = _full_history_episode_partitions(
        args, episode_lengths, batch_size=critic_batch_size, dp_size=dp_size, episode_order=episode_order
    )
    actor_partitions = [[i for episode in part for i in siblings[episode]] for part in partitions]
    critic_partitions = [[i for episode in part for i in critic_rows[episode]] for part in partitions]
    shards = _package_shards(args, {**data, "total_lengths": actor_lengths}, actor_partitions)
    critic_shards = _package_shards(args, {**critic_data, "total_lengths": critic_lengths}, critic_partitions)
    num_rollouts = [batch_size] * (num_episodes // batch_size)
    for actor, critic, episodes in zip(shards, critic_shards, partitions, strict=True):
        for shard, rows in ((actor, siblings), (critic, critic_rows)):
            shard["micro_batch_indices"] = _episode_microbatches(episodes, rows)
            shard["num_microbatches"] = [batch_size // dp_size] * len(num_rollouts)
            shard["num_rollouts"] = list(num_rollouts)
        if memory_critic:
            local_batch_size = critic_batch_size // dp_size
            critic["num_microbatches"] = [local_batch_size] * critic_steps
            critic["num_rollouts"] = [critic_batch_size] * critic_steps
    attach_value_slices(shards, critic_shards, spans)
    return shards, critic_shards


def _critic_episode_rows(data, critic_data, spans) -> tuple[list[list[int]], list[list[int]]]:
    """Map independent critic rows and actor value spans to whole logical episodes."""
    num_rows = len(critic_data["tokens"])
    rows_by_rollout: dict[int, list[int]] = {}
    for row, rollout_id in enumerate(critic_data.get("rollout_ids", range(num_rows))):
        rows_by_rollout.setdefault(rollout_id, []).append(row)
    critic_rows = list(rows_by_rollout.values())
    episode_of_row = {row: episode for episode, rows in enumerate(critic_rows) for row in rows}
    if len(spans) != len(data["tokens"]):
        raise ValueError("full_history requires exactly one critic span for each actor sample")
    siblings: list[list[int]] = [[] for _ in critic_rows]
    for index, (row, start, length) in enumerate(spans):
        if not 0 <= row < num_rows or start < 0 or length != data["response_lengths"][index]:
            raise ValueError(f"Invalid full_history critic span for actor sample {index}")
        if start + length > critic_data["response_lengths"][row]:
            raise ValueError(f"full_history critic span for actor sample {index} exceeds its episode")
        siblings[episode_of_row[row]].append(index)
    return critic_rows, siblings


def _episode_microbatches(episodes: list[int], rows: list[list[int]]) -> list[list[int]]:
    offset = 0
    microbatches = []
    for episode in episodes:
        count = len(rows[episode])
        microbatches.append(list(range(offset, offset + count)))
        offset += count
    return microbatches


def _memory_critic_episode_order(
    args, data: dict[str, Any], critic_steps: int, episode_rows: list[list[int]]
) -> list[int]:
    """Order complete prompt groups into equally sized critic optimizer batches."""
    num_episodes = len(episode_rows)
    if critic_steps == 1:
        return list(range(num_episodes))
    group_ids = data.get("prompt_group_ids")
    if group_ids is None or len(group_ids) != len(data["tokens"]):
        raise ValueError("Memory critic mini-batches require a prompt group identity for every episode")
    groups: dict[int, list[int]] = {}
    for episode, rows in enumerate(episode_rows):
        group_id = group_ids[rows[0]]
        if any(group_ids[row] != group_id for row in rows):
            raise ValueError("All segments of one critic episode must belong to the same prompt group")
        groups.setdefault(group_id, []).append(episode)
    if len(groups) % critic_steps:
        raise ValueError("Memory critic prompt group count must be divisible by critic mini-batch steps")
    expected_fanout = getattr(args, "n_samples_per_prompt", None)
    if expected_fanout is not None and any(len(group) != expected_fanout for group in groups.values()):
        raise ValueError("Memory critic mini-batches require complete n_samples_per_prompt response groups")
    groups_per_step = len(groups) // critic_steps
    ordered_groups = list(groups.values())
    minibatches = [
        [episode for group in ordered_groups[start : start + groups_per_step] for episode in group]
        for start in range(0, len(groups), groups_per_step)
    ]
    if any(len(batch) != num_episodes // critic_steps for batch in minibatches):
        raise ValueError("Memory critic prompt groups must form equally sized critic mini-batches")
    return [episode for batch in minibatches for episode in batch]


def _full_history_episode_partitions(
    args, lengths: list[int], *, batch_size: int, dp_size: int, episode_order: list[int] | None = None
) -> list[list[int]]:
    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    episode_order = list(range(len(lengths))) if episode_order is None else episode_order
    for start in range(0, len(lengths), batch_size):
        episodes = episode_order[start : start + batch_size]
        episode_lengths = [lengths[episode] for episode in episodes]
        if getattr(args, "balance_by_flops", False):
            # Use the same workload estimator as the standard rollout scheduler.
            costs = [calculate_fwd_flops([length], args) for length in episode_lengths]
        else:
            costs = episode_lengths
        if args.balance_data or getattr(args, "balance_by_flops", False):
            step = get_seqlen_balanced_partitions(costs, dp_size, equal_size=True)
        else:
            step = [list(range(rank, batch_size, dp_size)) for rank in range(dp_size)]
        for partition, indices in zip(partitions, step, strict=True):
            partition.extend(episodes[index] for index in indices)
    return partitions


def _memory_critic_sample(sample: Sample, siblings: list[Sample]):
    siblings = sorted(siblings, key=lambda s: s.metadata["segment"])
    if [s.metadata["segment"] for s in siblings] != list(range(len(siblings))):
        raise ValueError("Memory critic requires complete, consecutively numbered episode segments")
    if not all(getattr(s, "critic_memory_exact", False) for s in siblings):
        raise ValueError(
            "full_detach requires exact prefixes or explicit compaction boundaries; "
            "a template/retokenization break could repeat historical context"
        )
    raw_segments = getattr(sample, "critic_segments", None)
    if raw_segments is None:
        raise ValueError("full_detach requires explicit critic_segments from the rollout producer")
    segments = [Segment(**part) for part in raw_segments]
    if len(segments) != len(siblings):
        raise ValueError("Memory critic segments must match the actor's segment count")
    for actor, segment in zip(siblings, segments, strict=True):
        if (
            segment.response_length != actor.response_length
            or segment.tokens[segment.response_start :] != actor.tokens[-actor.response_length :]
            or segment.loss_mask != actor.loss_mask
        ):
            raise ValueError("Memory critic response tokens and masks must exactly match the actor")
    episode, spans = critic_episode(segments)
    own = copy.copy(sample)
    own.tokens, own.response_length, own.loss_mask = episode.tokens, episode.response_length, episode.loss_mask
    own._critic_segment_lengths = [len(segment.tokens) for segment in segments]
    return own, {id(s): span for s, span in zip(siblings, spans, strict=True)}


def _actor_summary_critic_samples(samples):
    """Keep reset-memory segments as separate native packed sequences."""
    siblings: dict[int, list[Sample]] = {}
    for sample in samples:
        if getattr(sample, "critic_tokens", None) is None:
            raise ValueError("Memory critic modes require episode-aware rollout samples")
        rollout = sample.rollout_id if sample.rollout_id is not None else sample.index
        siblings.setdefault(rollout, []).append(sample)
    for group in siblings.values():
        if sorted(sample.metadata["segment"] for sample in group) != list(range(len(group))):
            raise ValueError("Memory critic requires complete, consecutively numbered episode segments")
        if any(sample.group_index != group[0].group_index for sample in group):
            raise ValueError("All segments of one critic episode must belong to the same prompt group")
    critic = [_copy_critic_sample(sample) for sample in samples]
    return critic, [(index, 0, sample.response_length) for index, sample in enumerate(samples)]


def _copy_critic_sample(sample):
    own = copy.copy(sample)
    # Behavior-policy records describe actor tokens, not critic regression inputs.
    own.rollout_log_probs = None
    own.rollout_topk_token_ids = own.rollout_topk_log_probs = own.rollout_sampling_mask = None
    own.rollout_routed_experts = own.rollout_indexer_topk = None
    return own


def build_critic_samples(
    samples, *, context_mode: str = "full_history"
) -> tuple[list[Sample], list[tuple[int, int, int]]]:
    """The critic's own samples under --critic-context-mode full_history, and where each actor sample's values sit.

    A rollout function may attach an episode-level critic sample to the samples of one rollout: `critic_tokens`,
    `critic_response_length` and `critic_loss_mask` (the same on every sibling) and `critic_span = (start, length)`,
    the sibling's response region inside the critic sample's response region. Siblings share `rollout_id`. The
    critic then reads each episode once instead of once per sibling.

    Full-detach samples concatenate exact segments and carry their lengths for native detached memory.
    Actor-summary keeps each actor segment as an independent critic sequence, resetting attention and recurrent
    state at its packed boundary. Both modes retain whole episodes on the same DP rank.

    Returns critic samples and, per actor sample in batch order, `(critic sample index, start, length)`.
    """
    flat = [s for group in samples for s in group] if samples and isinstance(samples[0], list) else samples
    if context_mode == "actor-summary":
        return _actor_summary_critic_samples(flat)
    critic: list[Sample] = []
    spans: list[tuple[int, int, int]] = []
    index_of_rollout: dict = {}
    memory_spans: dict = {}
    sibling_positions: dict = {}
    if context_mode == "full_detach":
        for sample in flat:
            rollout = sample.rollout_id if sample.rollout_id is not None else sample.index
            sibling_positions.setdefault(rollout, []).append(sample)
    for sample in flat:
        tokens = getattr(sample, "critic_tokens", None)
        if tokens is None:
            if context_mode in ("full_detach", "actor-summary"):
                raise ValueError("Memory critic modes require episode-aware rollout samples")
            critic.append(copy.copy(sample))
            spans.append((len(critic) - 1, 0, sample.response_length))
            continue
        rollout = sample.rollout_id if sample.rollout_id is not None else sample.index
        if rollout not in index_of_rollout:
            own = _copy_critic_sample(sample)
            if context_mode == "full_detach":
                own, memory_spans[rollout] = _memory_critic_sample(own, sibling_positions[rollout])
            else:
                own.tokens = tokens
                own.response_length = sample.critic_response_length
                own.loss_mask = list(sample.critic_loss_mask)
            index_of_rollout[rollout] = len(critic)
            critic.append(own)
        own = critic[index_of_rollout[rollout]]
        if sample.group_index != own.group_index:
            raise ValueError("All segments of one critic episode must belong to the same prompt group")
        if sample.status == Sample.Status.TRUNCATED:
            own.status = Sample.Status.TRUNCATED
        start, length = memory_spans[rollout][id(sample)] if rollout in memory_spans else sample.critic_span
        region_start = len(own.tokens) - own.response_length
        assert length == sample.response_length, "critic_span must cover the sample's response region"
        assert (
            own.tokens[region_start + start : region_start + start + length]
            == sample.tokens[len(sample.tokens) - length :]
        ), "critic_span does not point at the sample's response tokens inside critic_tokens"
        spans.append((index_of_rollout[rollout], start, length))
    return critic, spans


def attach_value_slices(
    actor_shards: list[dict[str, Any]], critic_shards: list[dict[str, Any]], spans: list[tuple[int, int, int]]
) -> None:
    """Tell each critic shard how to cut its values into the per-sample vectors the actor's shard expects.

    Sets `value_slices` on every critic shard: for each sample of the actor's shard of the same rank, in order,
    `(local index of the critic sample, start, length)`. The critic sample must be on the same rank.
    """
    for actor, critic in zip(actor_shards, critic_shards, strict=True):
        local = {global_index: i for i, global_index in enumerate(critic["partition"])}
        slices = []
        for k in actor["partition"]:
            episode, start, length = spans[k]
            assert episode in local, (
                "--critic-context-mode full_history: a sample and its episode-level critic sample landed on "
                "different data-parallel ranks; use the joint full-history actor/critic shard planner"
            )
            slices.append((local[episode], start, length))
        critic["value_slices"] = slices


def can_schedule_on_rollout_side(args, data: dict[str, Any], train_parallel_config: dict | None) -> bool:
    """Whether the rollout side can precompute the full DP/mbs schedule."""
    if not has_full_schedule_config(train_parallel_config):
        return False
    if is_multi_lora_enabled(args):
        return False
    if "multimodal_train_inputs" in data:
        return False
    if "rollout_ids" not in data:
        return False
    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    return len(set(data["rollout_ids"])) >= global_batch_size


def split_train_data_by_dp_scheduled_raw(
    args, data: dict[str, Any], *, train_parallel_config: dict
) -> list[dict[str, Any]]:
    """DP split with the micro-batch schedule precomputed on the rollout side."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    partitions, micro_batch_indices, num_microbatches, num_rollouts = build_dp_schedule(
        args,
        train_parallel_config,
        total_lengths,
        global_batch_size=global_batch_size,
        rollout_indices=data["rollout_ids"],
    )
    logger.info(
        f"Rollout-side DP schedule: num_samples={len(total_lengths)}, "
        f"num_rollouts={num_rollouts}, num_microbatches={num_microbatches}"
    )

    shards = _package_shards(args, data, partitions)
    for rank, shard in enumerate(shards):
        shard["num_microbatches"] = num_microbatches
        shard["micro_batch_indices"] = micro_batch_indices[rank]
        shard["num_rollouts"] = num_rollouts
    return shards


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    # Multi-LoRA: sort partitions by adapter slot so each microbatch is
    # contiguous-by-slot (required by the per-adapter token-count math).
    adapter_slots = data.get("adapter_slots")
    if adapter_slots is not None:
        partitions = [sorted(p, key=lambda i: adapter_slots[i]) for p in partitions]

    return _package_shards(args, data, partitions)


def _package_shards(args, data: dict[str, Any], partitions) -> list[dict[str, Any]]:
    """Package one rollout_data shard per DP rank from precomputed partitions."""
    shards = []

    for i in range(len(partitions)):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "critic_loss_weights",
            "truncated",
            "loss_masks",
            "critic_segment_lengths",
            "round_number",
            "sample_indices",
            "prompt_group_ids",
            "rollout_ids",
            "rollout_mask_sums",
            "rollout_log_probs",
            "rollout_topk_token_ids",
            "rollout_topk_lengths",
            "rollout_topk_log_probs",
            "rollout_sampling_mask_ids",
            "rollout_sampling_mask_offsets",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "adapter_slots",
            "loss_weights",
            "advantages",
            "target_tokens",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
            "prompt_group_sizes",
            "loss_fn",
            "loss_fn_config",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        if "adapter_slots" in rollout_data:
            rollout_data["n_adapters"] = args.multi_lora_n_adapters
        shards.append(rollout_data)
    return shards


def process_rollout_data_shard(args, rollout_data):
    """Train-side completion of the DP split: drop the ``partition`` key and
    reorder the batch-global ``total_lengths`` into this shard's row order."""
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
