"""Build frozen validation packs without advancing the training data source."""

import copy

from miles.ray.rollout.debug_data import load_debug_rollout_data
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    build_critic_samples,
    convert_samples_to_train_data,
    split_episode_aligned_shards,
    split_full_history_shards,
)
from miles.utils import object_store
from miles.utils.data import RolloutDataPack


def validation_pack(args, *, replay_path, group_size, parallel):
    # Evaluation dimensions belong to this fixed dataset, not the training batch.
    eval_args = copy.copy(args)
    eval_args.load_debug_rollout_data = replay_path
    eval_args.load_debug_rollout_data_subsample = None
    eval_args.global_batch_size = group_size
    eval_args.rollout_batch_size = 1
    eval_args.n_samples_per_prompt = group_size
    eval_args.filter_truncated_in_actor_loss = False
    samples, _ = load_debug_rollout_data(eval_args, 0)
    if len({s.rollout_id for s in samples}) != group_size:
        raise ValueError("Frozen validation replay has incorrect episode count")

    def convert(rows):
        return convert_samples_to_train_data(
            eval_args,
            rows,
            metadata={},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )

    data = convert(samples)
    if args.critic_context_mode == "full_history":
        own, spans = build_critic_samples(samples)
        _, shards = split_full_history_shards(eval_args, data, convert(own), spans, parallel)
    else:
        shards = split_episode_aligned_shards(eval_args, data, dp_size=parallel["dp_size"])
    # Training keeps raw_reward batch-global for logging. Validation needs the
    # exact original targets in local row order before reducing predictions.
    for shard in shards:
        shard["raw_reward"] = [shard["raw_reward"][i] for i in shard["partition"]]
    store, refs = object_store.get_instance(), []
    try:
        for shard in shards:
            refs.append(store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC))
        return RolloutDataPack(sample_indices=data["sample_indices"], data_ref=refs)
    except Exception:
        for ref in refs:
            store.remove(ref)
        raise
