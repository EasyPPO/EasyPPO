"""Real executor conversion, episode packing and train-side critic schedules."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode
from tests.fast.fixtures.driver_fakes import FakeObjectStore
from tests.fast.fixtures.score_centering_fixtures import _args as _conversion_args

from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.loss.hub import losses as native_losses
from miles.backends.training_utils.loss.objective import drop_truncated_samples_from_actor_loss
from miles.ray.rollout import rollout_executor as executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.rollout.train_data_conversion import build_critic_samples, split_full_history_shards
from miles.utils import object_store
from miles.utils.data import process_rollout_data
from miles.utils.types import Sample


def _samples():
    samples = []
    rewards = []
    for group in range(32):
        group_rewards = []
        for response in range(16):
            episode_index = group * 16 + response
            reward = 0.5 if group == 0 else (group + 1) / 32 if response >= 8 else 0.0
            group_rewards.append(reward)
            # High rewards have more segments, exposing reward-variance bugs that
            # accidentally count a compacted response as several responses.
            segments = [
                Segment([1, 2] * (segment + 1) + [10000 + episode_index * 10 + segment, 9], 2, [0, 1])
                for segment in range(3 if response >= 8 else 1)
            ]
            memory = [segments[0]] + [Segment([90] + part.tokens[-2:], 2, [0, 1]) for part in segments[1:]]
            whole, spans = critic_episode(segments)
            for segment, (part, span) in enumerate(zip(segments, spans, strict=True)):
                sample = Sample(
                    index=episode_index,
                    group_index=100 + group * 7,
                    rollout_id=1000 + episode_index,
                    tokens=list(part.tokens),
                    response_length=2,
                    loss_mask=list(part.loss_mask),
                    reward=reward,
                    status=Sample.Status.TRUNCATED if response == 15 else Sample.Status.COMPLETED,
                    metadata={"segment": segment},
                )
                sample.critic_tokens = whole.tokens
                sample.critic_response_length = whole.response_length
                sample.critic_loss_mask = whole.loss_mask
                sample.critic_span = span
                sample.critic_segments = [asdict(part) for part in memory]
                sample.critic_memory_exact = True
                samples.append(sample)
        rewards.append(group_rewards)
    samples.sort(key=lambda sample: (sample.metadata["segment"], sample.index % 16, sample.group_index))
    raw = torch.tensor(rewards, dtype=torch.float32).var(dim=1, correction=0).sqrt().clamp_min(0.075).reciprocal()
    weights = (raw / raw.mean()).tolist()
    return samples, {100 + group * 7: weight for group, weight in enumerate(weights)}


def _executor(monkeypatch, *, mode, dp_size, balance, samples, num_groups=32, fanout=16, critic_steps=4):
    args = _conversion_args(
        advantage_estimator="ppo",
        critic_context_mode=mode,
        rollout_top_logprobs_num=0,
        use_dynamic_batch_size=True,
        use_dynamic_global_batch_size=False,
        global_batch_size=num_groups * fanout,
        rollout_batch_size=num_groups,
        n_samples_per_prompt=fanout,
        max_tokens_per_gpu=1000,
        critic_num_steps_per_rollout=critic_steps,
        critic_return_std_floor=0.075,
        filter_truncated_in_actor_loss=True,
        balance_data=balance,
        delay_split_train_data_by_dp=False,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        update_weights_interval=1,
        use_critic=True,
        num_critic_only_steps=0,
        lora_rank=0,
    )
    executor = object.__new__(RolloutExecutor)
    executor.args = args
    executor.data_source = SimpleNamespace()
    executor._rollouts_since_publish_of_model_id = defaultdict(int)
    executor._train_parallel_configs_of_model_id = {None: {"dp_size": dp_size}}
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._get_rollout_data = AsyncMock(return_value=(samples, {}, {}))
    monkeypatch.setattr(object_store, "_INSTANCE", FakeObjectStore())
    monkeypatch.setattr(executor_module, "save_debug_rollout_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor_module, "log_rollout_data", lambda *args, **kwargs: None)
    return executor


def _critic_steps(monkeypatch, args, critic):
    """Drain the real native iterator with the producer's optimizer-step boundaries."""
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: SimpleNamespace(vpp_size=1))
    data = {**critic, "row_indices": list(range(len(critic["tokens"])))}
    [iterator], counts, rollouts = data_utils.get_critic_train_schedule(
        args, data, rollout_id=0, num_rollouts=critic["num_rollouts"]
    )
    assert iterator.micro_batch_indices is critic["micro_batch_indices"]
    assert counts == critic["num_microbatches"] and rollouts == critic["num_rollouts"]
    return [
        [row for _ in range(count) for row in iterator.get_next(["row_indices"])["row_indices"]] for count in counts
    ]


def _step_groups(critic, indices):
    # Actor-summary keeps several independent rows for one response. Count each
    # logical response only once when checking complete global prompt groups.
    groups = {critic["rollout_ids"][index]: critic["prompt_group_ids"][index] for index in indices}
    return list(groups.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("balance", [False, True])
async def test_group_variance_four_steps_and_actor_filter_survive_executor_pack_and_process(
    monkeypatch, mode, dp_size, balance
):
    samples, expected_weights = _samples()
    executor = _executor(monkeypatch, mode=mode, dp_size=dp_size, balance=balance, samples=samples)
    pack = await executor.get(0)
    groups_per_step = [Counter() for _ in range(4)]
    seen_episodes = []
    for rank in range(dp_size):
        actor, actor_result = process_rollout_data(executor.args, pack.data_ref, rank, dp_size, witness_info=None)
        critic, critic_result = process_rollout_data(
            executor.args, pack.for_critic().data_ref, rank, dp_size, witness_info=None
        )
        with actor_result, critic_result:
            assert actor["num_rollouts"] == [512]
            assert actor["num_microbatches"] == [512 // dp_size]
            assert "critic_minibatch_indices" not in actor
            assert critic["num_rollouts"] == [128] * 4
            assert critic["num_microbatches"] == [128 // dp_size] * 4
            assert "critic_minibatch_indices" not in critic
            schedule = _critic_steps(monkeypatch, executor.args, critic)
            assert [len(_step_groups(critic, batch)) for batch in schedule] == [128 // dp_size] * 4
            assert sorted(index for batch in schedule for index in batch) == list(range(len(critic["tokens"])))
            assert set(actor["rollout_ids"]) == set(critic["rollout_ids"])
            seen_episodes.extend(set(critic["rollout_ids"]))
            for step, indices in enumerate(schedule):
                groups_per_step[step].update(_step_groups(critic, indices))
            for shard in (actor, critic):
                assert shard["critic_loss_weights"] == pytest.approx(
                    [expected_weights[group] for group in shard["prompt_group_ids"]]
                )
            responses = [
                tokens[-length:] for tokens, length in zip(critic["tokens"], critic["response_lengths"], strict=True)
            ]
            shipped = [responses[index][start : start + length] for index, start, length in critic["value_slices"]]
            assert shipped == [
                tokens[-length:] for tokens, length in zip(actor["tokens"], actor["response_lengths"], strict=True)
            ]
            assert all(critic["loss_masks"][index] for index, truncated in enumerate(critic["truncated"]) if truncated)
            # The actor's late mask filter must neither remove these critic
            # episodes nor change the variance weights computed before sharding.
            original_critic_masks = deepcopy(critic["loss_masks"])
            actor["loss_masks"] = [torch.tensor(mask) for mask in actor["loss_masks"]]
            drop_truncated_samples_from_actor_loss(actor)
            assert all(
                not mask.any()
                for mask, truncated in zip(actor["loss_masks"], actor["truncated"], strict=True)
                if truncated
            )
            assert critic["loss_masks"] == original_critic_masks
            # TP siblings request the same DP shard through this real processing
            # boundary; they must see the same episode order and step schedule.
            sibling, sibling_result = process_rollout_data(
                executor.args, pack.for_critic().data_ref, rank, dp_size, witness_info=None
            )
            with sibling_result:
                assert sibling["rollout_ids"] == critic["rollout_ids"]
                assert sibling["micro_batch_indices"] == critic["micro_batch_indices"]
    assert sorted(seen_episodes) == list(range(1000, 1512))
    assert [len(groups) for groups in groups_per_step] == [8] * 4
    assert all(set(groups.values()) == {16} for groups in groups_per_step)
    assert len(set.union(*(set(groups) for groups in groups_per_step))) == 32
    assert executor.args.filter_truncated_in_actor_loss  # Critic args were copied, not mutated.


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("balance", [False, True])
async def test_prompt_groups_can_span_dp_ranks_with_equal_episode_batches(monkeypatch, mode, balance):
    # Twelve prompt groups need not divide DP=8: 192 actor episodes and each
    # 48-episode critic mini-batch do. Each step retains three complete groups.
    samples, _ = _samples()
    samples = [sample for sample in samples if sample.group_index < 100 + 12 * 7]
    executor = _executor(monkeypatch, mode=mode, dp_size=8, balance=balance, samples=samples, num_groups=12)
    pack = await executor.get(0)
    groups_per_step = [Counter() for _ in range(4)]
    seen = []
    for rank in range(8):
        actor, actor_result = process_rollout_data(executor.args, pack.data_ref, rank, 8, witness_info=None)
        critic, critic_result = process_rollout_data(
            executor.args, pack.for_critic().data_ref, rank, 8, witness_info=None
        )
        with actor_result, critic_result:
            assert actor["num_rollouts"] == [192]
            assert actor["num_microbatches"] == [24]
            assert critic["num_rollouts"] == [48] * 4
            assert critic["num_microbatches"] == [6] * 4
            seen.extend(set(critic["rollout_ids"]))
            for step, indices in enumerate(_critic_steps(monkeypatch, executor.args, critic)):
                assert len(_step_groups(critic, indices)) == 6
                groups_per_step[step].update(_step_groups(critic, indices))
    assert len(seen) == len(set(seen)) == 192
    assert [len(groups) for groups in groups_per_step] == [3] * 4
    assert all(set(groups.values()) == {16} for groups in groups_per_step)
    assert len(set.union(*(set(groups) for groups in groups_per_step))) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_native_value_loss_consumes_global_weights_with_each_steps_token_mean(monkeypatch, mode):
    samples, _ = _samples()
    groups = {100 + group * 7 for group in (0, 8, 16, 24)}
    samples = [sample for sample in samples if sample.group_index in groups and sample.index % 16 in (0, 15)]
    executor = _executor(
        monkeypatch, mode=mode, dp_size=1, balance=True, samples=samples, num_groups=4, fanout=2, critic_steps=2
    )
    pack = await executor.get(0)
    critic, result = process_rollout_data(executor.args, pack.for_critic().data_ref, 0, 1, witness_info=None)
    with result:
        assert max(critic["critic_loss_weights"]) > 2 * min(critic["critic_loss_weights"])
        steps = _critic_steps(monkeypatch, executor.args, critic)
        args = SimpleNamespace(critic_return_std_floor=0.075, value_clip=0.2, value_loss_coef=0.5)
        for indices in steps:
            lengths = [critic["response_lengths"][index] for index in indices]
            masks = torch.cat([torch.tensor(critic["loss_masks"][index], dtype=torch.bool) for index in indices])
            weights = torch.tensor([critic["critic_loss_weights"][index] for index in indices])
            rewards = torch.tensor([critic["rewards"][index] for index in indices])
            count = masks.sum()
            # Distinct rows exercise weight expansion over unequal response
            # lengths, while large deltas activate the original clipped loss.
            old = [torch.full((length,), 0.4) for length in lengths]
            current = [torch.full((length,), 0.9, requires_grad=True) for length in lengths]
            batch = {
                "values": old,
                "returns": [reward.expand(length) for reward, length in zip(rewards, lengths, strict=True)],
                "critic_loss_weights": weights,
                "unconcat_tokens": [critic["tokens"][index] for index in indices],
                "total_lengths": [critic["total_lengths"][index] for index in indices],
                "response_lengths": lengths,
            }
            monkeypatch.setattr(native_losses, "get_values", lambda *a, current=current, **k: {"values": current})
            loss, _ = native_losses.value_loss_function(
                args, batch, torch.empty(0), lambda errors, masks=masks, count=count: errors[masks].sum() / count
            )
            # With old=.4 and current=.9, clipping at .2 gives .6. Compute
            # per-row expected loss independently of native weight expansion.
            expected = (
                sum(
                    weight
                    * max(float((0.9 - reward) ** 2), float((0.6 - reward) ** 2))
                    * sum(critic["loss_masks"][index])
                    for weight, reward, index in zip(weights, rewards, indices, strict=True)
                )
                * 0.5
                / count
            )
            torch.testing.assert_close(loss.detach(), expected)
            loss.backward()
            assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in current)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_all_truncated_warmup_preserves_all_critic_episodes_for_training(monkeypatch, mode):
    samples, _ = _samples()
    for sample in samples:
        sample.status = Sample.Status.TRUNCATED
    executor = _executor(monkeypatch, mode=mode, dp_size=4, balance=False, samples=samples)
    executor.args.num_critic_only_steps = 30
    pack = await executor.get(29)
    seen = []
    for rank in range(4):
        critic, result = process_rollout_data(executor.args, pack.for_critic().data_ref, rank, 4, witness_info=None)
        with result:
            assert all(critic["truncated"])
            assert all(any(mask) for mask in critic["loss_masks"])
            assert [
                len(_step_groups(critic, batch)) for batch in _critic_steps(monkeypatch, executor.args, critic)
            ] == [32] * 4
            assert all(weight > 0 for weight in critic["critic_loss_weights"])
            seen.extend(set(critic["rollout_ids"]))
    assert sorted(seen) == list(range(1000, 1512))
    assert executor.args.filter_truncated_in_actor_loss  # Warmup cannot change future actor conversions.


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_all_truncated_batch_still_refuses_empty_actor_update_after_warmup(monkeypatch, mode):
    samples, _ = _samples()
    for sample in samples:
        sample.status = Sample.Status.TRUNCATED
    executor = _executor(monkeypatch, mode=mode, dp_size=4, balance=False, samples=samples)
    executor.args.num_critic_only_steps = 30
    with pytest.raises(ValueError, match="removed every training sample"):
        await executor.get(30)


@pytest.mark.parametrize(
    "groups,fanout,message",
    [
        (None, 2, "prompt group identity"),
        ([0, 0, 1, 1, 2, 2, 2, 2], 2, "group count"),
        ([0, 1, 1, 1, 2, 2, 3, 3], 2, "complete n_samples_per_prompt"),
        ([0, 1, 1, 1, 2, 2, 3, 3], None, "equally sized"),
    ],
)
def test_invalid_prompt_group_layouts_fail_before_shards_are_published(groups, fanout, message):
    data = {"tokens": [[1, 2]] * 8, "response_lengths": [1] * 8}
    critic = dict(data)
    if groups is not None:
        critic["prompt_group_ids"] = groups
    args = SimpleNamespace(
        critic_context_mode="full_detach",
        use_dynamic_batch_size=True,
        global_batch_size=8,
        max_tokens_per_gpu=100,
        critic_num_steps_per_rollout=4,
        n_samples_per_prompt=fanout,
        balance_data=False,
    )
    with pytest.raises(ValueError, match=message):
        split_full_history_shards(args, data, critic, [(episode, 0, 1) for episode in range(8)], {"dp_size": 2})


def test_one_episode_cannot_mix_prompt_group_identities():
    samples, _ = _samples()
    same_episode = [sample for sample in samples if sample.rollout_id == 1031]
    same_episode[-1].group_index += 1
    with pytest.raises(ValueError, match="same prompt group"):
        build_critic_samples(same_episode, context_mode="full_detach")
