"""CPU tests of joint full-history placement and the unchanged shared path."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode
from miles.backends.training_utils.data.rollout import DataIterator
from miles.ray.rollout import train_data_conversion as conversion
from miles.utils.types import Sample


def _args(**overrides):
    return SimpleNamespace(
        **{
            "critic_context_mode": "full_history",
            "use_dynamic_batch_size": True,
            "global_batch_size": 8,
            "balance_data": False,
            "max_tokens_per_gpu": 1000,
            "critic_num_steps_per_rollout": 2,
            **overrides,
        }
    )


def _data(num_episodes=8):
    samples = []
    for episode in range(num_episodes):
        segments = [
            Segment(list(range(1000 * episode + 20 * i, 1000 * episode + 20 * i + 4 + i)), 2, [0, 1])
            for i in range(1 + episode % 3)
        ]
        critic, spans = critic_episode(segments)
        for segment, span in zip(segments, spans, strict=True):
            sample = Sample(
                index=len(samples),
                rollout_id=100 + episode,
                tokens=segment.tokens,
                response_length=2,
                loss_mask=segment.loss_mask,
                reward=episode / 10,
            )
            sample.critic_tokens = critic.tokens
            sample.critic_response_length = critic.response_length
            sample.critic_loss_mask = critic.loss_mask
            sample.critic_span = span
            samples.append(sample)
    # Noncontiguous siblings test lookup by episode rather than by row adjacency.
    samples = samples[::2] + samples[1::2]
    critic_samples, spans = conversion.build_critic_samples(samples)

    def rows(values):
        return {
            "tokens": [s.tokens for s in values],
            "response_lengths": [s.response_length for s in values],
            "loss_masks": [s.loss_mask for s in values],
            "rewards": [s.reward for s in values],
            "rollout_ids": [s.rollout_id for s in values],
            "sample_indices": [s.index for s in values],
        }

    return rows(samples), rows(critic_samples), spans


@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("balance", [False, True])
@pytest.mark.parametrize("num_episodes", [8, 16])
def test_every_segment_and_value_is_local_and_visited_once(dp_size, balance, num_episodes):
    actor, critic, spans = _data(num_episodes)
    original = deepcopy((actor, critic, spans))
    actors, critics = conversion.split_full_history_shards(
        _args(balance_data=balance), actor, critic, spans, {"dp_size": dp_size}
    )
    assert (actor, critic, spans) == original
    seen_actor, seen_critic = [], []
    for a, c in zip(actors, critics, strict=True):
        seen_actor.extend(a["partition"])
        seen_critic.extend(c["partition"])
        assert len(c["tokens"]) == num_episodes // dp_size
        assert set(a["rollout_ids"]) == set(c["rollout_ids"])
        assert a["num_microbatches"] == c["num_microbatches"] == [8 // dp_size] * (num_episodes // 8)
        assert a["num_rollouts"] == c["num_rollouts"] == [8] * (num_episodes // 8)
        # Each original field survives placement byte for byte, with no replicated/dropped samples.
        for shard, source in ((a, actor), (c, critic)):
            for key in ("tokens", "loss_masks", "rewards", "rollout_ids"):
                assert shard[key] == [source[key][i] for i in shard["partition"]]
            assert sorted(i for mb in shard["micro_batch_indices"] for i in mb) == list(range(len(shard["tokens"])))
        # Use token IDs as identifiable critic values and verify the actual outgoing slice order.
        values = [tokens[-length:] for tokens, length in zip(c["tokens"], c["response_lengths"], strict=True)]
        shipped = [values[i][start : start + length] for i, start, length in c["value_slices"]]
        assert shipped == [tokens[-length:] for tokens, length in zip(a["tokens"], a["response_lengths"], strict=True)]
        iterator = DataIterator(a, micro_batch_indices=a["micro_batch_indices"])
        for _ in range(sum(a["num_microbatches"])):
            micro_batch = iterator.get_next(["rollout_ids", "tokens"])
            assert len(set(micro_batch["rollout_ids"])) == 1
            assert sum(map(len, micro_batch["tokens"])) <= 1000
    assert sorted(seen_actor) == list(range(len(actor["tokens"])))
    assert sorted(seen_critic) == list(range(num_episodes))


def test_regression_independent_splits_fail_but_joint_split_succeeds():
    actor, critic, spans = _data()
    args = _args()
    actors = conversion.split_train_data_by_dp_raw(args, deepcopy(actor), dp_size=2)
    critics = conversion.split_train_data_by_dp_raw(args, deepcopy(critic), dp_size=2)
    with pytest.raises(AssertionError, match="different data-parallel ranks"):
        conversion.attach_value_slices(actors, critics, spans)
    conversion.split_full_history_shards(args, actor, critic, spans, {"dp_size": 2})


@pytest.mark.parametrize("balance", [False, True])
def test_shared_critic_steps_do_not_enable_episode_aligned_sharding(monkeypatch, balance):
    args = _args(critic_context_mode="shared", balance_data=balance, critic_num_steps_per_rollout=4)
    actor, _, _ = _data()
    # The legacy balanced path requires an even number of sample rows.
    actor = {key: values[:-1] for key, values in actor.items()}
    monkeypatch.setattr(conversion, "can_schedule_on_rollout_side", lambda *a: False)
    expected = conversion.split_train_data_by_dp_raw(args, deepcopy(actor), dp_size=2)
    actual = conversion.split_train_data_by_dp_shards(args, deepcopy(actor), {"dp_size": 2})
    assert actual == expected
    assert all("micro_batch_indices" not in shard for shard in actual)


def test_shared_rollout_schedule_remains_selected(monkeypatch):
    args = _args(critic_context_mode="shared", critic_num_steps_per_rollout=4)
    actor, _, _ = _data()
    sentinel = [object()]
    monkeypatch.setattr(conversion, "can_schedule_on_rollout_side", lambda *a: True)
    monkeypatch.setattr(conversion, "split_train_data_by_dp_scheduled_raw", lambda *a, **kw: sentinel)
    assert conversion.split_train_data_by_dp_shards(args, actor, {"dp_size": 2}) is sentinel


@pytest.mark.parametrize(
    "overrides,parallel,match",
    [
        ({"critic_context_mode": "shared"}, {}, "requires.*full_history"),
        ({"global_batch_size": 3}, {"dp_size": 2}, "divisible"),
        ({"global_batch_size": 6}, {}, "complete logical"),
        ({"critic_num_steps_per_rollout": 3}, {}, "critic mini-batch"),
        ({"use_dynamic_batch_size": False}, {}, "dynamic-batch-size"),
        ({"max_tokens_per_gpu": 3}, {}, "episode.*exceeds"),
        ({}, {"cp_size": 2}, "context parallel"),
        ({}, {"vpp_size": 2}, "virtual pipeline"),
    ],
)
def test_unsupported_layouts_fail_explicitly(overrides, parallel, match):
    actor, critic, spans = _data()
    with pytest.raises(ValueError, match=match):
        conversion.split_full_history_shards(_args(**overrides), actor, critic, spans, {"dp_size": 2, **parallel})


def test_invalid_span_fails_before_publishing_shards():
    actor, critic, spans = _data()
    spans[0] = (500, 0, 2)
    with pytest.raises(ValueError, match="Invalid full_history critic span"):
        conversion.split_full_history_shards(_args(), actor, critic, spans, {"dp_size": 2})


def test_flops_balance_uses_episode_costs(monkeypatch):
    actor, critic, spans = _data()
    lengths_seen = []

    def estimate(lengths, args):
        lengths_seen.extend(lengths)
        return sum(x * x for x in lengths)

    monkeypatch.setattr(conversion, "calculate_fwd_flops", estimate)
    conversion.split_full_history_shards(_args(balance_by_flops=True), actor, critic, spans, {"dp_size": 2})
    assert lengths_seen == list(map(len, critic["tokens"]))
