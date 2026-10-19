"""The critic's own samples (--critic-context-mode full_history): one per rollout, values cut back per actor sample."""

from types import SimpleNamespace

import pytest

from miles.ray.rollout.train_data_conversion import (
    attach_value_slices,
    build_critic_samples,
    split_train_data_by_dp_raw,
)
from miles.utils.data import RolloutDataPack
from miles.utils.types import Sample


def _episode(rollout_id: int, base: int):
    """Two segments of one episode: prompt p0, response r0 | prompt p1, response r1."""
    p0, r0, p1, r1 = [base + 1, base + 2], [base + 3, base + 4, base + 5], [base + 6], [base + 7, base + 8]
    log = p0 + r0 + p1 + r1
    region_mask = [1, 0, 1] + [0] + [1, 1]  # r0 has one environment token; p1 is not the actor's
    common = dict(critic_tokens=log, critic_response_length=len(log) - len(p0), critic_loss_mask=region_mask)
    first = Sample(index=rollout_id * 10, rollout_id=rollout_id, tokens=p0 + r0, response_length=3, loss_mask=[1, 0, 1], reward=0.5)
    second = Sample(index=rollout_id * 10 + 1, rollout_id=rollout_id, tokens=p1 + r1, response_length=2, loss_mask=[1, 1], reward=0.5)
    for sample, span in ((first, (0, 3)), (second, (4, 2))):
        for key, value in common.items():
            setattr(sample, key, value)
        sample.critic_span = span
        sample.rollout_log_probs = [-1.0] * sample.response_length
    return first, second, log


def test_one_critic_sample_per_rollout_and_spans_point_at_the_actor_tokens():
    a1, a2, log_a = _episode(1, 100)
    b1, b2, log_b = _episode(2, 200)
    plain = Sample(index=99, tokens=[1, 2, 3], response_length=2, loss_mask=[1, 1], reward=0.1)
    critic, spans = build_critic_samples([a1, a2, plain, b1, b2])
    assert [c.tokens for c in critic] == [log_a, [1, 2, 3], log_b]
    assert spans == [(0, 0, 3), (0, 4, 2), (1, 0, 2), (2, 0, 3), (2, 4, 2)]
    episode = critic[0]
    assert episode.response_length == len(log_a) - 2 and episode.loss_mask == [1, 0, 1, 0, 1, 1]
    assert episode.rollout_log_probs is None and episode.reward == 0.5 and episode.rollout_id == 1
    episode.validate()
    # every token of an episode appears once in the critic's samples
    assert sum(len(c.tokens) for c in critic) == len(log_a) + 3 + len(log_b)
    # the actor's samples are not touched
    assert a1.tokens == log_a[:5] and a1.rollout_log_probs == [-1.0] * 3


def test_truncation_of_any_sibling_marks_the_critic_sample():
    a1, a2, _ = _episode(1, 100)
    a2.status = Sample.Status.TRUNCATED
    critic, _ = build_critic_samples([[a1, a2]])
    assert critic[0].status == Sample.Status.TRUNCATED


def test_a_wrong_span_is_rejected():
    a1, a2, _ = _episode(1, 100)
    a2.critic_span = (3, 2)
    with pytest.raises(AssertionError, match="does not point at"):
        build_critic_samples([a1, a2])


def test_value_slices_cut_the_critic_values_into_the_actor_samples():
    a1, a2, log_a = _episode(1, 100)
    b1, b2, log_b = _episode(2, 200)
    actor_samples = [a1, a2, b1, b2]
    critic, spans = build_critic_samples(actor_samples)
    args = SimpleNamespace(balance_data=False)
    actor_data = {"tokens": [s.tokens for s in actor_samples], "response_lengths": [s.response_length for s in actor_samples]}
    critic_data = {"tokens": [c.tokens for c in critic], "response_lengths": [c.response_length for c in critic]}
    actor_shards = split_train_data_by_dp_raw(args, actor_data, dp_size=1)
    critic_shards = split_train_data_by_dp_raw(args, critic_data, dp_size=1)
    attach_value_slices(actor_shards, critic_shards, spans)
    slices = critic_shards[0]["value_slices"]
    # the critic returns one value per position of its response region; use the token id as a stand-in value
    values = [c.tokens[len(c.tokens) - c.response_length :] for c in critic]
    shipped = [values[i][start : start + length] for i, start, length in slices]
    assert shipped == [s.tokens[len(s.tokens) - s.response_length :] for s in actor_samples]


def test_samples_on_another_rank_than_their_episode_are_refused():
    a1, a2, _ = _episode(1, 100)
    b1, b2, _ = _episode(2, 200)
    actor_samples = [a1, b1, a2, b2]  # round-robin over two ranks puts a1, a2 on rank 0 and b1, b2 on rank 1
    critic, spans = build_critic_samples(actor_samples)
    args = SimpleNamespace(balance_data=False)
    actor_shards = split_train_data_by_dp_raw(args, {"tokens": [s.tokens for s in actor_samples]}, dp_size=2)
    critic_shards = split_train_data_by_dp_raw(args, {"tokens": [c.tokens for c in critic]}, dp_size=2)
    attach_value_slices(actor_shards, critic_shards, spans)  # episodes happen to sit with their samples here
    swapped = [critic_shards[1], critic_shards[0]]
    with pytest.raises(AssertionError, match="different data-parallel ranks"):
        attach_value_slices(actor_shards, swapped, spans)


def test_pack_for_critic():
    shared = RolloutDataPack(sample_indices=[0], data_ref=None)
    assert shared.for_critic() is shared and shared.data_refs == []
