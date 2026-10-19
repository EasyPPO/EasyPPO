import random

import pytest

from agentic_critic.critic_input import Segment, build_full, build_shared, split_values, value_mask


def make_episode(rng: random.Random, n_segments: int) -> list[Segment]:
    """Random linear episode. Token ids are globally unique so any misalignment is visible."""
    next_id = [0]

    def fresh(n: int) -> list[int]:
        ids = list(range(next_id[0], next_id[0] + n))
        next_id[0] += n
        return ids

    segments = []
    for s in range(n_segments):
        prompt = fresh(rng.randint(1, 6))  # task prompt (+ previous summary re-inserted as input)
        response: list[int] = []
        mask: list[int] = []
        for _ in range(rng.randint(1, 4)):  # turns: actor tokens then tool output
            a, o = rng.randint(1, 5), rng.randint(0, 4)
            response += fresh(a) + fresh(o)
            mask += [1] * a + [0] * o
        if s < n_segments - 1:  # compaction: instruction (env) then actor-written summary
            ins, summ = rng.randint(1, 3), rng.randint(1, 5)
            response += fresh(ins) + fresh(summ)
            mask += [0] * ins + [1] * summ
        segments.append(Segment(tokens=prompt + response, response_length=len(response), loss_mask=mask))
    return segments


def actor_tokens(segments: list[Segment]) -> list[int]:
    return [t for seg in segments for t, m in zip(seg.tokens[seg.response_start :], seg.loss_mask) if m]


@pytest.mark.parametrize("seed", range(20))
def test_zero_compaction_full_equals_shared(seed):
    segments = make_episode(random.Random(seed), n_segments=1)
    (shared,) = build_shared(segments)
    full = build_full(segments)
    assert full.tokens == shared.tokens
    assert full.response_spans == shared.response_spans
    assert value_mask(full, segments) == value_mask(shared, segments[:1])
    assert full.truncated_tokens == 0


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n_segments", [2, 3, 5])
def test_actor_input_is_a_slice_of_full_input(seed, n_segments):
    """Principle 2: every actor input can be computed from the critic input."""
    segments = make_episode(random.Random(seed), n_segments)
    full = build_full(segments)
    assert full.tokens == [t for seg in segments for t in seg.tokens]
    offset = 0
    for seg, (start, end) in zip(segments, full.response_spans):
        assert full.tokens[offset : offset + len(seg.tokens)] == seg.tokens
        assert full.tokens[start:end] == seg.tokens[seg.response_start :]
        offset += len(seg.tokens)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n_segments", [1, 2, 4])
def test_value_positions_are_exactly_the_actor_tokens(seed, n_segments):
    """Actor-generated tokens, summaries included, carry values; nothing else does."""
    segments = make_episode(random.Random(seed), n_segments)
    full = build_full(segments)
    mask = value_mask(full, segments)
    assert [t for t, m in zip(full.tokens, mask) if m] == actor_tokens(segments)
    shared_tokens = [
        t for ci, seg in zip(build_shared(segments), segments) for t, m in zip(ci.tokens, value_mask(ci, [seg])) if m
    ]
    assert shared_tokens == actor_tokens(segments)


@pytest.mark.parametrize("seed", range(20))
def test_split_values_round_trip(seed):
    """Using the token id as its own 'value' shows each value lands on the right state."""
    segments = make_episode(random.Random(seed), n_segments=3)
    full = build_full(segments)
    per_segment = split_values(full, full.tokens, segments)
    for seg, vals in zip(segments, per_segment):
        assert vals == seg.tokens[seg.response_start :]
    for ci, seg in zip(build_shared(segments), segments):
        assert split_values(ci, ci.tokens, [seg]) == [seg.tokens[seg.response_start :]]


@pytest.mark.parametrize("seed", range(20))
def test_truncation_is_recorded_and_alignment_survives(seed):
    rng = random.Random(seed)
    segments = make_episode(rng, n_segments=4)
    total = sum(len(s.tokens) for s in segments)
    max_len = rng.randint(1, total)
    full = build_full(segments, max_len=max_len)
    assert len(full.tokens) == max_len
    assert full.truncated_tokens == total - max_len
    assert full.tokens == [t for seg in segments for t in seg.tokens][total - max_len :]
    per_segment = split_values(full, full.tokens, segments)
    for seg, vals, dropped in zip(segments, per_segment, full.response_tokens_dropped):
        expected = seg.tokens[seg.response_start :]
        assert vals == [None] * dropped + expected[dropped:]
    mask = value_mask(full, segments)
    kept = set(full.tokens)
    assert [t for t, m in zip(full.tokens, mask) if m] == [t for t in actor_tokens(segments) if t in kept]


def test_no_truncation_when_within_limit():
    segments = make_episode(random.Random(0), n_segments=2)
    total = sum(len(s.tokens) for s in segments)
    assert build_full(segments, max_len=total) == build_full(segments)
