import random

import pytest

from agentic_critic.critic_input import Segment, build_full
from agentic_critic.miles_samples import critic_episode, critic_segments, critic_tokens


def episode(rng: random.Random, n_segments: int) -> list[Segment]:
    segs = []
    for _ in range(n_segments):
        prompt = [rng.randrange(1000) for _ in range(rng.randrange(3, 9))]
        mask = [rng.randrange(2) for _ in range(rng.randrange(4, 12))]
        mask[0] = 1  # the response region starts at the first actor token
        segs.append(Segment(tokens=prompt + [rng.randrange(1000) for _ in mask], response_length=len(mask), loss_mask=mask))
    return segs


@pytest.mark.parametrize("seed", range(20))
def test_response_regions_are_identical_in_both_modes(seed):
    rng = random.Random(seed)
    actor = episode(rng, rng.randrange(1, 5))
    shared, _ = critic_segments(actor, "shared")
    full, cuts = critic_segments(actor, "full")
    assert shared == actor and cuts == [0] * len(actor)
    for a, f in zip(actor, full, strict=True):
        assert f.response_length == a.response_length and f.loss_mask == a.loss_mask
        assert f.tokens[-len(a.tokens) :] == a.tokens  # the actor input is the tail of the critic input


def test_no_compaction_gives_the_same_sample_in_both_modes():
    actor = episode(random.Random(0), 1)
    assert critic_segments(actor, "full")[0] == critic_segments(actor, "shared")[0] == actor


@pytest.mark.parametrize("seed", range(10))
def test_full_prompt_is_the_append_only_log_so_far(seed):
    rng = random.Random(seed)
    actor = episode(rng, 4)
    full, _ = critic_segments(actor, "full")
    log = build_full(actor)
    for s, f in enumerate(full):
        end = log.response_spans[s][1]
        assert f.tokens == log.tokens[:end]
        # Miles reads the value of response token t at the position before it; that position holds the same
        # token in the per-segment sample and in the whole log.
        start = len(f.tokens) - f.response_length
        assert f.tokens[start - 1 : len(f.tokens) - 1] == log.tokens[log.response_spans[s][0] - 1 : end - 1]


def test_cut_removes_oldest_history_only_and_is_reported():
    actor = episode(random.Random(1), 3)
    longest = max(len(s.tokens) for s in actor)
    full, cuts = critic_segments(actor, "full", max_len=longest + 5)
    uncut, _ = critic_segments(actor, "full")
    assert cuts[0] == 0 and any(cuts)
    for a, f, u, cut in zip(actor, full, uncut, cuts, strict=True):
        assert f.tokens == u.tokens[cut:] and len(f.tokens) <= max(longest + 5, len(a.tokens))
        assert f.tokens[-len(a.tokens) :] == a.tokens and f.loss_mask == a.loss_mask


def test_a_segment_longer_than_the_limit_is_kept_whole():
    actor = episode(random.Random(2), 2)
    full, cuts = critic_segments(actor, "full", max_len=1)
    assert [f.tokens for f in full] == [a.tokens for a in actor]
    assert cuts == [0, len(actor[0].tokens)]


def test_token_count_grows_with_the_history():
    actor = episode(random.Random(3), 3)
    n = [len(a.tokens) for a in actor]
    assert critic_tokens(critic_segments(actor, "shared")[0]) == sum(n)
    assert critic_tokens(critic_segments(actor, "full")[0]) == 3 * n[0] + 2 * n[1] + n[2]


@pytest.mark.parametrize("seed", range(20))
def test_episode_sample_holds_every_segment_once(seed):
    rng = random.Random(seed)
    actor = episode(rng, rng.randrange(1, 5))
    sample, spans = critic_episode(actor)
    assert sample.tokens == [t for seg in actor for t in seg.tokens]  # every token once
    region = sample.tokens[len(sample.tokens) - sample.response_length :]
    per_segment, _ = critic_segments(actor, "full")
    for seg, full, (start, length) in zip(actor, per_segment, spans, strict=True):
        assert length == seg.response_length
        assert region[start : start + length] == seg.tokens[-length:]
        assert sample.loss_mask[start : start + length] == seg.loss_mask
        # the position a value is read from (one before the token) has the same history as in the per-segment version
        absolute = len(sample.tokens) - sample.response_length + start
        assert sample.tokens[:absolute] == full.tokens[: len(full.tokens) - length]
    # nothing outside the segments' response regions carries a value target
    covered = [0] * sample.response_length
    for start, length in spans:
        covered[start : start + length] = [1] * length
    assert all(c or not m for c, m in zip(covered, sample.loss_mask))
    assert sum(sample.loss_mask) == sum(sum(seg.loss_mask) for seg in actor)


def test_episode_sample_without_compaction_is_the_actor_sample():
    actor = episode(random.Random(0), 1)
    sample, spans = critic_episode(actor)
    assert sample == actor[0] and spans == [(0, actor[0].response_length)]
