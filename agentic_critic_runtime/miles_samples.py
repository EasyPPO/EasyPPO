"""Critic-side training samples for Miles, for the Shared and Full settings.

The Full setting used in Miles is `critic_episode`: ONE critic sample per episode (the whole append-only log), whose
response region contains every segment's response region; `spans` say where each actor sample's values sit in it.
The critic then reads each token once. `critic_segments` below is the first version (one critic sample per segment,
each carrying the whole earlier history): it needs no mapping but re-reads the history once per later segment,
about S(S+1)/2 segments of tokens for S segments. It is kept for offline use and as the reference in tests.

---- first version ----
Critic-side training samples for Miles, one per actor segment, for the Shared and Full settings.

Miles reads values at the response region of each sample (the last ``response_length`` tokens) and pairs the
critic's values with the actor's samples by position. It has no way to hand the trainer a second token sequence
per sample (docs/code_map.md). So the decoupling is done with two data packs instead of extra fields:

  actor pack   one sample per segment, the actor's own context (what Miles already builds)
  critic pack  one sample per segment with the SAME response region and loss mask, and a different prompt:
                 shared  the segment as the actor saw it (identical to the actor pack)
                 full    everything that happened before the segment, then the segment

Because the response regions are identical token for token, the values the critic returns line up with the
actor's samples without any mapping, and returns, advantages and the actor loss are untouched.

The price of doing it per segment is that the Full critic re-reads the older segments once per later segment;
``critic_tokens`` reports the total so that every result can state it.
"""

from agentic_critic_runtime.critic_input import Segment

MODES = ("shared", "full")


def critic_segments(segments: list[Segment], mode: str, max_len: int | None = None) -> tuple[list[Segment], list[int]]:
    """Critic samples for one episode and, per sample, how many history tokens were cut to fit ``max_len``.

    The cut removes the oldest tokens of the history before the segment, never the segment itself: a cut is an
    implicit compaction of the critic's input and must be reported. A segment that alone exceeds ``max_len`` is
    returned whole (the actor trains on it at that length too).
    """
    assert mode in MODES
    out, cuts, history = [], [], []
    for seg in segments:
        prefix = history if mode == "full" else []
        cut = max(len(prefix) + len(seg.tokens) - max_len, 0) if max_len is not None else 0
        cut = min(cut, len(prefix))
        out.append(Segment(tokens=prefix[cut:] + seg.tokens, response_length=seg.response_length, loss_mask=list(seg.loss_mask)))
        cuts.append(cut)
        history = history + seg.tokens
    return out, cuts


def critic_tokens(segments: list[Segment]) -> int:
    """Tokens the critic processes for these samples in one pass."""
    return sum(len(s.tokens) for s in segments)


def critic_episode(segments: list[Segment]) -> tuple[Segment, list[tuple[int, int]]]:
    """One critic sample for a whole episode, and where each segment's values sit in it.

    The sample's tokens are the segments concatenated in time order. Its response region starts at the first
    segment's response region and runs to the end, with mask 1 exactly on the actor's tokens (later segments'
    prompts are masked like any other non-actor text). Returns the sample and, per segment, (start, length) of that
    segment's response region inside the sample's response region. Miles reads the value of a token at the position
    before it, so `values[start : start + length]` are the values of that segment's response tokens, each read
    with the whole earlier history in view.
    """
    tokens: list[int] = []
    absolute = []
    for seg in segments:
        offset = len(tokens)
        tokens.extend(seg.tokens)
        absolute.append((offset + seg.response_start, offset + len(seg.tokens)))
    first = absolute[0][0]
    mask = [0] * (len(tokens) - first)
    for seg, (a, b) in zip(segments, absolute, strict=True):
        mask[a - first : b - first] = seg.loss_mask
    return Segment(tokens=tokens, response_length=len(tokens) - first, loss_mask=mask), [(a - first, b - a) for a, b in absolute]
