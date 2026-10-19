"""Critic samples and value-position mappings for shared and full-history contexts.

`critic_episode` builds one complete episode sample and the response spans needed
by Miles to return values to the actor's segment samples. `critic_segments` builds
per-segment inputs with optional historical prefixes and explicit truncation counts.
"""

from agentic_critic.critic_input import Segment

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
