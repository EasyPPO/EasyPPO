"""Critic-input construction for the Shared and Full settings.

An episode is a single linear trajectory: decode -> compaction -> decode -> ...
Each actor compaction ends one *segment* and starts the next. A segment is exactly
what the actor is trained on (one Miles ``Sample``): ``tokens`` is the actor-visible
context, whose last ``response_length`` tokens form the response region, and
``loss_mask`` (length ``response_length``) is 1 on actor-generated tokens (including
an actor-written summary) and 0 on environment / tool tokens.

Shared: the critic sees each segment as is.
Full:   the critic sees the segments concatenated in time order (an append-only log),
        so every actor input is a contiguous slice of the critic input.

Values are always defined on the response region of each segment, and ``value_mask``
selects the actor-generated tokens there, the same in both settings.
"""

from dataclasses import dataclass


@dataclass
class Segment:
    tokens: list[int]
    response_length: int
    loss_mask: list[int]

    def __post_init__(self) -> None:
        assert 0 <= self.response_length <= len(self.tokens)
        assert len(self.loss_mask) == self.response_length

    @property
    def response_start(self) -> int:
        return len(self.tokens) - self.response_length


@dataclass
class CriticInput:
    tokens: list[int]
    # One (start, end) per segment: where that segment's response region sits in ``tokens``.
    # A segment whose response region was (partly) truncated away gets the surviving part.
    response_spans: list[tuple[int, int]]
    # Per segment, how many leading response tokens were lost to truncation (0 normally).
    response_tokens_dropped: list[int]
    # Tokens removed from the front of the log to fit ``max_len``. Nonzero means the critic
    # underwent an implicit compaction and must be reported.
    truncated_tokens: int = 0


def build_shared(segments: list[Segment]) -> list[CriticInput]:
    """One critic input per segment, identical to the actor input."""
    return [
        CriticInput(
            tokens=list(seg.tokens),
            response_spans=[(seg.response_start, len(seg.tokens))],
            response_tokens_dropped=[0],
        )
        for seg in segments
    ]


def build_full(segments: list[Segment], max_len: int | None = None) -> CriticInput:
    """One critic input per episode: all segments concatenated in time order.

    If the log exceeds ``max_len`` its oldest tokens are dropped and the amount is recorded.
    """
    tokens: list[int] = []
    spans: list[tuple[int, int]] = []
    for seg in segments:
        offset = len(tokens)
        tokens.extend(seg.tokens)
        spans.append((offset + seg.response_start, offset + len(seg.tokens)))

    cut = max(len(tokens) - max_len, 0) if max_len is not None else 0
    dropped = [min(max(cut - start, 0), end - start) for start, end in spans]
    spans = [(max(start - cut, 0), max(end - cut, 0)) for start, end in spans]
    return CriticInput(tokens=tokens[cut:], response_spans=spans, response_tokens_dropped=dropped, truncated_tokens=cut)


def split_values(critic_input: CriticInput, values: list, segments: list[Segment], fill=None) -> list[list]:
    """Map per-token critic outputs back onto each segment's response region.

    ``values`` has one entry per token of ``critic_input.tokens``. Returns, per segment, a
    list of length ``response_length`` aligned with that segment's ``loss_mask``. Positions
    lost to truncation get ``fill``.
    """
    assert len(values) == len(critic_input.tokens)
    assert len(critic_input.response_spans) == len(segments)
    out = []
    for seg, (start, end), dropped in zip(
        segments, critic_input.response_spans, critic_input.response_tokens_dropped, strict=True
    ):
        assert dropped + (end - start) == seg.response_length
        out.append([fill] * dropped + list(values[start:end]))
    return out


def value_mask(critic_input: CriticInput, segments: list[Segment]) -> list[int]:
    """1 on tokens of ``critic_input.tokens`` that carry a value target (actor-generated), else 0."""
    mask = [0] * len(critic_input.tokens)
    for seg, (start, end), dropped in zip(
        segments, critic_input.response_spans, critic_input.response_tokens_dropped, strict=True
    ):
        mask[start:end] = seg.loss_mask[dropped:]
    return mask
