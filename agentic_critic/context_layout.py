"""Exact actor-segment layouts for problem / AM history / resume-suffix attention.

No text is decoded or retokenized. Linear layers continue to receive each original
actor segment as an independent packed sequence. Only full-attention reads differ.
"""

from dataclasses import asdict, dataclass
from itertools import groupby

SCHEMA = "problem_am_resume_v1"


@dataclass(frozen=True)
class ContextSegment:
    episode: str
    index: int
    count: int
    length: int
    problem_prefix: int
    summary_span: tuple[int, int] | None

    def __post_init__(self):
        if not 0 <= self.index < self.count or not 0 < self.problem_prefix < self.length:
            raise ValueError("Invalid critic context segment or problem prefix")
        if self.summary_span is not None:
            a, b = self.summary_span
            if not self.problem_prefix <= a < b <= self.length:
                raise ValueError("Summary must follow the protected problem prefix")
        elif self.index + 1 < self.count:
            raise ValueError("Nonfinal AM segments require original summary event offsets")

    def to_dict(self):
        return dict(schema=SCHEMA, **asdict(self))

    @classmethod
    def from_dict(cls, row):
        if row.get("schema") != SCHEMA:
            raise ValueError("Unsupported critic context schema")
        values = {key: row[key] for key in cls.__dataclass_fields__}
        if values["summary_span"] is not None:
            values["summary_span"] = tuple(values["summary_span"])
        return cls(**values)


def episode_layout(segments, *, episode):
    """Protect the exact token prefix shared by the initial and resumed prompts.

    This is a conservative token boundary: BPE may place a template delimiter in
    the suffix. Keeping the entire remaining resume prompt preserves it exactly,
    including the summary preamble, continuation instruction and assistant start.
    """
    if not segments or any(s.response_start < 1 for s in segments):
        raise ValueError("AM requires complete original prompts and response masks")
    prompts = [s.tokens[: s.response_start] for s in segments]
    prefix = min(map(len, prompts))
    first = prompts[0]
    for prompt in prompts[1:]:
        prefix = next((i for i in range(prefix) if first[i] != prompt[i]), prefix)
    if prefix == 0:
        raise ValueError("Actor prompts have no identical problem/template token prefix")
    return [ContextSegment(str(episode), i, len(segments), len(s.tokens), prefix, s.summary_span) for i, s in enumerate(segments)]


def validate_packed_layout(rows, lengths):
    """Refuse incomplete/reordered episodes instead of silently dropping memory."""
    if len(rows) != len(lengths):
        raise ValueError("AM layout must cover every real packed segment")
    layouts = [ContextSegment.from_dict(row) for row in rows]
    if [s.length for s in layouts] != list(lengths):
        raise ValueError("AM layout lengths differ from original actor tokens")
    seen = set()
    for episode, group in groupby(layouts, key=lambda s: s.episode):
        members = list(group)
        if episode in seen or [s.index for s in members] != list(range(members[0].count)):
            raise ValueError("AM micro-batches must contain complete, ordered episodes")
        if any(s.count != len(members) or s.problem_prefix != members[0].problem_prefix for s in members):
            raise ValueError("Inconsistent episode context layout")
        seen.add(episode)
    return layouts
