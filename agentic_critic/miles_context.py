"""Attach exact AM layouts to existing Miles samples without altering actor data."""

import json
from functools import lru_cache
from pathlib import Path

from agentic_critic.context_layout import episode_layout
from agentic_critic.critic_input import Segment
from agentic_critic.fcs_env import Episode, Event


def _identity(row):
    return str(row["episode_id"] if "episode_id" in row else row["sample"])


@lru_cache(maxsize=4)
def _event_index(path, size, mtime):
    offsets = {}
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            identity = _identity(json.loads(line))
            if identity in offsets:
                raise ValueError("Duplicate original episode identity in critic context events")
            offsets[identity] = offset
    return offsets


def _recover_segments(path, samples):
    source = Path(path)
    stat = source.stat()
    offsets = _event_index(str(source.resolve()), stat.st_size, stat.st_mtime_ns)
    identity = _identity(samples[0].metadata)
    if identity not in offsets:
        raise ValueError(f"Original critic context events missing episode {identity}")
    with source.open("rb") as stream:
        stream.seek(offsets[identity])
        record = json.loads(stream.readline())
    events = [Event(**{k: e[k] for k in ("tokens", "actor", "segment", "kind")}) for e in record["events"]]
    segments = Episode(events=events, n_compactions=record["n_compactions"]).to_segments()
    if len(segments) != len(samples):
        raise ValueError("Original event episode and replay segment count disagree")
    for segment, sample in zip(segments, samples, strict=True):
        if segment.tokens != sample.tokens or segment.loss_mask != sample.loss_mask or segment.response_length != sample.response_length:
            raise ValueError("Original event tokens/masks disagree with Miles replay")
        span = sample.metadata.get("summary_span")
        if span is not None and tuple(span) != segment.summary_span:
            raise ValueError("Replay summary offsets disagree with original events")
    return segments


def prepare_context_samples(samples, *, events=None):
    """Populate train_metadata; retain original tokens, order, masks and rewards.

    Legacy replay may lack summary offsets. Recover them only from its original
    event sidecar, checking exact token identities; never infer a summary from text.
    """
    groups = {}
    for sample in samples:
        groups.setdefault(sample.rollout_id, []).append(sample)
    for rollout, siblings in groups.items():
        if rollout is None:
            raise ValueError("AM requires episode rollout IDs")
        ordered = sorted(siblings, key=lambda s: s.metadata["segment"])
        declared = ordered[0].metadata.get("n_compactions")
        if (declared is None and events is None) or (declared is not None and declared + 1 != len(ordered)):
            raise ValueError("AM requires the declared complete compaction history")
        if [s.metadata["segment"] for s in ordered] != list(range(len(ordered))):
            raise ValueError("AM requires complete original episode segments")
        missing = any(s.metadata.get("summary_span") is None for s in ordered[:-1])
        if missing and events is None:
            raise ValueError("Legacy AM replay needs --critic-context-events pointing to original episodes.jsonl")
        segments = _recover_segments(events, ordered) if missing or declared is None else [Segment(s.tokens, s.response_length, s.loss_mask, s.metadata.get("summary_span")) for s in ordered]
        for sample, layout in zip(ordered, episode_layout(segments, episode=rollout), strict=True):
            sample.train_metadata = {**(sample.train_metadata or {}), "critic_context": layout.to_dict()}
