"""Read existing Miles replay and original events without generating trajectories."""

import json
from collections import Counter
from pathlib import Path

import torch

from agentic_critic.fcs_env import Episode, Event
from agentic_critic.kv_compact import validate_summary_segments
from agentic_critic.memory_config import resolve_memory_config
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.validation import validate_training_replay


def index_events(path):
    """A small byte-offset index; never retain the whole trajectory pool in RAM."""
    offsets = {}
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            record = json.loads(line)
            identity = str(record.get("episode_id", record["sample"]))
            if identity in offsets:
                raise ValueError("Duplicate trajectory identity in existing events")
            offsets[identity] = offset
    return offsets


def load_batch(path, source, offsets):
    """Use exact replay tokens; recover absent summary offsets from existing events."""
    groups = {}
    for sample in torch.load(path, map_location="cpu", weights_only=False)["samples"]:
        groups.setdefault(sample["rollout_id"], []).append(sample)
    episodes = []
    with Path(source).open("rb") as stream:
        for rollout_id, samples in groups.items():
            samples.sort(key=lambda s: s["metadata"]["segment"])
            meta = samples[0]["metadata"]
            identity = str(meta.get("episode_id", meta["sample"]))
            stream.seek(offsets[identity])
            record = json.loads(stream.readline())
            if str(record.get("episode_id", record["sample"])) != identity:
                raise ValueError("Trajectory byte index no longer matches source")
            if record["finish_reason"] not in ("max_turns", "submissions_exhausted"):
                raise ValueError("Training requires existing naturally completed trajectories")
            original = Episode(events=[Event(**e) for e in record["events"]], n_compactions=record["n_compactions"])
            segments = original.to_segments()
            if len(segments) != len(samples):
                raise ValueError("Replay does not contain the complete trajectory")
            reward = max(record["scores"], default=0.0) / 100.0
            for index, (sample, segment) in enumerate(zip(samples, segments, strict=True)):
                m = sample["metadata"]
                if (
                    m["segment"] != index
                    or str(m.get("episode_id", m["sample"])) != identity
                    or str(m["problem"]) != str(record["problem"])
                ):
                    raise ValueError("Replay episode/segment identities differ from original trajectory")
                if (
                    sample["tokens"] != segment.tokens
                    or sample["loss_mask"] != segment.loss_mask
                    or sample["response_length"] != segment.response_length
                ):
                    raise ValueError(
                        "Replay tokens or masks differ from original trajectory; no retokenization allowed"
                    )
                if float(sample["reward"]) != reward or not any(segment.loss_mask):
                    raise ValueError("Replay return or supervised-token count is invalid")
                if m.get("summary_span") is not None and tuple(m["summary_span"]) != segment.summary_span:
                    raise ValueError("Replay summary span differs from original actor event")
            validate_summary_segments(segments, resolve_memory_config("attention_matching", None))
            episodes.append(
                dict(
                    episode_id=identity,
                    rollout_id=rollout_id,
                    problem=str(record["problem"]),
                    reward=reward,
                    segments=segments,
                    indices=[s["index"] for s in samples],
                )
            )
    return episodes


def prepare_replay(directory, plan, *, steps, mini_batches, max_tokens):
    """Validate the existing cohort once; return immutable identities and file hashes."""
    validate_training_replay(plan, directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = directory / "episodes.jsonl"
    offsets = index_events(source)
    declared = json.loads((directory / "episodes.identities.json").read_text())["episodes"]
    if set(offsets) != {str(r["episode_id"]) for r in declared}:
        raise ValueError("Original events and frozen training identities disagree")
    config = manifest["args"]
    count = config["problems_per_step"] * config["episodes_per_problem"]
    if (
        min(config["problems_per_step"], config["episodes_per_problem"], mini_batches) < 1
        or count % mini_batches
        or mini_batches > count
    ):
        raise ValueError("Replay batch must divide into nonempty mini-batches")
    steps = len(manifest["steps"]) if steps is None else steps
    if not 0 < steps <= len(manifest["steps"]):
        raise ValueError("Requested training steps exceed the existing replay")
    seen, files = set(), []
    for step, row in enumerate(manifest["steps"][:steps]):
        if row["step"] != step or row["episodes"] != count:
            raise ValueError("Existing replay must start at zero and contain complete configured batches")
        path = directory / f"rollout_{step}.pt"
        episodes = load_batch(path, source, offsets)
        if len(episodes) != count or Counter(ep["problem"] for ep in episodes) != Counter(
            {str(p): config["episodes_per_problem"] for p in row["problems"]}
        ):
            raise ValueError("Replay prompt groups differ from manifest")
        for episode in episodes:
            identity = episode["episode_id"]
            if identity in seen or identity in plan["episodes"]:
                raise ValueError("Repeated training trajectory or validation leakage")
            seen.add(identity)
            if sum(len(s.tokens) for s in episode["segments"]) > max_tokens:
                raise ValueError("Training trajectory exceeds max-tokens; truncation is forbidden")
        files.append(
            dict(path=str(path.resolve()), sha256=data_sha256(path), episode_ids=[e["episode_id"] for e in episodes])
        )
    if steps == len(manifest["steps"]) and seen != set(offsets):
        raise ValueError("Complete replay does not consume the existing trajectory cohort exactly once")
    return dict(
        source=str(source.resolve()),
        source_sha256=data_sha256(source),
        offsets=offsets,
        files=files,
        episodes_per_batch=count,
        mini_batches=mini_batches,
        steps=steps,
    )
