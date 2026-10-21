"""Trajectory schema, token alignment, and deterministic cohort scheduling.

These helpers contain no critic training or optimizer implementation.
"""

import hashlib
import json
import math
import random
from array import array
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkItem:
    episode: dict
    input_ids: Sequence[int]
    positions: list
    locations: list  # (segment index, offset in response suffix)


def validate_episode(ep):
    for field in ("episode_id", "problem_id", "split", "reward", "segments"):
        if field not in ep:
            raise ValueError(f"Missing episode field {field}")
    if not math.isfinite(ep["reward"]) or not ep["segments"]:
        raise ValueError("Episode must have finite reward and nonempty segments")
    count = 0
    for segment in ep["segments"]:
        ids, start, mask = segment["input_ids"], segment["response_start"], segment["loss_mask"]
        if not 1 <= start <= len(ids) or len(mask) != len(ids) - start:
            raise ValueError("Response must have a preceding token and an exactly aligned suffix mask")
        if any(x not in (0, 1) for x in mask) or any(not isinstance(i, int) or not 0 <= i < 2**32 for i in ids):
            raise ValueError("Token IDs must be unsigned 32-bit integers and loss masks must be binary")
        count += sum(mask)
    if not count:
        raise ValueError("Episode has no supervised actor tokens")


def read_episodes(path, *, split=None):
    """Stream JSONL and retain only compact, training-required fields.

    uint32 token buffers and byte masks cost at most five bytes per token;
    events, text, and rollout metadata are released after each input record.
    Only one original JSON record is materialized at a time. Duplicate checking
    and schema validation still cover every split in the source file.
    """
    if array("I").itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit array('I') storage")
    episodes, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            ep = json.loads(line)
            validate_episode(ep)
            if ep["episode_id"] in seen:
                raise ValueError(f"Duplicate episode ID: {ep['episode_id']}")
            seen.add(ep["episode_id"])
            if split is None or ep["split"] == split:
                episodes.append(
                    {
                        "episode_id": ep["episode_id"],
                        "problem_id": ep["problem_id"],
                        "split": ep["split"],
                        "reward": ep["reward"],
                        "segments": [
                            {
                                "input_ids": array("I", segment["input_ids"]),
                                "response_start": segment["response_start"],
                                "loss_mask": bytearray(segment["loss_mask"]),
                            }
                            for segment in ep["segments"]
                        ],
                    }
                )
            del ep
    if not episodes:
        raise ValueError(f"No episodes for split {split} in {path}")
    return episodes


def data_sha256(path):
    """Hash using a bounded reusable buffer instead of reading the entire JSONL."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def work_items(ep, mode, max_tokens):
    """The two views supervise exactly the same actor tokens at t-1 positions."""
    validate_episode(ep)
    items, tokens, positions, locations = [], array("I"), [], []
    for index, segment in enumerate(ep["segments"]):
        start = segment["response_start"]
        offset = len(tokens) if mode == "full_history" else 0
        pos = [offset + start + j - 1 for j, flag in enumerate(segment["loss_mask"]) if flag]
        loc = [(index, j) for j, flag in enumerate(segment["loss_mask"]) if flag]
        if mode == "actor_summary":
            if pos:
                items.append(WorkItem(ep, segment["input_ids"], pos, loc))
        elif mode == "full_history":
            tokens.extend(segment["input_ids"])
            positions.extend(pos)
            locations.extend(loc)
        else:
            raise ValueError(f"Unknown mode {mode}")
    if mode == "full_history":
        items.append(WorkItem(ep, tokens, positions, locations))
    for item in items:
        if len(item.input_ids) > max_tokens:
            raise ValueError(
                f"{ep['episode_id']}: {len(item.input_ids)} > max_tokens={max_tokens}; truncation is forbidden"
            )
    return items


def batch_schedule(episodes, problems_per_batch, episodes_per_problem, epochs, seed, allow_incomplete):
    by_problem = defaultdict(list)
    for ep in episodes:
        by_problem[ep["problem_id"]].append(ep)
    if not allow_incomplete:
        if len(by_problem) % problems_per_batch or any(len(v) % episodes_per_problem for v in by_problem.values()):
            raise ValueError(
                "Dataset does not divide into requested problem/episode batches; "
                "use --allow-incomplete-batches only for an explicit smoke run"
            )
    schedule = []
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        shuffled = {p: list(v) for p, v in by_problem.items()}
        for values in shuffled.values():
            rng.shuffle(values)
        for start in range(0, max(map(len, shuffled.values())), episodes_per_problem):
            problems = [p for p, values in shuffled.items() if start < len(values)]
            rng.shuffle(problems)
            for group in range(0, len(problems), problems_per_batch):
                batch = [
                    ep
                    for p in problems[group : group + problems_per_batch]
                    for ep in shuffled[p][start : start + episodes_per_problem]
                ]
                if not allow_incomplete and len(batch) != problems_per_batch * episodes_per_problem:
                    raise ValueError("Uneven episode counts create an incomplete logical batch")
                schedule.append(batch)
    return schedule
