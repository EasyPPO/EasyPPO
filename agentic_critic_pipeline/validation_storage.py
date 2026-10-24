"""Compact validation predictions: segment-local means of 100 generated tokens."""

import io
import json
from collections import defaultdict

import numpy as np

from agentic_critic_pipeline.data import digest
from agentic_critic_pipeline.validation_metrics import ValidationMetrics, _moments

VALUE_BIN_SIZE = 100


def compact_episode(episode):
    """Reduce predictions only after computing exact token-level sufficient statistics."""
    stats = ValidationMetrics(
        dict(selected_episode_ids=[], baseline_tokens=1024, profile_span=3076),
        mode=episode.get("mode", ""),
        batch=episode.get("batch", 0),
        collect_profiles=False,
    )
    events = io.StringIO()
    stats.add(episode, events=events)
    segments = []
    for segment in episode["segments"]:
        values = np.asarray(segment["values"], dtype=np.float64)
        positions = segment["response_positions"]
        if len(values) != len(positions) or not len(values):
            raise ValueError("Empty or misaligned validation segment")
        starts = list(range(0, len(values), VALUE_BIN_SIZE))
        ends = [min(start + VALUE_BIN_SIZE, len(values)) for start in starts]
        segments.append(
            dict(
                segment_index=segment["segment_index"],
                tokens=len(values),
                positions_sha256=digest(positions),
                values=[float(values[start:end].mean()) for start, end in zip(starts, ends, strict=True)],
                token_counts=[end - start for start, end in zip(starts, ends, strict=True)],
                response_position_ranges=[
                    [positions[start], positions[end - 1] + 1] for start, end in zip(starts, ends, strict=True)
                ],
                moments=_moments(values, episode["reward"]).tolist(),
            )
        )
    # Only O(compactions) scalar statistics, never per-event L1 arrays.
    boundaries = [
        {k: v for k, v in event.items() if k not in ("episode_id", "problem_id", "mode", "batch")}
        for event in map(json.loads, events.getvalue().splitlines())
    ]
    result = dict(episode, schema_version=2, value_bin_size=VALUE_BIN_SIZE, segments=segments, boundaries=boundaries)
    result["data_sha256"] = digest(result)
    return result


def compact_rollout(rollout_data):
    """Convert a whole-episode DP shard while full predictions are still in memory."""

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    grouped = defaultdict(list)
    full = "value_slices" in rollout_data
    slices = (
        rollout_data.get("value_slices")
        if full
        else [(i, 0, length) for i, length in enumerate(rollout_data["response_lengths"])]
    )
    targets = {}
    arrays = {}
    for local, start, length in slices:
        identity = int(rollout_data["rollout_ids"][local])
        if local not in arrays:
            arrays[local] = (
                array(rollout_data["values"][local]).astype(np.float64),
                array(rollout_data["loss_masks"][local]).astype(bool),
            )
        values, mask = (part[start : start + length] for part in arrays[local])
        if len(values) != length or len(mask) != length:
            raise ValueError("Validation prediction/mask length mismatch")
        target = float(rollout_data["raw_reward"][local])
        if identity in targets and targets[identity] != target:
            raise ValueError("Validation sibling return targets differ")
        targets[identity] = target
        grouped[identity].append(
            dict(
                segment_index=len(grouped[identity]),
                values=values[mask].tolist(),
                response_positions=np.flatnonzero(mask).tolist(),
            )
        )
    return [
        dict(
            rollout_id=identity,
            prediction=compact_episode(
                dict(
                    episode_id=str(identity),
                    problem_id="",
                    reward=targets[identity],
                    n_compactions=len(segments) - 1,
                    segments=segments,
                )
            ),
        )
        for identity, segments in grouped.items()
    ]


def validate_compact_episode(episode, frozen):
    """Check persisted means, counts and source alignment before consuming them."""
    if episode.get("schema_version") != 2 or episode.get("value_bin_size") != VALUE_BIN_SIZE:
        raise ValueError("Expected 100-token validation value bins")
    if episode.get("data_sha256") != digest({k: v for k, v in episode.items() if k != "data_sha256"}):
        raise ValueError("Saved validation values failed their content checksum")
    if any(episode[key] != frozen[key] for key in ("reward", "n_compactions")):
        raise ValueError("Saved validation values differ from frozen returns/compactions")
    segments = episode["segments"]
    if [s["segment_index"] for s in segments] != list(range(len(frozen["segments"]))):
        raise ValueError("Saved validation values have incomplete or unordered segments")
    for segment, expected in zip(segments, frozen["segments"], strict=True):
        count = expected["tokens"]
        counts = [min(VALUE_BIN_SIZE, count - start) for start in range(0, count, VALUE_BIN_SIZE)]
        if (
            segment["tokens"] != count
            or segment["positions_sha256"] != expected["positions_sha256"]
            or segment["token_counts"] != counts
            or len(segment["values"]) != len(counts)
            or len(segment["response_position_ranges"]) != len(counts)
            or not np.isfinite(segment["values"]).all()
            or len(segment["moments"]) != 6
            or segment["moments"][0] != count
            or not np.isfinite(segment["moments"]).all()
        ):
            raise ValueError("Saved validation values have incorrect token alignment/statistics")
