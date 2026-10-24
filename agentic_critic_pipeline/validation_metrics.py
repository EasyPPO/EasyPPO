"""Streaming validation EV, compaction deviations and 100-token traces.

G is the observed undiscounted final return (a Monte Carlo value target), not
an oracle estimate of the conditional expected value. Only generated actor
and summary tokens are on the horizontal axis; inserted context is masked.
"""

import json
from collections import defaultdict

import numpy as np


def _moments(values, target):
    error = values - target
    return np.array(
        [len(values), len(values) * target, len(values) * target**2, error.sum(), error @ error, np.abs(error).sum()],
        dtype=np.float64,
    )


def _metrics(moment):
    n, sy, syy, se, see, sae = moment
    if not n:
        return dict(tokens=0, ev=None, mse=None, mae=None, target_variance=None)
    variance = max(0.0, syy / n - (sy / n) ** 2)
    residual_variance = max(0.0, see / n - (se / n) ** 2)
    return dict(
        tokens=int(n),
        ev=None if variance <= 1e-14 else 1 - residual_variance / variance,
        mse=see / n,
        mae=sae / n,
        target_variance=variance,
    )


def _distribution(values):
    if not values:
        return dict(events=0, mean=None, median=None, p25=None, p75=None)
    return dict(
        events=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p25=float(np.quantile(values, 0.25)),
        p75=float(np.quantile(values, 0.75)),
    )


class ValidationMetrics:
    """Aggregate exact token statistics and optionally reconstruct display profiles."""

    def __init__(self, plan, *, mode, batch, collect_profiles=True):
        self.mode, self.batch = mode, batch
        self.collect_profiles = collect_profiles
        self.baseline_tokens, self.span = plan["baseline_tokens"], plan["profile_span"]
        self.selected = set(plan["selected_episode_ids"])
        self.moments = defaultdict(lambda: np.zeros(6))
        self.profile_offsets = list(range(-self.span, self.span))
        self.profile_sum = defaultdict(lambda: np.zeros(2 * self.span))
        self.profile_count = defaultdict(lambda: np.zeros(2 * self.span, dtype=np.int64))
        self.jumps = defaultdict(list)
        self.episodes = self.events = self.ineligible = 0

    def add(self, episode, *, events=None, traces=None):
        if episode.get("value_bin_size"):
            self._add_compact(episode, traces=traces)
            return
        segments = [np.asarray(s["values"], dtype=np.float64) for s in episode["segments"]]
        values = np.concatenate(segments)
        target, total = episode["reward"], episode["n_compactions"]
        if not len(values) or not np.isfinite(values).all() or not np.isfinite(target):
            raise ValueError("Empty or nonfinite value trajectory")
        self.episodes += 1
        self.moments["all"] += _moments(values, target)
        self.moments[f"total/{total}"] += _moments(values, target)
        for prior, part in enumerate(segments):
            self.moments[f"prior/{prior}"] += _moments(part, target)
        boundaries = np.cumsum([len(part) for part in segments])[:-1]
        for ordinal, center in enumerate(boundaries, 1):
            event = self._event(episode, values, int(center), ordinal)
            if events is not None:
                events.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        if traces is not None and episode["episode_id"] in self.selected:
            trace = dict(
                episode_id=episode["episode_id"],
                problem_id=episode["problem_id"],
                mode=self.mode,
                batch=self.batch,
                n_compactions=total,
                actor_token_index=list(range(len(values))),
                value_prediction=values.tolist(),
                value_target=[target] * len(values),
                target_definition="observed_final_return",
                compaction_boundaries=boundaries.tolist(),
                segments=episode["segments"],
            )
            traces.write(json.dumps(trace, separators=(",", ":"), allow_nan=False) + "\n")

    def _add_compact(self, episode, *, traces):
        self.episodes += 1
        total = episode["n_compactions"]
        for segment in episode["segments"]:
            moment = np.asarray(segment["moments"], dtype=np.float64)
            for key in ("all", f"total/{total}", f'prior/{segment["segment_index"]}'):
                self.moments[key] += moment
        for event in episode["boundaries"]:
            self.events += 1
            if not event["eligible"]:
                self.ineligible += 1
                continue
            keys = ["all", f'index/{event["compaction_index"]}', f"total/{total}"]
            for key in keys:
                self.jumps[f"{key}/first_token"].append(event["first_token_jump"])
                if event["post_window_tokens"] == self.baseline_tokens:
                    self.jumps[f"{key}/mean_shift_1024"].append(event["mean_shift"])
            if self.collect_profiles:
                self._compact_profile(episode, event, keys)
        if traces is not None and episode["episode_id"] in self.selected:
            values = [v for segment in episode["segments"] for v in segment["values"]]
            counts = [n for segment in episode["segments"] for n in segment["token_counts"]]
            ends = np.cumsum(counts)
            starts = ends - counts
            trace = dict(
                episode_id=episode["episode_id"],
                problem_id=episode["problem_id"],
                mode=self.mode,
                batch=self.batch,
                n_compactions=total,
                value_bin_size=100,
                actor_token_index=((starts + ends - 1) / 2).tolist(),
                actor_token_ranges=list(map(list, zip(starts.tolist(), ends.tolist(), strict=True))),
                token_counts=counts,
                value_prediction=values,
                value_target=[episode["reward"]] * len(values),
                target_definition="observed_final_return",
                compaction_boundaries=[event["actor_token_index"] for event in episode["boundaries"]],
            )
            traces.write(json.dumps(trace, separators=(",", ":"), allow_nan=False) + "\n")

    def _compact_profile(self, episode, event, keys):
        # Integrate piecewise-constant bin means on a 100-token relative grid.
        # This is |mean(V_bin)-baseline|, NOT mean(|V_token-baseline|).
        size = episode["value_bin_size"]
        edges = sorted({-self.span, self.span, *range(-(self.span // size) * size, self.span, size)})
        self.profile_offsets = edges[:-1]
        values = np.array([v for segment in episode["segments"] for v in segment["values"]])
        counts = np.array([n for segment in episode["segments"] for n in segment["token_counts"]])
        ends = np.cumsum(counts) - event["actor_token_index"]
        starts = ends - counts
        deviation = np.abs(values - event["baseline"])
        for key in keys:
            if key not in self.profile_sum:
                self.profile_sum[key] = np.zeros(len(edges) - 1)
                self.profile_count[key] = np.zeros(len(edges) - 1, dtype=np.int64)
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            overlap = np.maximum(0, np.minimum(ends, right) - np.maximum(starts, left))
            if overlap.sum():
                mean = float(deviation @ overlap / overlap.sum())
                for key in keys:
                    self.profile_sum[key][i] += mean
                    self.profile_count[key][i] += 1

    def _event(self, episode, values, center, ordinal):
        self.events += 1
        event = dict(
            episode_id=episode["episode_id"],
            problem_id=episode["problem_id"],
            batch=self.batch,
            mode=self.mode,
            compaction_index=ordinal,
            total_compactions=episode["n_compactions"],
            actor_token_index=center,
            baseline_tokens=self.baseline_tokens,
            eligible=center >= self.baseline_tokens,
        )
        if not event["eligible"]:
            self.ineligible += 1
            return dict(event, reason="fewer_than_1024_preceding_actor_tokens")
        baseline = float(values[center - self.baseline_tokens : center].mean())
        point_jump = float(abs(values[center] - baseline))
        post = values[center : min(len(values), center + self.baseline_tokens)]
        mean_jump = float(abs(post.mean() - baseline))
        keys = ["all", f"index/{ordinal}", f'total/{episode["n_compactions"]}']
        profile = {}
        if self.collect_profiles:
            left, right = max(0, center - self.span), min(len(values), center + self.span)
            offsets = np.arange(left - center, right - center)
            l1 = np.abs(values[left:right] - baseline)
            for key in keys:
                self.profile_sum[key][offsets + self.span] += l1
                self.profile_count[key][offsets + self.span] += 1
            profile = dict(offsets=offsets.tolist(), l1_from_baseline=l1.tolist())
        for key in keys:
            self.jumps[f"{key}/first_token"].append(point_jump)
            # Keep partial windows in the raw record, but compare equal windows in summaries.
            if len(post) == self.baseline_tokens:
                self.jumps[f"{key}/mean_shift_1024"].append(mean_jump)
        return dict(
            event,
            baseline=baseline,
            **profile,
            first_token_jump=point_jump,
            post_window_tokens=len(post),
            mean_shift=mean_jump,
            post_mean_l1=float(np.abs(post - baseline).mean()),
        )

    def summary(self):
        profiles = {}
        for key in sorted(self.profile_sum):
            counts = self.profile_count[key]
            means = np.divide(self.profile_sum[key], counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
            profiles[key] = dict(
                offsets=self.profile_offsets,
                mean_l1=[float(v) if n else None for v, n in zip(means, counts, strict=True)],
                event_counts=counts.tolist(),
            )
        return dict(
            schema_version=1,
            mode=self.mode,
            batch=self.batch,
            episodes=self.episodes,
            target_definition="observed_final_return",
            ev_definition="1 - Var(G - V) / Var(G); token weighted",
            baseline_definition=f"mean of {self.baseline_tokens} actor tokens immediately before compaction",
            profile_span=self.span,
            baseline_tokens=self.baseline_tokens,
            compaction_events=self.events,
            ineligible_events=self.ineligible,
            ev={k: _metrics(v) for k, v in sorted(self.moments.items())},
            profiles=profiles,
            jumps={k: _distribution(v) for k, v in sorted(self.jumps.items())},
        )
