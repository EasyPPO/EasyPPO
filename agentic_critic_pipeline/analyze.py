"""Paired offline critic diagnostics; token statistics stream into compact records.

Confidence intervals resample problems, jointly for both critics. Undefined
statistics are JSON null; no missing episode or token alignment is ignored.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BUCKETS = ("all", "0", "1", "2", "3", "4+")


def bucket(segment):
    return str(segment) if segment < 4 else "4+"


def finite(value):
    return float(value) if np.isfinite(value) else None


def statistics(values, reward):
    values = np.asarray(values, dtype=np.float64)
    error = values - reward
    return np.array(
        [len(values), len(values) * reward, len(values) * reward**2, error.sum(), error @ error, np.abs(error).sum()],
        dtype=np.float64,
    )


def metrics(stats):
    n, sy, syy, se, see, sae = stats
    if n == 0:
        return dict(tokens=0, ev=None, mse=None, mae=None, bias=None, target_variance=None)
    variance = max(0.0, syy / n - (sy / n) ** 2)
    residual_variance = max(0.0, see / n - (se / n) ** 2)
    return dict(
        tokens=int(n),
        ev=None if variance <= 1e-14 else 1 - residual_variance / (variance + 1e-5),
        mse=see / n,
        mae=sae / n,
        bias=se / n,
        target_variance=variance,
    )


def _key(row):
    return row["phase"], row["batch"], row["episode_id"], row["segment_index"]


def load_values(paths, mode, *, phase="all", window=32, span=256, allow_input_swap_diagnostic=False, trace_bin=0):
    records, files = {}, set()
    trained_modes = set()
    retain = max(span, 2 * window)
    for source in paths:
        path = Path(source)
        files.update(path.glob("values.rank*.jsonl") if path.is_dir() else [path])
    if not files:
        raise ValueError(f"No value files found for {mode}")
    for path in sorted(files):
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["mode"] != mode:
                    raise ValueError(f"Wrong mode in {path}: {row['mode']} != {mode}")
                if row["phase"] not in ("pre_update", "heldout"):
                    raise ValueError(f"Unknown phase {row['phase']}")
                if phase != "all" and row["phase"] != phase:
                    continue
                if row["phase"] == "heldout" and row["split"] not in ("validation", "test"):
                    raise ValueError("Heldout values must come from validation/test, never the training split")
                if "trained_mode" not in row or "input_mode" not in row:
                    raise ValueError("Value provenance missing trained_mode/input_mode; regenerate the value dump")
                if row["input_mode"] != mode or row["trained_mode"] not in ("actor_summary", "full_history"):
                    raise ValueError("Invalid trained_mode/input_mode provenance")
                if row["trained_mode"] != mode and not allow_input_swap_diagnostic:
                    raise ValueError(
                        "Input-swap values cannot be used as a trained-critic comparison; "
                        "use --allow-input-swap-diagnostic explicitly"
                    )
                if row["trained_mode"] != mode and row["phase"] != "heldout":
                    raise ValueError("Input-swap diagnostics require heldout values")
                trained_modes.add(row["trained_mode"])
                if len(trained_modes) > 1:
                    raise ValueError("Cannot mix checkpoints trained with different context modes in one analysis side")
                key = _key(row)
                if key in records:
                    raise ValueError(f"Duplicate value record: {key}")
                if not isinstance(row["segment_index"], int) or row["segment_index"] < 0:
                    raise ValueError(f"Invalid segment index: {key}")
                positions = row.pop("response_positions")
                values = np.asarray(row.pop("values"), dtype=np.float64)
                if (
                    len(values) == 0
                    or len(values) != len(positions)
                    or not np.isfinite(values).all()
                    or not np.isfinite(row["reward"])
                    or any(not isinstance(x, int) or x < 0 for x in positions)
                    or any(a >= b for a, b in zip(positions, positions[1:], strict=False))
                ):
                    raise ValueError(f"Invalid values/positions/reward: {key}")
                row["positions_hash"] = hashlib.sha256(
                    json.dumps(positions, separators=(",", ":")).encode()
                ).hexdigest()
                row["stats"] = statistics(values, row["reward"])
                if trace_bin:
                    row["trace"] = [
                        dict(
                            start=i, end=min(i + trace_bin, len(values)), value=float(values[i : i + trace_bin].mean())
                        )
                        for i in range(0, len(values), trace_bin)
                    ]
                row["first"], row["last"] = values[:retain].copy(), values[-retain:].copy()
                midpoint = len(values) // 2
                row["mid_before"] = values[max(0, midpoint - retain) : midpoint].copy()
                row["mid_after"] = values[midpoint : midpoint + retain].copy()
                records[key] = row
    if not records:
        raise ValueError(f"No {phase} records for {mode}")
    return records


def pair_records(actor, full):
    if actor.keys() != full.keys():
        missing = list(actor.keys() - full.keys())[:3]
        extra = list(full.keys() - actor.keys())[:3]
        raise ValueError(f"Unpaired episode/segment/batch: missing Full={missing}, extra Full={extra}")
    pairs = []
    for key in sorted(actor):
        a, b = actor[key], full[key]
        for field in ("problem_id", "split", "reward", "positions_hash"):
            if a[field] != b[field]:
                raise ValueError(f"Pair mismatch {field}: {key}")
        if a["stats"][0] != b["stats"][0]:
            raise ValueError(f"Pair token count mismatch: {key}")
        pairs.append((a, b))
    episodes = defaultdict(list)
    for a, _ in pairs:
        episodes[_key(a)[:3]].append(a)
    for key, rows in episodes.items():
        if [r["segment_index"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Incomplete nonconsecutive episode segments: {key}")
        if len({(r["problem_id"], r["split"], r["reward"]) for r in rows}) != 1:
            raise ValueError(f"Inconsistent episode metadata: {key}")
    trained_episodes = {a["episode_id"] for a, _ in pairs if a["phase"] == "pre_update"}
    heldout_episodes = {a["episode_id"] for a, _ in pairs if a["phase"] == "heldout"}
    overlap = trained_episodes & heldout_episodes
    if overlap:
        raise ValueError(f"Heldout episodes overlap training episodes: {sorted(overlap)[:3]}")
    return pairs


def interval(draws, problems):
    valid = np.asarray([x for x in draws if x is not None and np.isfinite(x)], dtype=float)
    if problems < 2 or not len(valid):
        return dict(low=None, high=None, valid_resamples=len(valid), problems=problems)
    return dict(
        low=float(np.quantile(valid, 0.025)),
        high=float(np.quantile(valid, 0.975)),
        valid_resamples=len(valid),
        problems=problems,
    )


def paired_metrics(pairs, bootstrap=2000, seed=0, baseline_return=0.5):
    batches = {batch: i for i, batch in enumerate(sorted({a["batch"] for a, _ in pairs}))}
    groups = defaultdict(lambda: np.zeros((len(batches), 2, 6)))
    for a, b in pairs:
        groups[a["problem_id"]][batches[a["batch"]]] += np.stack([a["stats"], b["stats"]])
    by_batch = sum(groups.values(), np.zeros((len(batches), 2, 6)))
    total = by_batch.sum(axis=0)

    def within_batch(stats):
        valid = stats[stats[:, 0] > 0]
        if not len(valid):
            return None
        n, sy, syy, se, see = valid[:, :5].T
        residual_ss = np.maximum(0.0, see - se**2 / n).sum()
        target_ss = np.maximum(0.0, syy - sy**2 / n).sum()
        return None if target_ss <= 1e-14 else 1 - residual_ss / target_ss

    result = dict(
        actor_summary=metrics(total[0]),
        full_history=metrics(total[1]),
        episodes=len({(_key(a)[:3]) for a, _ in pairs}),
        problems=len(groups),
        segments=len(pairs),
    )
    result["actor_summary"]["within_batch_ev"] = within_batch(by_batch[:, 0])
    result["full_history"]["within_batch_ev"] = within_batch(by_batch[:, 1])
    n, sy, syy = total[0, :3]
    baseline = np.array(
        [
            n,
            sy,
            syy,
            n * baseline_return - sy,
            n * baseline_return**2 - 2 * baseline_return * sy + syy,
            sum(a["stats"][0] * abs(baseline_return - a["reward"]) for a, _ in pairs),
        ]
    )
    result["constant_baseline"] = dict(value=baseline_return, **metrics(baseline))
    result["oracle_constant_mse"] = None if not n else max(0.0, syy / n - (sy / n) ** 2)
    result["full_minus_actor"] = {}
    draws = defaultdict(list)
    if groups:
        array = np.stack(list(groups.values()))
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            resampled_batches = array[rng.integers(len(array), size=len(array))].sum(axis=0)
            resampled = resampled_batches.sum(axis=0)
            ma, mb = metrics(resampled[0]), metrics(resampled[1])
            ma["within_batch_ev"] = within_batch(resampled_batches[:, 0])
            mb["within_batch_ev"] = within_batch(resampled_batches[:, 1])
            for name in ("ev", "within_batch_ev", "mse", "mae", "bias"):
                draws[name].append(None if ma[name] is None or mb[name] is None else mb[name] - ma[name])
    for name in ("ev", "within_batch_ev", "mse", "mae", "bias"):
        a, b = result["actor_summary"][name], result["full_history"][name]
        result["full_minus_actor"][name] = dict(
            value=None if a is None or b is None else b - a, **interval(draws[name], len(groups))
        )
    return result


def boundary_events(pairs, *, window=32):
    episodes = defaultdict(list)
    for pair in pairs:
        episodes[_key(pair[0])[:3]].append(pair)
    boundary, midpoint = [], []

    def event(left, right, meta, index):
        sides = []
        for before, after in zip(left, right, strict=True):
            v0, v1 = float(before[-window:].mean()), float(after[:window].mean())
            reward = meta["reward"]
            sides.append(
                dict(
                    before=before,
                    after=after,
                    before_mean=v0,
                    after_mean=v1,
                    signed_jump=v1 - v0,
                    abs_jump=abs(v1 - v0),
                    error_improvement=abs(v0 - reward) - abs(v1 - reward),
                )
            )
        return dict(
            problem_id=meta["problem_id"],
            batch=meta["batch"],
            reward=meta["reward"],
            episode_id=meta["episode_id"],
            index=index,
            sides=sides,
            removed_tokens=meta.get("removed_tokens"),
            removed_cumulative=meta.get("removed_cumulative"),
        )

    for rows in episodes.values():
        rows.sort(key=lambda pair: pair[0]["segment_index"])
        for index, pair in enumerate(rows):
            a = pair[0]
            if min(len(a["mid_before"]), len(a["mid_after"])) >= 2 * window:
                midpoint.append(event([x["mid_before"] for x in pair], [x["mid_after"] for x in pair], a, index))
            if index and min(rows[index - 1][0]["stats"][0], a["stats"][0]) >= window:
                boundary.append(event([x["last"] for x in rows[index - 1]], [x["first"] for x in pair], a, index))
    return boundary, midpoint


def event_metrics(events, bootstrap=2000, seed=0):
    fields = ("signed_jump", "abs_jump", "error_improvement")
    groups = defaultdict(list)
    for event in events:
        groups[event["problem_id"]].append([[side[name] for name in fields] for side in event["sides"]])
    result = dict(events=len(events), problems=len(groups))
    means = np.mean([x for group in groups.values() for x in group], axis=0) if events else np.full((2, 3), np.nan)
    differences = []
    if groups:
        sums = np.array([np.sum(g, axis=0) for g in groups.values()])
        counts = np.array([len(g) for g in groups.values()])
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            take = rng.integers(len(groups), size=len(groups))
            m = sums[take].sum(axis=0) / counts[take].sum()
            differences.append(m[1] - m[0])
    for index, field in enumerate(fields):
        result[field] = dict(
            actor_summary=finite(means[0, index]),
            full_history=finite(means[1, index]),
            full_minus_actor=dict(
                value=finite(means[1, index] - means[0, index]),
                **interval([d[index] for d in differences], len(groups)),
            ),
        )
    return result


def profile(events, span, bin_size):
    valid = [e for e in events if all(len(s["before"]) >= span and len(s["after"]) >= span for s in e["sides"])]
    output = dict(events=len(valid), bins=[])
    if not valid:
        return output
    for start in range(-span, span, bin_size):
        end = min(start + bin_size, span)
        record = dict(start=start, end=end)
        for mode_index, mode in enumerate(("actor_summary", "full_history")):
            deviations, errors = [], []
            for e in valid:
                side = e["sides"][mode_index]
                values = np.concatenate([side["before"][-span:], side["after"][:span]])[start + span : end + span]
                deviations.append(values.mean() - side["before_mean"])
                errors.append(np.abs(values - e["reward"]).mean())
            record[mode] = dict(
                signed=float(np.mean(deviations)), absolute=float(np.abs(deviations).mean()), mae=float(np.mean(errors))
            )
        output["bins"].append(record)
    return output


def correlation(x, y):
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return None if norm <= 1e-14 else float(x @ y / norm)


def retention(events, bootstrap=2000, seed=0):
    grouped = defaultdict(list)
    for e in events:
        grouped[e["problem_id"], e["batch"]].append(e)
    problems = defaultdict(list)
    for (problem, _), group in grouped.items():
        if len(group) >= 3:
            matrix = np.array(
                [
                    [
                        e["sides"][0]["before_mean"],
                        e["sides"][0]["after_mean"],
                        e["sides"][1]["before_mean"],
                        e["sides"][1]["after_mean"],
                        e["reward"],
                    ]
                    for e in group
                ]
            )
            problems[problem].append(matrix - matrix.mean(axis=0))
    pieces = [np.concatenate(x) for x in problems.values()]

    def score(matrix, column):
        before, after, reward = matrix[:, column], matrix[:, column + 1], matrix[:, 4]
        cb, ca = correlation(before, reward), correlation(after, reward)
        return dict(
            retained_correlation=correlation(before, after),
            slope=None if before @ before <= 1e-14 else float(before @ after / (before @ before)),
            reward_correlation_before=cb,
            reward_correlation_after=ca,
            reward_correlation_change=None if cb is None or ca is None else ca - cb,
        )

    data = np.concatenate(pieces) if pieces else np.zeros((0, 5))
    result = dict(events=len(data), problems=len(pieces), actor_summary=score(data, 0), full_history=score(data, 2))
    samples = []
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap if pieces else 0):
        sampled = np.concatenate([pieces[i] for i in rng.integers(len(pieces), size=len(pieces))])
        a, b = score(sampled, 0)["reward_correlation_change"], score(sampled, 2)["reward_correlation_change"]
        samples.append(None if a is None or b is None else b - a)
    a, b = result["actor_summary"]["reward_correlation_change"], result["full_history"]["reward_correlation_change"]
    result["change_full_minus_actor"] = dict(
        value=None if a is None or b is None else b - a, **interval(samples, len(pieces))
    )
    return result


def variance_share(pairs):
    episodes = defaultdict(list)
    for pair in pairs:
        episodes[_key(pair[0])[:3]].append(pair)
    out = {}
    for mode_index, mode in enumerate(("actor_summary", "full_history")):
        between, total = 0.0, 0.0
        for rows in episodes.values():
            stats = np.stack([pair[mode_index]["stats"] for pair in rows])
            counts, sums, squares = stats[:, 0], stats[:, 3], stats[:, 4]
            # Reward is constant within an episode, so residual/value variance coincide.
            correction = sums.sum() ** 2 / counts.sum()
            between += max(0.0, np.sum(sums**2 / counts) - correction)
            total += max(0.0, squares.sum() - correction)
        out[mode] = dict(
            between_segment_fraction=None if total <= 1e-14 else between / total,
            between_sum_squares=between,
            total_sum_squares=total,
        )
    return out


def attach_episode_metadata(pairs, paths):
    records = {a["episode_id"]: {} for a, _ in pairs}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if ep["episode_id"] not in records:
                    continue
                if records[ep["episode_id"]]:
                    raise ValueError(f"Duplicate episode metadata {ep['episode_id']}")
                segments = ep["segments"]
                metadata = []
                removed_cumulative = 0
                for index, segment in enumerate(segments):
                    positions = [i for i, flag in enumerate(segment["loss_mask"]) if flag]
                    digest = hashlib.sha256(json.dumps(positions, separators=(",", ":")).encode()).hexdigest()
                    removed = len(segments[index - 1]["input_ids"]) - segment["response_start"] if index else 0
                    removed_cumulative += removed
                    metadata.append(
                        dict(positions_hash=digest, removed_tokens=removed, removed_cumulative=removed_cumulative)
                    )
                records[ep["episode_id"]] = dict(reward=ep["reward"], problem_id=ep["problem_id"], segments=metadata)
    for a, b in pairs:
        ep = records[a["episode_id"]]
        if not ep:
            raise ValueError(f"Missing original episode {a['episode_id']}")
        k = a["segment_index"]
        if ep["reward"] != a["reward"] or ep["problem_id"] != a["problem_id"]:
            raise ValueError("Original episode reward/problem mismatch")
        segment = ep["segments"][k]
        if segment["positions_hash"] != a["positions_hash"]:
            raise ValueError("Original episode actor-token positions mismatch")
        for row in (a, b):
            row["removed_tokens"] = segment["removed_tokens"]
            row["removed_cumulative"] = segment["removed_cumulative"]


def analyze(
    pairs, *, bootstrap=2000, seed=0, window=32, span=256, bin_size=16, primary_last_batches=30, baseline_return=0.5
):
    if min(window, span, bin_size, primary_last_batches, bootstrap) < 1:
        raise ValueError("Windows, bootstrap and primary batch count must be positive")
    result = dict(
        settings=dict(
            bootstrap=bootstrap,
            seed=seed,
            window=window,
            span=span,
            bin_size=bin_size,
            primary_last_batches=primary_last_batches,
            baseline_return=baseline_return,
        ),
        phases={},
    )
    provenance = {}
    for side, input_mode in enumerate(("actor_summary", "full_history")):
        trained = {pair[side]["trained_mode"] for pair in pairs}
        if len(trained) != 1:
            raise ValueError("Each analysis side must have exactly one trained_mode")
        provenance[input_mode] = dict(trained_mode=trained.pop(), input_mode=input_mode)
    provenance["comparison_kind"] = (
        "input_swap_diagnostic"
        if any(provenance[m]["trained_mode"] != m for m in ("actor_summary", "full_history"))
        else "matched_trained_critics"
    )
    result["provenance"] = provenance
    for phase in sorted({a["phase"] for a, _ in pairs}):
        rows = [p for p in pairs if p[0]["phase"] == phase]
        batches = sorted({a["batch"] for a, _ in rows})
        primary_batches = batches[-primary_last_batches:] if phase == "pre_update" else batches
        primary = [p for p in rows if p[0]["batch"] in primary_batches]
        comparisons = {
            name: paired_metrics(
                [p for p in primary if name == "all" or bucket(p[0]["segment_index"]) == name],
                bootstrap,
                seed,
                baseline_return,
            )
            for name in BUCKETS
        }
        boundaries, midpoints = boundary_events(primary, window=window)
        curves = [
            dict(batch=batch, **paired_metrics([p for p in rows if p[0]["batch"] == batch], 0, seed, baseline_return))
            for batch in batches
        ]
        removed = []
        available = [e for e in boundaries if e["removed_tokens"] is not None]
        for group in np.array_split(sorted(available, key=lambda e: e["removed_tokens"]), min(5, len(available)) or 1):
            if len(group):
                removed.append(
                    dict(
                        mean_removed_tokens=float(np.mean([e["removed_tokens"] for e in group])),
                        **event_metrics(list(group), bootstrap, seed),
                    )
                )
        result["phases"][phase] = dict(
            primary_batches=primary_batches,
            by_compactions=comparisons,
            curve=curves,
            at_compaction=event_metrics(boundaries, bootstrap, seed),
            at_midpoint=event_metrics(midpoints, bootstrap, seed),
            profile=dict(compaction=profile(boundaries, span, bin_size), midpoint=profile(midpoints, span, bin_size)),
            by_compaction_index={
                name: event_metrics([e for e in boundaries if bucket(e["index"]) == name], bootstrap, seed)
                for name in BUCKETS[2:]
            },
            by_removed_tokens=removed,
            variance_share=variance_share(primary),
            retention=dict(
                compaction=retention(boundaries, bootstrap, seed), midpoint=retention(midpoints, bootstrap, seed)
            ),
        )
    return result


def write_outputs(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    report = [
        "# Paired offline critic analysis",
        "",
        "The two inputs are paired exactly by episode, segment, and actor token position. "
        "Confidence intervals use paired resampling by problem; null indicates insufficient data "
        "or an undefined statistic.",
        "",
        "Training results use the last 30 batches (or the configured count); held-out results use all specified data. "
        "The constant baseline return is configured explicitly. Oracle constant MSE describes target variance, "
        "not an independent prediction.",
        "",
    ]
    provenance = result["provenance"]
    if provenance["comparison_kind"] == "input_swap_diagnostic":
        report += [
            "**This report is an input-swap diagnostic and cannot serve as the main comparison "
            "of two separately trained critics.**",
            "",
        ]
    for mode in ("actor_summary", "full_history"):
        report += [
            f"{mode} arm: trained input `{provenance[mode]['trained_mode']}`, "
            f"evaluated input `{provenance[mode]['input_mode']}`.",
            "",
        ]
    for phase, data in result["phases"].items():
        report += [
            f"## {phase}",
            "",
            "| Prior compactions | Actor EV | Full EV | EV difference 95% CI |",
            "|---|---:|---:|---|",
        ]
        fmt = lambda v: "null" if v is None else f"{v:.5f}"
        for name, stats in data["by_compactions"].items():
            ci = stats["full_minus_actor"]["ev"]
            report.append(
                f"| {name} | {fmt(stats['actor_summary']['ev'])} | {fmt(stats['full_history']['ev'])} | "
                f"[{fmt(ci['low'])}, {fmt(ci['high'])}] |"
            )
        jump = data["at_compaction"]["abs_jump"]
        report += [
            "",
            f"Absolute boundary jump: Actor {fmt(jump['actor_summary'])}, Full {fmt(jump['full_history'])}. "
            "Short smoke tests cannot establish model quality; that requires adequate task sampling "
            "and complete training.",
            "",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for mode, color in (("actor_summary", "tab:orange"), ("full_history", "tab:blue")):
            curve = data["curve"]
            axes[0].plot([r["batch"] for r in curve], [r[mode]["ev"] for r in curve], label=mode, color=color)
            axes[1].plot(
                BUCKETS[1:],
                [data["by_compactions"][k][mode]["ev"] for k in BUCKETS[1:]],
                marker="o",
                label=mode,
                color=color,
            )
            bins = data["profile"]["compaction"]["bins"]
            axes[2].plot(
                [(r["start"] + r["end"]) / 2 for r in bins],
                [r[mode]["absolute"] for r in bins],
                label=mode,
                color=color,
            )
        for ax, title, xlabel in zip(
            axes,
            ("Pre-update / held-out EV", "EV by prior compactions", "At-compaction deviation"),
            ("Batch", "Prior compactions", "Actor token offset"),
            strict=True,
        ):
            ax.set(title=title, xlabel=xlabel)
            ax.grid(alpha=0.2)
        axes[0].legend()
        for ax in axes[:2]:
            if not any(np.isfinite(line.get_ydata(orig=False)).any() for line in ax.lines):
                ax.text(
                    0.5,
                    0.5,
                    "EV undefined\n(constant returns or insufficient data)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        fig.savefig(output / f"{phase}.png", dpi=160)
        plt.close(fig)
        report += [f"![{phase}]({phase}.png)", ""]
    (output / "report.md").write_text("\n".join(report) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-summary", nargs="+", required=True)
    parser.add_argument("--full-history", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("all", "pre_update", "heldout"), default="all")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--span", type=int, default=256)
    parser.add_argument("--bin-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary-last-batches", type=int, default=30)
    parser.add_argument("--baseline-return", type=float, default=0.5)
    parser.add_argument(
        "--allow-input-swap-diagnostic",
        action="store_true",
        help="Explicitly label off-distribution input swaps; never call them the main trained comparison",
    )
    args = parser.parse_args()
    a = load_values(
        args.actor_summary,
        "actor_summary",
        phase=args.phase,
        window=args.window,
        span=args.span,
        allow_input_swap_diagnostic=args.allow_input_swap_diagnostic,
    )
    b = load_values(
        args.full_history,
        "full_history",
        phase=args.phase,
        window=args.window,
        span=args.span,
        allow_input_swap_diagnostic=args.allow_input_swap_diagnostic,
    )
    pairs = pair_records(a, b)
    if args.episodes:
        attach_episode_metadata(pairs, args.episodes)
    result = analyze(
        pairs,
        bootstrap=args.bootstrap,
        seed=args.seed,
        window=args.window,
        span=args.span,
        bin_size=args.bin_size,
        primary_last_batches=args.primary_last_batches,
        baseline_return=args.baseline_return,
    )
    write_outputs(result, args.out)
    print(f"Paired {len(pairs)} segments; wrote {args.out}/analysis.json, report.md and PNG figures")


if __name__ == "__main__":
    main()
