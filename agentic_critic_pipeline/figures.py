"""Render paired critic diagnostics with explicit, reusable numerical plot data."""

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze import (
    BUCKETS,
    analyze,
    attach_episode_metadata,
    boundary_events,
    bucket,
    correlation,
    load_values,
    pair_records,
    paired_metrics,
    profile,
    retention,
)
from .robustness import build_robustness

MODES = ("actor_summary", "full_history")
HISTOGRAM_EDGES = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def mean_interval(events, side, field, bootstrap, seed):
    groups = {}
    for event in events:
        groups.setdefault(event["problem_id"], []).append(event["sides"][side][field])
    if not groups:
        return dict(value=None, low=None, high=None, events=0, problems=0)
    sums = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap if len(groups) > 1 else 0):
        indices = rng.integers(len(groups), size=len(groups))
        draws.append(sums[indices].sum() / counts[indices].sum())
    return dict(
        value=float(sums.sum() / counts.sum()),
        low=float(np.quantile(draws, 0.025)) if draws else None,
        high=float(np.quantile(draws, 0.975)) if draws else None,
        events=int(counts.sum()),
        problems=len(groups),
    )


JUMP_ERROR_FIELDS = ("error_improvement", "sq_error_improvement", "abs_error_before", "abs_error_after")


def jump_row(events, bootstrap, seed, **labels):
    # Mode keys hold the absolute jump (what the plots draw); the error fields beside them
    # say whether that jump moved the value toward or away from the final return.
    row = dict(
        **labels, **{mode: mean_interval(events, i, "abs_jump", bootstrap, seed) for i, mode in enumerate(MODES)}
    )
    for field in JUMP_ERROR_FIELDS:
        row[field] = {mode: mean_interval(events, i, field, bootstrap, seed) for i, mode in enumerate(MODES)}
    return row


def retention_row(events, bootstrap, seed, **labels):
    result = retention(events, 0, seed)
    grouped = {}
    for event in events:
        grouped.setdefault((event["problem_id"], event["batch"]), []).append(event)
    grams = {}
    for (problem, _), group in grouped.items():
        if len(group) < 3:
            continue
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
        matrix -= matrix.mean(axis=0)
        grams[problem] = grams.get(problem, np.zeros((5, 5))) + matrix.T @ matrix
    samples = {mode: [] for mode in MODES}
    differences = []
    rng = np.random.default_rng(seed)
    blocks = np.asarray(list(grams.values()))
    for _ in range(bootstrap if len(blocks) > 1 else 0):
        gram = blocks[rng.integers(len(blocks), size=len(blocks))].sum(axis=0)
        changes = []
        for mode, before, after in ((MODES[0], 0, 1), (MODES[1], 2, 3)):

            def corr(column, gram=gram):
                norm = np.sqrt(max(0.0, gram[column, column] * gram[4, 4]))
                return None if norm <= 1e-14 else float(gram[column, 4] / norm)

            a, b = corr(before), corr(after)
            value = None if a is None or b is None else b - a
            changes.append(value)
            if value is not None:
                samples[mode].append(value)
        if all(x is not None for x in changes):
            differences.append(changes[1] - changes[0])
    for mode in MODES:
        values = samples[mode]
        result[mode]["low"] = float(np.quantile(values, 0.025)) if values else None
        result[mode]["high"] = float(np.quantile(values, 0.975)) if values else None
    result["change_full_minus_actor"].update(
        low=float(np.quantile(differences, 0.025)) if differences else None,
        high=float(np.quantile(differences, 0.975)) if differences else None,
        valid_resamples=len(differences),
    )
    return dict(**labels, **result)


def offset_events(events, offset, window):
    output = []
    for event in events:
        item = dict(event, sides=[])
        for side in event["sides"]:
            values = side["after"][offset : offset + window]
            after = float(values.mean())
            item["sides"].append(dict(side, after_mean=after))
        output.append(item)
    return output


def direction_rows(events, side, bootstrap, seed):
    ordered = sorted(events, key=lambda e: e["reward"] - e["sides"][side]["before_mean"])
    rows = []
    for group in np.array_split(np.asarray(ordered, dtype=object), min(5, len(ordered)) or 1):
        if not len(group):
            continue
        rows.append(
            dict(
                gap=float(np.mean([e["reward"] - e["sides"][side]["before_mean"] for e in group])),
                signed_jump=mean_interval(list(group), side, "signed_jump", bootstrap, seed),
            )
        )
    gaps = np.array([e["reward"] - e["sides"][side]["before_mean"] for e in events])
    jumps = np.array([e["sides"][side]["signed_jump"] for e in events])
    centered = gaps - gaps.mean() if len(gaps) else gaps
    return dict(
        bins=rows,
        events=len(events),
        before_mae=float(np.abs(gaps).mean()) if len(gaps) else None,
        after_mae=float(np.abs(gaps - jumps).mean()) if len(gaps) else None,
        fraction_closer=float((np.abs(gaps - jumps) < np.abs(gaps)).mean()) if len(gaps) else None,
        centered_slope=None if centered @ centered <= 1e-14 else float(centered @ jumps / (centered @ centered)),
        centered_correlation=correlation(centered, jumps - jumps.mean()) if len(jumps) else None,
    )


def build_figures(
    pairs,
    *,
    bootstrap=2000,
    seed=0,
    window=32,
    span=256,
    bin_size=16,
    primary_last_batches=30,
    moving_batches=8,
    offsets=(0, 128, 512, 2048),
):
    if min(bootstrap, window, span, bin_size, primary_last_batches, moving_batches) < 1:
        raise ValueError("Figure windows and bootstrap count must be positive")
    if not offsets or any(x < 0 for x in offsets):
        raise ValueError("Offsets must be nonnegative and nonempty")
    base = analyze(
        pairs,
        bootstrap=bootstrap,
        seed=seed,
        window=window,
        span=span,
        bin_size=bin_size,
        primary_last_batches=primary_last_batches,
    )
    result = dict(
        settings=dict(base["settings"], moving_batches=moving_batches, offsets=list(offsets)),
        provenance=base["provenance"],
        phases={},
    )
    for phase in base["phases"]:
        rows = [p for p in pairs if p[0]["phase"] == phase]
        batches = sorted({a["batch"] for a, _ in rows})
        primary = [p for p in rows if p[0]["batch"] in base["phases"][phase]["primary_batches"]]
        boundary, midpoint = boundary_events(primary, window=window)
        controls = {"compaction": boundary, "midpoint": midpoint}
        data = dict(
            primary_batches=base["phases"][phase]["primary_batches"],
            paired_metrics=base["phases"][phase]["by_compactions"],
            variance_share=base["phases"][phase]["variance_share"],
            ev_training_by_compaction=[],
            ev_training_comparison=base["phases"][phase]["curve"],
            jump_training=[],
            boundary_profile={},
            jump_by_compaction_index=[],
            jump_by_removed_tokens=[],
            jump_histogram={},
            jump_direction={},
            retention_by_compaction_index=[],
            retention_by_offset=[],
            episode_traces=[],
        )
        for start in range(0, len(batches), moving_batches):
            selected = batches[start : start + moving_batches]
            block = [p for p in rows if p[0]["batch"] in selected]
            for name in BUCKETS:
                subset = [p for p in block if name == "all" or bucket(p[0]["segment_index"]) == name]
                stats = paired_metrics(subset, 0, seed)
                data["ev_training_by_compaction"].append(
                    dict(
                        first_batch=selected[0],
                        last_batch=selected[-1],
                        bucket=name,
                        **{mode: stats[mode] for mode in MODES},
                    )
                )
            block_boundary, block_midpoint = boundary_events(block, window=window)
            for kind, events in (("compaction", block_boundary), ("midpoint", block_midpoint)):
                data["jump_training"].append(
                    jump_row(events, bootstrap, seed, kind=kind, first_batch=selected[0], last_batch=selected[-1])
                )
        for kind, events in controls.items():
            data["boundary_profile"][kind] = profile(events, span, bin_size)
            data["jump_histogram"][kind] = {}
            data["jump_direction"][kind] = {}
            for i, mode in enumerate(MODES):
                values = np.array([e["sides"][i]["abs_jump"] for e in events])
                counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES + [np.inf])
                data["jump_histogram"][kind][mode] = dict(
                    lower_edges=HISTOGRAM_EDGES,
                    upper_edges=HISTOGRAM_EDGES[1:] + [None],
                    counts=counts.tolist(),
                    proportions=(counts / len(values)).tolist() if len(values) else [None] * len(counts),
                    events=len(values),
                    **{
                        label: float(np.quantile(values, q)) if len(values) else None
                        for label, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99))
                    },
                )
                data["jump_direction"][kind][mode] = direction_rows(events, i, bootstrap, seed)
            for index in ("1", "2", "3", "4", "5+"):
                subset = [e for e in events if (str(e["index"]) if e["index"] < 5 else "5+") == index]
                data["jump_by_compaction_index"].append(jump_row(subset, bootstrap, seed, kind=kind, index=index))
            for index in ("1", "2", "3+"):
                subset = [e for e in events if (str(e["index"]) if e["index"] < 3 else "3+") == index]
                data["retention_by_compaction_index"].append(
                    retention_row(subset, bootstrap, seed, kind=kind, index=index)
                )
            eligible = [e for e in events if all(len(s["after"]) >= max(offsets) + window for s in e["sides"])]
            for offset in offsets:
                data["retention_by_offset"].append(
                    retention_row(
                        offset_events(eligible, offset, window),
                        bootstrap,
                        seed,
                        kind=kind,
                        offset=offset,
                        eligible_events=len(eligible),
                    )
                )
        available = sorted([e for e in boundary if e["removed_tokens"] is not None], key=lambda e: e["removed_tokens"])
        for group in np.array_split(np.asarray(available, dtype=object), min(5, len(available)) or 1):
            if len(group):
                data["jump_by_removed_tokens"].append(
                    jump_row(
                        list(group),
                        bootstrap,
                        seed,
                        mean_removed_tokens=float(np.mean([e["removed_tokens"] for e in group])),
                    )
                )
        data["removed_midpoint_control"] = jump_row(midpoint, bootstrap, seed)
        episodes = {}
        for pair in rows:
            episodes.setdefault((pair[0]["batch"], pair[0]["episode_id"]), []).append(pair)
        used_problems = set()
        for key in sorted(episodes, key=lambda k: (-k[0], k[1])):
            episode = sorted(episodes[key], key=lambda p: p[0]["segment_index"])
            first = episode[0][0]
            if len(episode) < 4 or first["problem_id"] in used_problems:
                continue
            trace = dict(
                episode_id=first["episode_id"],
                problem_id=first["problem_id"],
                batch=first["batch"],
                reward=first["reward"],
                boundaries=[],
                actor_summary=[],
                full_history=[],
            )
            offset = 0
            for pair in episode:
                if offset:
                    trace["boundaries"].append(offset)
                for i, mode in enumerate(MODES):
                    trace[mode].extend(
                        dict(start=t["start"] + offset, end=t["end"] + offset, value=t["value"])
                        for t in pair[i].get("trace", [])
                    )
                offset += int(pair[0]["stats"][0])
            data["episode_traces"].append(trace)
            used_problems.add(first["problem_id"])
            if len(used_problems) == 4:
                break
        data["robustness"] = build_robustness(
            primary,
            window=window,
            bootstrap=bootstrap,
            seed=seed,
            jump_row=jump_row,
            retention_row=retention_row,
        )
        result["phases"][phase] = data
    return result


def render(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    colors = {"actor_summary": "tab:orange", "full_history": "tab:blue"}

    def values(rows, mode, key="value"):
        return [np.nan if r[mode].get(key) is None else r[mode][key] for r in rows]

    def band(ax, x, rows, mode, key="value", **kwargs):
        y = values(rows, mode, key)
        ax.plot(x, y, color=colors[mode], label=mode, **kwargs)
        if rows and all("low" in r[mode] for r in rows):
            ax.fill_between(x, values(rows, mode, "low"), values(rows, mode, "high"), color=colors[mode], alpha=0.12)

    for phase, data in result["phases"].items():
        directory = output / phase
        directory.mkdir(exist_ok=True)

        def finish(fig, axes, name, phase=phase, directory=directory):
            for ax in np.asarray(axes, dtype=object).reshape(-1):
                ax.grid(alpha=0.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7)
                if not any(
                    not line.get_label().startswith("_")
                    and np.isfinite(np.asarray(line.get_ydata(), dtype=float)).any()
                    for line in ax.lines
                ):
                    if not any(np.isfinite(patch.get_height()) for patch in ax.patches):
                        ax.text(
                            0.5,
                            0.5,
                            "Undefined or insufficient eligible data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
            fig.suptitle(f"{phase}: {name.replace('_', ' ')}")
            fig.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for ax, name in zip(axes.flat, BUCKETS, strict=True):
            rows = [r for r in data["ev_training_by_compaction"] if r["bucket"] == name]
            for mode in MODES:
                band(ax, [r["last_batch"] for r in rows], rows, mode, "within_batch_ev", marker=".")
            ax.set(
                title=f"Prior compactions: {name}",
                xlabel="Batch (window end)",
                ylabel="Within-batch EV",
                yscale="symlog",
            )
        finish(fig, axes, "ev_training_by_compaction")
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        rows = data["ev_training_comparison"]
        for mode in MODES:
            band(ax, [r["batch"] for r in rows], rows, mode, "ev")
        ax.set(xlabel="Batch", ylabel="EV", yscale="symlog")
        finish(fig, [ax], "ev_training_comparison")

        for name in ("boundary_profile", "paired_boundary_profile"):
            fig, axes = plt.subplots(
                1, 2 if name == "boundary_profile" else 1, figsize=(12, 4), squeeze=False, constrained_layout=True
            )
            for kind_index, kind in enumerate(("compaction", "midpoint")):
                ax = axes[0, kind_index if name == "boundary_profile" else 0]
                rows = data["boundary_profile"][kind]["bins"]
                for mode in MODES:
                    ax.plot(
                        [(r["start"] + r["end"]) / 2 for r in rows],
                        [r[mode]["absolute"] for r in rows],
                        color=colors[mode],
                        ls="-" if kind == "compaction" else "--",
                        label=f"{mode}: {kind}",
                    )
                ax.axvline(0, color="gray", alpha=0.4)
                ax.set(xlabel="Actor token offset", ylabel="Absolute deviation from pre-event mean")
            finish(fig, axes, name)
        for name, key, xlabel in (
            ("jump_by_compaction_index", "index", "Compaction index"),
            ("retention_by_compaction_index", "index", "Compaction index"),
            ("retention_by_offset", "offset", "Post-event actor token offset"),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for ax, kind in zip(axes, ("compaction", "midpoint"), strict=True):
                rows = [r for r in data[name] if r["kind"] == kind]
                x = list(range(len(rows)))
                for mode in MODES:
                    band(
                        ax,
                        x,
                        rows,
                        mode,
                        "reward_correlation_change" if name.startswith("retention") else "value",
                        marker="o",
                    )
                ax.set_xticks(x, [str(r[key]) for r in rows])
                ax.set(
                    title=kind,
                    xlabel=xlabel,
                    ylabel="Reward correlation change" if name.startswith("retention") else "Mean absolute jump",
                )
            finish(fig, axes, name)
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        rows = data["jump_by_removed_tokens"]
        for mode in MODES:
            band(ax, [r["mean_removed_tokens"] for r in rows], rows, mode, marker="o")
            control = data["removed_midpoint_control"][mode]["value"]
            if control is not None:
                ax.axhline(control, color=colors[mode], ls="--", label=f"{mode}: midpoint")
        ax.set(xlabel="Mean tokens removed (event quintile)", ylabel="Mean absolute jump")
        finish(fig, [ax], "jump_by_removed_tokens")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
        for ax, kind in zip(axes, ("compaction", "midpoint"), strict=True):
            for i, mode in enumerate(MODES):
                rows = data["jump_histogram"][kind][mode]
                heights = [np.nan if x is None else x for x in rows["proportions"]]
                ax.bar(np.arange(len(heights)) + i * 0.35, heights, width=0.35, color=colors[mode], label=mode)
            ax.set_xticks(np.arange(len(HISTOGRAM_EDGES)) + 0.175, [str(x) + "+" for x in HISTOGRAM_EDGES], rotation=45)
            ax.set(title=kind, xlabel="Absolute jump bin lower edge", ylabel="Event proportion")
        finish(fig, axes, "jump_histogram")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, kind in zip(axes, ("compaction", "midpoint"), strict=True):
            for mode in MODES:
                bins = data["jump_direction"][kind][mode]["bins"]
                wrapped = [{mode: r["signed_jump"]} for r in bins]
                band(ax, [r["gap"] for r in bins], wrapped, mode, marker="o")
            ax.set(title=kind, xlabel="Return minus pre-event value (quintiles)", ylabel="Signed jump")
        finish(fig, axes, "jump_direction")
        for name in ("jump_training", "paired_jump_training"):
            fig, axes = plt.subplots(
                1, 2 if name == "jump_training" else 1, figsize=(12, 4), squeeze=False, constrained_layout=True
            )
            for i, mode in enumerate(MODES):
                ax = axes[0, i if name == "jump_training" else 0]
                for kind in ("compaction", "midpoint"):
                    rows = [r for r in data["jump_training"] if r["kind"] == kind]
                    ax.plot(
                        [r["last_batch"] for r in rows],
                        values(rows, mode),
                        color=colors[mode],
                        ls="-" if kind == "compaction" else "--",
                        label=f"{mode}: {kind}",
                    )
                ax.set(xlabel="Batch (window end)", ylabel="Mean absolute jump")
                if any(np.any(np.asarray(line.get_ydata()) > 0) for line in ax.lines):
                    ax.set_yscale("log")
            finish(fig, axes, name)
        for name, grouping, xkey in (
            ("window_sensitivity", "window", "index"),
            ("training_halves", "half", "index"),
            ("same_episodes", "kind", "index"),
            ("retention_by_removed", "first_only", "mean_removed_tokens"),
            ("jump_by_cumulative_removed", "first_only", "mean_removed_tokens"),
        ):
            rows = data["robustness"][name]
            groups = list(dict.fromkeys(r[grouping] for r in rows)) or [None]
            fig, axes = plt.subplots(
                1, len(groups), figsize=(5 * len(groups), 4), squeeze=False, constrained_layout=True
            )
            for ax, group in zip(axes.flat, groups, strict=True):
                selected = [r for r in rows if r[grouping] == group and r.get("kind", "compaction") == "compaction"]
                x = [r[xkey] for r in selected]
                for mode in MODES:
                    band(
                        ax,
                        x,
                        selected,
                        mode,
                        "value" if name == "jump_by_cumulative_removed" else "reward_correlation_change",
                        marker="o",
                    )
                ax.set(
                    title=f"{grouping}: {group}",
                    xlabel=xkey.replace("_", " "),
                    ylabel="Mean absolute jump"
                    if name == "jump_by_cumulative_removed"
                    else "Reward correlation change",
                )
            finish(fig, axes, name)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
        for ax, trace in zip(axes.flat, data["episode_traces"], strict=False):
            for mode in MODES:
                ax.plot(
                    [(r["start"] + r["end"]) / 2 for r in trace[mode]],
                    [r["value"] for r in trace[mode]],
                    color=colors[mode],
                    label=mode,
                )
            for boundary in trace["boundaries"]:
                ax.axvline(boundary, color="gray", alpha=0.4)
            ax.axhline(trace["reward"], color="black", ls=":", label="Final return")
            ax.set(title=trace["episode_id"], xlabel="Cumulative actor tokens", ylabel="Value")
        finish(fig, axes, "episode_traces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-summary", nargs="+", required=True)
    parser.add_argument("--full-history", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("all", "pre_update", "heldout"), default="all")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--span", type=int, default=256)
    parser.add_argument("--bin-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary-last-batches", type=int, default=30)
    parser.add_argument("--moving-batches", type=int, default=8)
    parser.add_argument("--offsets", nargs="+", type=int, default=[0, 128, 512, 2048])
    parser.add_argument("--trace-bin", type=int, default=256)
    parser.add_argument("--allow-input-swap-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.trace_bin < 1 or min(args.offsets) < 0:
        parser.error("trace-bin must be positive and offsets nonnegative")
    records = [
        load_values(
            paths,
            mode,
            phase=args.phase,
            window=args.window,
            span=max(args.span, max(args.offsets) + args.window, 1024),
            trace_bin=args.trace_bin,
            allow_input_swap_diagnostic=args.allow_input_swap_diagnostic,
        )
        for paths, mode in ((args.actor_summary, MODES[0]), (args.full_history, MODES[1]))
    ]
    pairs = pair_records(*records)
    attach_episode_metadata(pairs, args.episodes)
    result = build_figures(
        pairs,
        bootstrap=args.bootstrap,
        seed=args.seed,
        window=args.window,
        span=args.span,
        bin_size=args.bin_size,
        primary_last_batches=args.primary_last_batches,
        moving_batches=args.moving_batches,
        offsets=args.offsets,
    )
    result["settings"]["trace_bin"] = args.trace_bin
    render(result, args.out)


if __name__ == "__main__":
    main()
