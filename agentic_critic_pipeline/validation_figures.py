"""Focused, reproducible critic comparison; saved 100-token means stay intact."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from agentic_critic_pipeline.validation import load_plan
from agentic_critic_pipeline.validation_metrics import ValidationMetrics
from agentic_critic_pipeline.validation_storage import validate_compact_episode

MODES = ("full_history", "actor_summary")
COLORS = {"full_history": "#0072B2", "actor_summary": "#D55E00", "attention_matching": "#009E73"}
LABELS = {"full_history": "Full history", "actor_summary": "Actor summary", "attention_matching": "Attention matching"}


def _load(root):
    plan = load_plan(root / "validation_replay/validation_plan.json")
    results = {}
    for mode in MODES:
        rows = [
            json.loads(p.read_text()) for p in sorted((root / mode / "train/validation").glob("batch_*/metrics.json"))
        ]
        if not rows or len({r["batch"] for r in rows}) != len(rows):
            raise ValueError(f"Missing or duplicated validation metrics for {mode}")
        for row in rows:
            if row["plan_hash"] != plan["plan_hash"] or row["mode"] != mode:
                raise ValueError("Cannot compare different frozen validation cohorts or modes")
            if row["episodes"] != len(plan["episodes"]):
                raise ValueError("Incomplete validation metrics")
        config = json.loads((root / mode / "train/miles_run.json").read_text())
        expected = list(
            range(config["validation_interval"], config["logical_batches"] + 1, config["validation_interval"])
        )
        if [r["batch"] for r in rows] != expected:
            raise ValueError(f"Missing scheduled validation checkpoints for {mode}")
        results[mode] = rows
    if [r["batch"] for r in results[MODES[0]]] != [r["batch"] for r in results[MODES[1]]]:
        raise ValueError("Comparison requires paired validation checkpoints")
    # Boundary figures use the final checkpoint. Stream its saved values rather
    # than storing a second per-event copy of every L1 curve during training.
    for mode, rows in results.items():
        rows[-1]["profiles"] = _rebuild_profiles(root, plan, mode=mode, metrics=rows[-1])
    for left, right in zip(results[MODES[0]], results[MODES[1]], strict=True):
        left_counts = {key: value["tokens"] for key, value in left["ev"].items()}
        right_counts = {key: value["tokens"] for key, value in right["ev"].items()}
        left_profiles = {key: value["event_counts"] for key, value in left["profiles"].items()}
        right_profiles = {key: value["event_counts"] for key, value in right["profiles"].items()}
        if left_counts != right_counts or left_profiles != right_profiles:
            raise ValueError("Unpaired token/event cohorts across critic modes")
    return plan, results


def _rebuild_profiles(root, plan, *, mode, metrics, validation_dir="train/validation"):
    batch = metrics["batch"]
    stats = ValidationMetrics(plan, mode=mode, batch=batch)
    seen = set()
    path = root / mode / validation_dir / f"batch_{batch:04d}/values.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        for episode in map(json.loads, stream):
            identity = episode["episode_id"]
            if identity in seen or identity not in plan["episodes"]:
                raise ValueError("Duplicate or unexpected trajectory in saved validation values")
            seen.add(identity)
            expected = plan["episodes"][identity]
            if episode["mode"] != mode or episode["batch"] != batch:
                raise ValueError("Saved validation values have the wrong mode or batch")
            if any(episode[key] != expected[key] for key in ("problem_id", "reward", "n_compactions")):
                raise ValueError("Saved validation values differ from the frozen cohort")
            validate_compact_episode(episode, expected)
            stats.add(episode)
    if seen != plan["episodes"].keys():
        raise ValueError("Saved validation values do not cover the complete frozen cohort")
    rebuilt = stats.summary()
    for key in ("episodes", "ev", "jumps", "compaction_events", "ineligible_events"):
        if rebuilt[key] != metrics[key]:
            raise ValueError(f"Saved validation values disagree with recorded {key}")
    return rebuilt["profiles"]


def _finish(fig, axes, output, name):
    for ax in np.asarray(axes, dtype=object).flat:
        if not ax.get_visible():
            continue
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=10)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    # Avoid keeping hundreds of trace figures alive.
    import matplotlib.pyplot as plt

    plt.close(fig)


def _ev_figures(plt, results, output):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for mode, rows in results.items():
        ax.plot(
            [r["batch"] for r in rows],
            [r["ev"]["all"]["ev"] for r in rows],
            "o-",
            color=COLORS[mode],
            label=LABELS[mode],
            linewidth=2,
        )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(title="Frozen validation explained variance", xlabel="Completed training batches", ylabel="EV")
    _finish(fig, [ax], output, "validation_ev")
    for grouping, label in [("prior", "Prior compactions at token"), ("total", "Total compactions in trajectory")]:
        keys = sorted(
            {k for rows in results.values() for r in rows for k in r["ev"] if k.startswith(grouping + "/")},
            key=lambda k: int(k.split("/")[1]),
        )
        fig, axes = plt.subplots(
            (len(keys) + 2) // 3, 3, figsize=(15, 4 * ((len(keys) + 2) // 3)), squeeze=False, layout="constrained"
        )
        for ax, key in zip(axes.flat, keys, strict=False):
            for mode, rows in results.items():
                values = [r["ev"].get(key, {}).get("ev") for r in rows]
                ax.plot([r["batch"] for r in rows], values, "o-", color=COLORS[mode], label=LABELS[mode])
                count = rows[-1]["ev"].get(key, {}).get("tokens", 0)
            ax.set(
                title=f'{label}: {key.split("/")[1]}\n{count:,} evaluated tokens',
                xlabel="Completed training batches",
                ylabel="EV",
            )
            ax.axhline(0, color="gray", linewidth=0.8)
            if all(r["ev"].get(key, {}).get("ev") is None for rows in results.values() for r in rows):
                ax.text(0.5, 0.5, "Undefined EV: zero target variance", ha="center", transform=ax.transAxes)
        for ax in list(axes.flat)[len(keys) :]:
            ax.set_visible(False)
        _finish(fig, axes, output, f"ev_by_{grouping}_compaction_count")


def _profile_figures(plt, results, output):
    latest = {mode: rows[-1] for mode, rows in results.items()}
    keys = ["all"] + sorted(
        {k for r in latest.values() for k in r["profiles"] if k.startswith("index/")},
        key=lambda k: int(k.split("/")[1]),
    )
    fig, axes = plt.subplots(
        len(keys),
        2,
        figsize=(15, 4 * len(keys)),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": [3, 1]},
    )
    for (ax, count_ax), key in zip(axes, keys, strict=True):
        for mode, row in latest.items():
            profile = row["profiles"].get(key)
            if profile is None:
                continue
            x = profile["offsets"] + [row["profile_span"]]
            ax.step(
                x,
                profile["mean_l1"] + profile["mean_l1"][-1:],
                where="post",
                color=COLORS[mode],
                label=LABELS[mode],
                linewidth=1.5,
            )
            count_ax.step(
                x,
                profile["event_counts"] + profile["event_counts"][-1:],
                where="post",
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.axvspan(-1024, 0, alpha=0.1, color="gray", label="Baseline window")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(
            title=f'Batch {next(iter(latest.values()))["batch"]}: {key}',
            xlabel="Actor tokens relative to compaction",
            ylabel="Mean |100-token mean V − baseline|",
            xlim=(-3076, 3075),
        )
        count_ax.set(
            title="Eligible events at each offset", xlabel="Token offset", ylabel="Event count", ylim=(0, None)
        )
        count_ax.yaxis.get_major_locator().set_params(integer=True)
        if not any(key in r["profiles"] for r in latest.values()):
            ax.text(0.5, 0.5, "No events with 1024 baseline tokens", ha="center", transform=ax.transAxes)
    _finish(fig, axes, output, "compaction_l1_profiles")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for i, (grouping, xlabel) in enumerate(
        [("index", "Compaction ordinal"), ("total", "Total trajectory compactions")]
    ):
        for j, (measure, ylabel) in enumerate(
            [
                ("first_token", "|First post value − baseline|"),
                ("mean_shift_1024", "|Mean post 1024 values − baseline|"),
            ]
        ):
            ax = axes[i, j]
            for mode, row in latest.items():
                keys = sorted(
                    [k for k in row["jumps"] if k.startswith(grouping + "/") and k.endswith("/" + measure)],
                    key=lambda k: int(k.split("/")[1]),
                )
                x = [int(k.split("/")[1]) for k in keys]
                data = [row["jumps"][k] for k in keys]
                ax.plot(x, [r["mean"] for r in data], "o-", color=COLORS[mode], label=LABELS[mode] + " mean")
                ax.fill_between(
                    x,
                    [r["p25"] for r in data],
                    [r["p75"] for r in data],
                    color=COLORS[mode],
                    alpha=0.15,
                    label="Event IQR (not CI)",
                )
                if mode == MODES[0]:
                    for position, r in zip(x, data, strict=True):
                        ax.annotate(
                            f'n={r["events"]}',
                            (position, r["mean"]),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            fontsize=9,
                        )
            ax.set(xlabel=xlabel, ylabel=ylabel, title="Compaction jump magnitude")
            ax.xaxis.get_major_locator().set_params(integer=True)
    _finish(fig, axes, output, "compaction_jump_by_count")


def _trace_figures(plt, root, plan, results, output, *, validation_dir="train/validation"):
    modes = tuple(results)
    batch = results[modes[0]][-1]["batch"]
    traces = {}
    for mode in modes:
        path = root / mode / validation_dir / f"batch_{batch:04d}/selected_trajectories.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            traces[mode] = {row["episode_id"]: row for row in map(json.loads, stream)}
        if traces[mode].keys() != set(plan["selected_episode_ids"]):
            raise ValueError("Selected trajectory records do not match the preselected IDs")
    trace_output = output / "trajectories"
    trace_output.mkdir(exist_ok=True)
    for number, identity in enumerate(plan["selected_episode_ids"], 1):
        fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
        for mode in modes:
            row = traces[mode][identity]
            ax.stairs(
                row["value_prediction"],
                [row["actor_token_ranges"][0][0]] + [end for _, end in row["actor_token_ranges"]],
                baseline=None,
                color=COLORS[mode],
                label=LABELS[mode],
                linewidth=1,
                alpha=0.85,
                rasterized=True,
            )
        row = traces[modes[0]][identity]
        for mode in modes[1:]:
            other = traces[mode][identity]
            if any(row[key] != other[key] for key in ("value_target", "compaction_boundaries", "actor_token_ranges")):
                raise ValueError("Unpaired trajectory targets/boundaries")
        ax.plot(
            [row["actor_token_ranges"][0][0], row["actor_token_ranges"][-1][1]],
            [row["value_target"][0], row["value_target"][-1]],
            color="black",
            linestyle="--",
            label="Observed final return G",
            linewidth=1.5,
        )
        for index, boundary in enumerate(row["compaction_boundaries"], 1):
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)
            ax.text(boundary, 1, str(index), transform=ax.get_xaxis_transform(), va="top", fontsize=9)
        ax.set(
            title=f"Trajectory {number:02d} · batch {batch} · {identity}",
            xlabel="Cumulative generated actor tokens",
            ylabel="100-token mean value / return target",
        )
        _finish(fig, [ax], trace_output, f"trajectory_{number:02d}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    # Plotting is optional during training; required only when rendering the report.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "pdf.fonttype": 42})
    plan, results = _load(args.output_root)
    output = args.output_root / "validation_report"
    output.mkdir(exist_ok=True)
    (output / "plot_data.json").write_text(json.dumps(results, allow_nan=False) + "\n")
    _ev_figures(plt, results, output)
    _profile_figures(plt, results, output)
    _trace_figures(plt, args.output_root, plan, results, output)


if __name__ == "__main__":
    main()
