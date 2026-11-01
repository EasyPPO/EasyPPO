"""CPU-only analysis of saved trajectory/segment gradients; no model inference."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic_critic_pipeline.records import data_sha256

MODES = ("actor_summary", "full_history")
LABELS = {"actor_summary": "Actor-summary critic", "full_history": "Full-history critic"}
METRICS = {
    "segment_grad_norm": "Segment contribution gradient norm",
    "cosine_with_trajectory": "Cosine(segment, trajectory)",
    "norm_ratio": "Contribution norm / trajectory norm",
    "segment_mean_grad_norm": "Segment token-mean gradient norm",
    "segment_mean_norm_ratio": "Token-mean norm / trajectory norm",
    "token_fraction": "Segment fraction of trajectory actor tokens",
}


def summarize(rows, metric, *, seed, bootstrap):
    """Equal episode weights within problem, then equal problem weights.

    This prevents long trajectories with many segments and prolific problems
    from dominating a cell. Bootstrap whole problems, never individual segments.
    """
    by_episode = defaultdict(list)
    problems = {}
    for row in rows:
        value = row[metric]
        if value is not None and math.isfinite(value):
            by_episode[row["episode_id"]].append(value)
            problems[row["episode_id"]] = row["problem_id"]
    by_problem = defaultdict(list)
    for eid, values in by_episode.items():
        by_problem[problems[eid]].append(float(np.mean(values)))
    means = np.array([np.mean(by_problem[key]) for key in sorted(by_problem)])
    ci = None
    if len(means) > 1:
        rng = np.random.default_rng(seed)
        samples = [float(rng.choice(means, len(means), replace=True).mean()) for _ in range(bootstrap)]
        ci = np.quantile(samples, [0.025, 0.975]).tolist()
    return dict(
        mean=float(means.mean()) if len(means) else None,
        ci95=ci,
        records=len(rows),
        valid=sum(map(len, by_episode.values())),
        undefined=len(rows) - sum(map(len, by_episode.values())),
        trajectories=len(by_episode),
        problems=len(by_problem),
    )


def _group(rows, count, metrics, seed, bootstrap):
    groups = defaultdict(list)
    for row in rows:
        groups[row[count]].append(row)
    return [
        dict(
            count=value,
            metrics={metric: summarize(group, metric, seed=seed, bootstrap=bootstrap) for metric in metrics},
        )
        for value, group in sorted(groups.items())
    ]


def _draw(ax, groups, metric, label):
    defined = [row for row in groups if row["metrics"][metric]["mean"] is not None]
    x = [row["count"] for row in defined]
    y = [row["metrics"][metric]["mean"] for row in defined]
    (line,) = ax.plot(x, y, "o-", label=label, markersize=4)
    ci = [row for row in defined if row["metrics"][metric]["ci95"] is not None]
    if ci:
        ax.fill_between(
            [row["count"] for row in ci],
            [row["metrics"][metric]["ci95"][0] for row in ci],
            [row["metrics"][metric]["ci95"][1] for row in ci],
            color=line.get_color(),
            alpha=0.18,
        )
    ax.grid(alpha=0.2)
    ax.xaxis.get_major_locator().set_params(integer=True)


def _save(fig, report, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(report / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _paired(left, right, *, segment):
    def key(row):
        return (row["episode_id"], row["segment_index"]) if segment else row["episode_id"]

    maps = [{key(row): row for row in rows} for rows in (left, right)]
    if any(len(mapping) != len(rows) for mapping, rows in zip(maps, (left, right), strict=True)):
        raise ValueError("Duplicate analysis records")
    if maps[0].keys() != maps[1].keys():
        raise ValueError("The two critics have different trajectory/segment coverage")
    metrics = list(METRICS) if segment else ["trajectory_grad_norm", "loss_tokens"]
    result = []
    for identity, a in maps[0].items():
        b = maps[1][identity]
        for field in ("problem_id", "total_compactions", "loss_tokens", "context_tokens"):
            if a[field] != b[field]:
                raise ValueError("The two critics have different cohort metadata")
        row = {field: a[field] for field in ("episode_id", "problem_id", "total_compactions")}
        if segment:
            row.update(segment_index=a["segment_index"], prior_compactions=a["prior_compactions"])
        row.update(
            {
                metric: a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
                for metric in metrics
            }
        )
        result.append(row)
    return result


def _read(root, mode, name):
    with (root / mode / f"{name}.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def render(root, *, seed=1234, bootstrap=2000):
    if bootstrap < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["results_sha256"].items():
            if data_sha256(root / name) != digest:
                raise ValueError("Saved measurement checksum mismatch")
    segments = {mode: _read(root, mode, "segments") for mode in MODES}
    trajectories = {mode: _read(root, mode, "trajectories") for mode in MODES}
    paired_segments = _paired(*[segments[mode] for mode in MODES], segment=True)
    paired_trajectories = _paired(*[trajectories[mode] for mode in MODES], segment=False)
    report = root / "gradient_report"
    report.mkdir(exist_ok=True)
    plot_data = dict(
        schema_version=1,
        seed=seed,
        bootstrap=bootstrap,
        weighting="equal episodes within each problem, then equal problems; bootstrap problems",
        paired_difference="actor_summary minus full_history on identical episodes/segments",
        groups={},
        paired={},
    )
    for count in ("prior_compactions", "total_compactions"):
        groups = {mode: _group(segments[mode], count, METRICS, seed, bootstrap) for mode in MODES}
        plot_data["groups"][count] = groups
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
        for ax, (metric, title) in zip(axes.flat, METRICS.items(), strict=True):
            for mode in MODES:
                _draw(ax, groups[mode], metric, LABELS[mode])
            ax.set(title=title, xlabel=count.replace("_", " ").capitalize())
            if metric == "cosine_with_trajectory":
                ax.axhline(0, color="gray", linewidth=0.7)
                ax.set_ylim(-1.05, 1.05)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Actor gradients by compaction count | problem-balanced means and 95% problem bootstrap CI")
        _save(fig, report, f"segments_by_{count}")
        differences = _group(paired_segments, count, METRICS, seed, bootstrap)
        plot_data["paired"][count] = differences
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, metric in zip(axes, list(METRICS)[:3], strict=True):
            _draw(ax, differences, metric, "Actor-summary minus full-history")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set(title=METRICS[metric], xlabel=count.replace("_", " ").capitalize(), ylabel="Paired difference")
        fig.suptitle("Actor-summary minus full-history | paired means and 95% problem bootstrap CI")
        _save(fig, report, f"paired_by_{count}")
    trajectory_metrics = {
        "trajectory_grad_norm": "Trajectory gradient norm",
        "loss_tokens": "Actor tokens per trajectory",
    }
    groups = {
        mode: _group(trajectories[mode], "total_compactions", trajectory_metrics, seed, bootstrap) for mode in MODES
    }
    plot_data["groups"]["trajectories_by_total_compactions"] = groups
    plot_data["paired"]["trajectories_by_total_compactions"] = _group(
        paired_trajectories,
        "total_compactions",
        trajectory_metrics,
        seed,
        bootstrap,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (metric, title) in zip(axes, trajectory_metrics.items(), strict=True):
        for mode in MODES:
            _draw(ax, groups[mode], metric, LABELS[mode])
        ax.set(title=title, xlabel="Total trajectory compactions")
    axes[0].legend(fontsize=8)
    fig.suptitle("Whole-trajectory actor gradients | 95% problem bootstrap CI")
    _save(fig, report, "trajectories_by_total_compactions")
    # The conditional grid separates segment position from trajectory length.
    conditional = {}
    for mode in MODES:
        cells = defaultdict(list)
        for row in segments[mode]:
            cells[(row["prior_compactions"], row["total_compactions"])].append(row)
        conditional[mode] = [
            dict(
                prior_compactions=k[0],
                total_compactions=k[1],
                metrics={
                    metric: summarize(rows, metric, seed=seed, bootstrap=bootstrap)
                    for metric in ("cosine_with_trajectory", "norm_ratio")
                },
            )
            for k, rows in sorted(cells.items())
        ]
    plot_data["conditional_by_prior_and_total"] = conditional
    for name, rows in (("paired_segments", paired_segments), ("paired_trajectories", paired_trajectories)):
        with (report / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    for mode in MODES:
        for name, rows in (("segments", segments[mode]), ("trajectories", trajectories[mode])):
            with (report / f"{mode}_{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        mode: dict(
            trajectories=len(trajectories[mode]),
            segments=len(segments[mode]),
            zero_trajectory_gradients=sum(row["trajectory_grad_norm"] == 0 for row in trajectories[mode]),
            zero_segment_gradients=sum(row["segment_grad_norm"] == 0 for row in segments[mode]),
            negative_segment_cosines=sum(
                row["cosine_with_trajectory"] is not None and row["cosine_with_trajectory"] < 0
                for row in segments[mode]
            ),
            metrics={metric: summarize(segments[mode], metric, seed=seed, bootstrap=bootstrap) for metric in METRICS},
            trajectory_norm=summarize(trajectories[mode], "trajectory_grad_norm", seed=seed, bootstrap=bootstrap),
        )
        for mode in MODES
    }
    summary["paired_segment_differences"] = {
        metric: summarize(paired_segments, metric, seed=seed, bootstrap=bootstrap) for metric in METRICS
    }
    summary["interpretation"] = [
        "Negative cosine indicates a segment opposing its own trajectory gradient; self-contribution is included.",
        "Norm ratios may exceed one when segment gradients cancel; they are not additive percentages.",
        "Compare token-mean norms and token fractions to assess segment-length effects.",
        "Prior-count curves condition on trajectories reaching that segment; total-count curves are observational.",
        "Checkpoint training and advantage magnitudes can affect differences; these are not causal compaction effects.",
        "Undefined zero-denominator metrics are retained as null and excluded only from that metric's mean.",
        "A single problem has no bootstrap confidence interval; cell counts are saved in plot_data.json.",
    ]
    for name, value in (("plot_data", plot_data), ("summary", summary)):
        (report / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args(argv)
    manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
    render(
        args.output_root,
        seed=args.seed if args.seed is not None else manifest["arguments"]["seed"],
        bootstrap=args.bootstrap if args.bootstrap is not None else manifest["arguments"]["bootstrap"],
    )


if __name__ == "__main__":
    main()
