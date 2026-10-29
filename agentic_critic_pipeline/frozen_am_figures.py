"""Render the frozen-checkpoint ablation with the training experiment's figures."""

import argparse
import json
from pathlib import Path

from agentic_critic_pipeline.validation import load_plan
from agentic_critic_pipeline.validation_figures import (
    COLORS,
    LABELS,
    _ev_figures,
    _finish,
    _profile_figures,
    _rebuild_profiles,
    _trace_figures,
)


def load_results(root):
    run = json.loads((root / "frozen_run.json").read_text())
    plan = load_plan(root / "validation_plan.json")
    if run["status"] != "complete" or run["plan_hash"] != plan["plan_hash"] or run["trained_mode"] != "full_history":
        raise ValueError("Frozen evaluation is incomplete or has inconsistent provenance")
    if run["modes"] != ["full_history", "actor_summary", "attention_matching"]:
        raise ValueError("Frozen evaluation requires all three input conditions")
    results = {}
    for mode in run["modes"]:
        results[mode] = []
        for checkpoint in run["checkpoints"]:
            path = root / mode / f"validation/batch_{checkpoint['batch']:04d}/metrics.json"
            row = json.loads(path.read_text())
            if (
                row["plan_hash"] != plan["plan_hash"]
                or row["mode"] != mode
                or row["batch"] != checkpoint["batch"]
                or row["checkpoint_sha256"] != checkpoint["sha256"]
                or row["trained_mode"] != "full_history"
            ):
                raise ValueError("Unpaired validation plans, checkpoints or input modes")
            if row["episodes"] != len(plan["episodes"]):
                raise ValueError("Incomplete validation cohort")
            # Recheck all stored values, including early checkpoints, before plotting.
            row["profiles"] = _rebuild_profiles(root, plan, mode=mode, metrics=row, validation_dir="validation")
            results[mode].append(row)
    for index in range(len(run["checkpoints"])):
        reference = results["full_history"][index]
        for mode in run["modes"][1:]:
            other = results[mode][index]
            if {k: v["tokens"] for k, v in reference["ev"].items()} != {
                k: v["tokens"] for k, v in other["ev"].items()
            }:
                raise ValueError("Unpaired validation token cohorts")
    return plan, run, results


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan, run, results = load_results(root)
    output = root / "validation_report"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    (output / "plot_data.json").write_text(json.dumps(dict(experiment=run, results=results), allow_nan=False) + "\n")
    _ev_figures(plt, results, output)
    _profile_figures(plt, results, output)
    _trace_figures(plt, root, plan, results, output, validation_dir="validation")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for mode, rows in results.items():
        batches = [r["batch"] for r in rows]
        axes[0].plot(
            batches, [r["timing"]["sum_rank_seconds"] for r in rows], "o-", label=LABELS[mode], color=COLORS[mode]
        )
        axes[1].plot(
            batches,
            [r["timing"]["peak_allocated_bytes"] / 2**30 for r in rows],
            "o-",
            label=LABELS[mode],
            color=COLORS[mode],
        )
    axes[0].set(
        xlabel="Checkpoint completed training batches",
        ylabel="Sum of rank evaluation seconds",
        title="Frozen validation forward time",
    )
    axes[1].set(
        xlabel="Checkpoint completed training batches",
        ylabel="Peak CUDA allocated GiB (0 on CPU)",
        title="Frozen validation peak memory",
    )
    _finish(fig, axes, output, "evaluation_cost")
    (output / "README.txt").write_text(
        "Same frozen full-history checkpoint, three inference inputs. No AM training.\n"
        "EV uses exact token moments; displayed values use the original segment-local 100-token means.\n"
        "Timing includes forward, cache construction and compaction, excludes model loading and plotting.\n"
        "Timing is validation-only, not an estimate of training overhead. GPU timings synchronize at episode boundaries.\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    render(args.output_root)


if __name__ == "__main__":
    main()
