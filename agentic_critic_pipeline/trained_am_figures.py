"""Local figures for independently HF-trained full/shared/AM critics."""

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
    run = json.loads((root / "training_run.json").read_text())
    plan = load_plan(root / "validation_plan.json")
    if (
        run["experiment"] != "trained_am_comparison"
        or run["backend"] != "hf_replicated"
        or run["status"] != "complete"
        or run["plan_hash"] != plan["plan_hash"]
    ):
        raise ValueError("Incomplete or mismatched trained-AM experiment")
    results, training = {}, {}
    for mode in run["modes"]:
        completed = json.loads((root / mode / "complete.json").read_text())
        log = json.loads((root / mode / "training_log.json").read_text())
        if completed != dict(mode=mode, steps=run["steps"], optimizer_updates=run["steps"] * run["mini_batches"]):
            raise ValueError("Incomplete trained critic")
        if (
            log["mode"] != mode
            or log["backend"] != run["backend"]
            or [r["batch"] for r in log["steps"]] != list(range(1, run["steps"] + 1))
        ):
            raise ValueError("Training log does not cover all requested replay batches")
        training[mode] = log["steps"]
        results[mode] = []
        for batch in run["validation_batches"]:
            row = json.loads((root / mode / f"validation/batch_{batch:04d}/metrics.json").read_text())
            if (
                row["mode"] != mode
                or row["trained_mode"] != mode
                or row["batch"] != batch
                or row["plan_hash"] != plan["plan_hash"]
                or row["backend"] != run["backend"]
                or row["episodes"] != len(plan["episodes"])
            ):
                raise ValueError("Validation does not match trained mode, batch or frozen cohort")
            row["profiles"] = _rebuild_profiles(root, plan, mode=mode, metrics=row, validation_dir="validation")
            results[mode].append(row)
    for index in range(len(run["validation_batches"])):
        cohorts = [{k: v["tokens"] for k, v in rows[index]["ev"].items()} for rows in results.values()]
        if any(cohort != cohorts[0] for cohort in cohorts):
            raise ValueError("Unpaired validation token cohorts")
    return plan, run, results, training


def _training_figures(plt, training, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for mode, rows in training.items():
        updates = [update for row in rows for update in row["updates"]]
        for ax, key, label in zip(
            axes.flat,
            ("value_loss", "grad_norm", "value_clipfrac"),
            ("Value loss", "Pre-clip global gradient norm", "Value clipping fraction"),
            strict=False,
        ):
            ax.plot(range(1, len(updates) + 1), [u[key] for u in updates], label=LABELS[mode], color=COLORS[mode])
            ax.set(xlabel="Optimizer mini-batch update", ylabel=label)
        axes[1, 1].plot(
            [r["batch"] for r in rows],
            [r["pre_update"]["ev"] for r in rows],
            "o-",
            label=LABELS[mode],
            color=COLORS[mode],
        )
        axes[1, 1].set(xlabel="Replay batch", ylabel="Pre-update training EV")
    _finish(fig, axes, output, "training_metrics")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for mode, rows in training.items():
        for ax, key, scale, label in zip(
            axes,
            ("value_seconds", "train_seconds", "peak_allocated_bytes"),
            (1, 1, 2**30),
            (
                "Value pass seconds",
                "Training and gradient synchronization seconds",
                "Peak CUDA allocated GiB (0 on CPU)",
            ),
            strict=True,
        ):
            ax.plot(
                [r["batch"] for r in rows],
                [r[key] / scale for r in rows],
                "o-",
                label=LABELS[mode],
                color=COLORS[mode],
            )
            ax.set(xlabel="Replay batch", ylabel=label)
    _finish(fig, axes, output, "training_cost")


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan, run, results, training = load_results(root)
    output = root / "training_report"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    (output / "plot_data.json").write_text(
        json.dumps(dict(experiment=run, validation=results, training=training), allow_nan=False) + "\n"
    )
    _ev_figures(plt, results, output)
    _profile_figures(plt, results, output)
    _trace_figures(plt, root, plan, results, output, validation_dir="validation")
    _training_figures(plt, training, output)
    (output / "README.txt").write_text(
        f"HF critic training on existing replay. Trained modes: {', '.join(run['modes'])}.\n"
        "Each selected mode has its own trained critic.\n"
        "Exact token-level EV/MSE/MAE; plotted values use segment-local 100-token means.\n"
        "Training time includes AM fitting/cache reconstruction and gradient synchronization, excludes validation/checkpoint export.\n"
        "Do not combine these curves with native Miles runs as a controlled backend comparison.\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    render(parser.parse_args().output_root)


if __name__ == "__main__":
    main()
