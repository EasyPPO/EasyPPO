"""Summarize a FrontierSmith/Frontier-CS run from verl's dumps.

Reads the validation dumps (``trainer.validation_data_dir``, one ``<step>.jsonl`` per validation) and the
per-step metrics of verl's ``file`` logger, and writes:

  val_curve.csv       step, Avg@5 and Best@5 (max of the 5 samples) on the validation problems (0-100), and
                      verl's logged val-core metrics (its best@5 is a bootstrap estimate, lower than the max)
  val_per_problem.csv step, problem id, the 5 sample scores
  train_metrics.csv   selected per-step training metrics
  curves.png          (with --plot, needs matplotlib) validation and training curves

usage: python summarize.py --validation-dir DIR --metrics FILE [FILE ...] --val-parquet PARQUET --out DIR [--plot]
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import pandas as pd

TRAIN_KEYS = [
    "critic/score/mean",
    "critic/rewards/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "actor/entropy",
    "actor/grad_norm",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "timing_s/step",
]


def prompt_text(rendered):
    """The user message of a rendered chat prompt ("user\\n<content>\\nassistant\\n...")."""
    body = rendered.split("user\n", 1)[1]
    return body.rsplit("\nassistant\n", 1)[0].strip()


SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def plot(curve, train, path, title):
    """Small multiples, one y-axis each: validation Avg@5/Best@5, train reward, response length, truncation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    line = dict(linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    import matplotlib.ticker
    vs = [r["step"] for r in curve]
    ts = [r["step"] for r in train]
    xmax = max(vs + ts + [1])
    panels = [
        (axes[0][0], "Validation score (Frontier-CS, 0–100)",
         [("Best@5", vs, [r["best@5"] for r in curve], ORANGE), ("Avg@5", vs, [r["avg@5"] for r in curve], BLUE)]),
        (axes[0][1], "Train reward (mean, 0–1)",
         [(None, ts, [r["critic/score/mean"] for r in train], BLUE)]),
        (axes[1][0], "Response length (mean tokens)",
         [(None, ts, [r["response_length/mean"] for r in train], BLUE)]),
        (axes[1][1], "Truncated at 32,768 tokens (% of responses)",
         [(None, ts, [100 * r["response_length/clip_ratio"] for r in train], BLUE)]),
    ]
    for ax, name, series in panels:
        ax.set_facecolor(SURFACE)
        ax.set_title(name, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("training step")
        for label, x, y, color in series:
            pts = [(a, b) for a, b in zip(x, y) if b is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, label=label, **(line if len(xs) <= 60 else dict(linewidth=2)))
            if label:   # direct label at the last point, in ink, next to the colored mark
                ax.annotate(f"{label} {ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", color=INK2, fontsize=9)
        if any(lbl for lbl, *_ in series):
            ax.legend(frameon=False, loc="center left", labelcolor=INK2)
        ax.set_xlim(-0.5, xmax * 1.08 + 0.5)   # one step range for every panel
        ax.set_ylim(bottom=0)                  # magnitudes: start at zero
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}"))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--title", default="Dr. GRPO on FrontierSmith, validated on Frontier-CS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.val_parquet)
    problem_of = {r["prompt"][-1]["content"].strip(): r["reward_model"]["ground_truth"] for _, r in df.iterrows()}

    # one JSON object per logged step; a resumed job re-logs steps, the latest record wins
    train, val_logged = {}, {}
    for path in args.metrics:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                data = rec.get("data", rec)
                step = rec.get("step", data.get("training/global_step"))
                if step is not None and "val-core/frontiercs/acc/mean@5" in data:
                    val_logged[int(step)] = data
                if step is not None and any(k in data for k in TRAIN_KEYS):
                    train[int(step)] = {"step": int(step), **{k: data.get(k) for k in TRAIN_KEYS}}

    curve, per_problem = [], []
    steps = sorted(int(f[:-6]) for f in os.listdir(args.validation_dir) if re.fullmatch(r"\d+\.jsonl", f))
    for step in steps:
        scores = defaultdict(list)
        with open(os.path.join(args.validation_dir, f"{step}.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                scores[problem_of[prompt_text(row["input"])]].append(float(row["acc"]))
        n = len(scores)
        curve.append({
            "step": step,
            "problems": n,
            "avg@5": sum(sum(s) / len(s) for s in scores.values()) / n,
            "best@5": sum(max(s) for s in scores.values()) / n,
            "verl_mean@5": val_logged.get(step, {}).get("val-core/frontiercs/acc/mean@5"),
            "verl_best@5_bootstrap": val_logged.get(step, {}).get("val-core/frontiercs/acc/best@5/mean"),
        })
        for pid in sorted(scores, key=lambda p: int(p.rsplit("_", 1)[1])):
            per_problem.append({"step": step, "problem": pid, **{f"s{i}": v for i, v in enumerate(scores[pid])}})

    for name, rows in [("val_curve.csv", curve), ("val_per_problem.csv", per_problem),
                       ("train_metrics.csv", [train[s] for s in sorted(train)])]:
        if not rows:
            continue
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if args.plot and curve:
        rows = [train[s] for s in sorted(train)]
        num = lambda v: float(v) if v not in (None, "") else None
        tr = [{k: (num(v) if k != "step" else int(v)) for k, v in r.items()} for r in rows]
        plot(curve, tr, os.path.join(args.out, "curves.png"), args.title)
    for r in curve:
        print(f"step {r['step']:4d}  avg@5 {r['avg@5']:6.2f}  best@5 {r['best@5']:6.2f}  ({r['problems']} problems)")


if __name__ == "__main__":
    main()
