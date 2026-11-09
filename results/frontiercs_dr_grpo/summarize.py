"""Summarize a FrontierSmith/Frontier-CS run from verl's dumps.

Reads the validation dumps (``trainer.validation_data_dir``, one ``<step>.jsonl`` per validation) and the
per-step metrics of verl's ``file`` logger, and writes:

  val_curve.csv       step, Avg@5 and Best@5 (max of the 5 samples) on the validation problems (0-100), and
                      verl's logged val-core metrics (its best@5 is a bootstrap estimate, lower than the max)
  val_per_problem.csv step, problem id, the 5 sample scores
  train_metrics.csv   selected per-step training metrics

usage: python summarize.py --validation-dir DIR --metrics FILE [FILE ...] --val-parquet PARQUET --out DIR
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--out", required=True)
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
    for r in curve:
        print(f"step {r['step']:4d}  avg@5 {r['avg@5']:6.2f}  best@5 {r['best@5']:6.2f}  ({r['problems']} problems)")


if __name__ == "__main__":
    main()
