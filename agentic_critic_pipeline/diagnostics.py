"""Extended experiment 9 diagnostics, using strict pairing and problem-held-out fits.

Raw value arrays are reduced one JSONL record at a time. Gradient vectors are
retained per segment; token arrays are never retained for the whole corpus.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import (
    BUCKETS,
    _key,
    attach_episode_metadata,
    bucket,
    interval,
    load_values,
    metrics,
    pair_records,
    paired_metrics,
    statistics,
)

FEATURES = ("submissions_before", "best_before", "last_before", "tokens_before")
POSITIONS = ((0, 32), (32, 256), (256, 2048), (2048, None))


def paths_to_files(paths):
    files = set()
    for source in paths:
        path = Path(source)
        files.update(path.glob("values.rank*.jsonl") if path.is_dir() else [path])
    return sorted(files)


def enrich_raw(records, paths):
    """Keep sufficient statistics for actor-position slices, not raw token arrays."""
    for path in paths_to_files(paths):
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = _key(row)
                if key not in records:
                    continue
                values = np.asarray(row["values"], dtype=float)
                records[key]["position_stats"] = [statistics(values[lo:hi], row["reward"]) for lo, hi in POSITIONS]


def attach_history(pairs, paths):
    attach_episode_metadata(pairs, paths)
    wanted = {a["episode_id"] for a, _ in pairs}
    history = {}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                episode = json.loads(line)
                eid = episode["episode_id"]
                if eid not in wanted:
                    continue
                if eid in history:
                    raise ValueError(f"Duplicate episode history {eid}")
                events = episode["events"]
                submissions = episode["submissions"]
                rows = []
                for index in range(len(episode["segments"])):
                    prior = [s for s in submissions if s["segment"] < index]
                    scores = [float(s["score"]) / 100 for s in prior]
                    rows.append(
                        [
                            len(prior),
                            max(scores, default=0.0),
                            scores[-1] if scores else 0.0,
                            sum(len(e["tokens"]) for e in events if e["segment"] < index),
                        ]
                    )
                history[eid] = rows
    for a, b in pairs:
        for row in (a, b):
            row["history_features"] = history[a["episode_id"]][a["segment_index"]]


def linear_predict(x_train, y_train, x_test, weights=None):
    """Train-fold standardization and weighted least squares with an intercept."""
    x_train, x_test = np.asarray(x_train, float), np.asarray(x_test, float)
    weights = np.ones(len(x_train)) if weights is None else np.asarray(weights, float)
    mean = np.average(x_train, axis=0, weights=weights)
    scale = np.sqrt(np.average((x_train - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.0)
    train = np.column_stack(((x_train - mean) / scale, np.ones(len(x_train))))
    test = np.column_stack(((x_test - mean) / scale, np.ones(len(x_test))))
    root = np.sqrt(weights)
    coef = np.linalg.lstsq(train * root[:, None], np.asarray(y_train) * root, rcond=None)[0]
    return test @ coef


def lopo(x, y, problems, weights=None):
    problems = np.asarray(problems)
    if len(set(problems)) < 2:
        return None
    prediction = np.empty(len(y))
    for problem in sorted(set(problems)):
        test = problems == problem
        prediction[test] = linear_predict(
            np.asarray(x)[~test],
            np.asarray(y)[~test],
            np.asarray(x)[test],
            None if weights is None else np.asarray(weights)[~test],
        )
    return prediction


def paired_estimate(rows, compute, bootstrap, seed):
    """Joint cluster resampling; fitted out-of-fold predictions remain fixed."""
    problems = sorted({r[0]["problem_id"] for r in rows})
    point = compute(rows)
    draws = defaultdict(list)
    grouped = {p: [r for r in rows if r[0]["problem_id"] == p] for p in problems}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap if problems else 0):
        sample = [r for i in rng.integers(len(problems), size=len(problems)) for r in grouped[problems[i]]]
        stats = compute(sample)
        for name, (a, b) in stats.items():
            draws[name].append(None if a is None or b is None else b - a)
    return {
        name: dict(
            actor_summary=a,
            full_history=b,
            full_minus_actor=dict(
                value=None if a is None or b is None else b - a, **interval(draws[name], len(problems))
            ),
        )
        for name, (a, b) in point.items()
    }


def residual_history(pairs, bootstrap=2000, seed=0):
    rows = [p for p in pairs if p[0]["segment_index"] > 0]
    if not rows or len({a["problem_id"] for a, _ in rows}) < 2:
        return dict(segments=len(rows), status="insufficient_problems", metrics={})
    x = np.array([a["history_features"] for a, _ in rows])
    problems = [a["problem_id"] for a, _ in rows]
    for side in range(2):
        y = np.array([-p[side]["stats"][3] / p[side]["stats"][0] for p in rows])
        prediction = lopo(x, y, problems)
        for pair, target, pred in zip(rows, y, prediction, strict=True):
            pair[side]["residual_target"], pair[side]["residual_prediction"] = target, pred

    def compute(sample):
        stats = []
        for side in range(2):
            y = np.array([p[side]["residual_target"] for p in sample])
            pred = np.array([p[side]["residual_prediction"] for p in sample])
            denominator = np.sum((y - y.mean()) ** 2)
            r2 = None if denominator <= 1e-14 else float(1 - np.sum((y - pred) ** 2) / denominator)
            stats.append((r2, None if r2 is None else max(0.0, r2) * float(np.mean(y * y))))
        return {name: (stats[0][i], stats[1][i]) for i, name in enumerate(("r2", "squared_error_explained"))}

    return dict(
        segments=len(rows),
        features=list(FEATURES),
        fit="leave_one_problem_out_train_fold_standardization",
        bootstrap="fixed_out_of_fold_predictions_resampled_by_problem",
        metrics=paired_estimate(rows, compute, bootstrap, seed),
    )


def score_spans(episode):
    """Actor-event token spans and visible best scores, in chronological order.

    Full history retains every judged score. Segment resets on resume; summary
    additionally carries the previous best when its printed number occurs in
    the summary final answer, matching prefix_max_reference's heuristic.
    """
    full, local, carried = 0.0, 0.0, 0.0
    spans = []
    for event in episode["events"]:
        kind = event["kind"]
        if kind == "resume":
            local = 0.0
        if kind in ("feedback", "compact_request"):
            match = re.search(r"Best score so far:\s*([0-9]+(?:\.[0-9]+)?)", event.get("text", ""))
            if match:
                full = max(full, float(match.group(1)) / 100)
                local = full
            # No-code feedback is legitimate and reveals no new score.
            # It must not restore history that the latest summary discarded.
        if event["actor"] and event["tokens"]:
            spans.append(
                dict(
                    episode_id=episode["episode_id"],
                    problem_id=episode["problem_id"],
                    segment_index=event["segment"],
                    reward=episode["reward"],
                    n=len(event["tokens"]),
                    full=full,
                    segment=local,
                    summary=max(local, carried),
                )
            )
        if kind == "summary":
            if "text" not in event:
                raise ValueError("Summary text required for visible-score baseline")
            text = event["text"].rsplit("</think>", 1)[-1]
            value = full * 100
            forms = {f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"}
            retained = any(re.search(r"(?<![\d.])" + re.escape(form) + r"(?![\d]|\.\d)", text) for form in forms)
            carried = full if full > 0 and retained else 0.0
    return spans


def score_baseline(pairs, episode_paths=()):
    """Token-time score-only reference, fitted exclusively on training episodes.

    The summary-number visibility heuristic follows experiment 9. Training
    predictions leave the entire target problem out. Validation may reuse train
    problems with distinct episodes; no heldout episode ever fits its own label.
    """
    spans, training, seen = {}, [], set()
    for path in episode_paths:
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                ep = json.loads(line)
                if ep["episode_id"] in seen:
                    raise ValueError("Duplicate episode in score reference")
                seen.add(ep["episode_id"])
                own = score_spans(ep)
                spans[ep["episode_id"]] = own
                if ep.get("split") == "train":
                    training.extend(own)
    train_episodes = {r["episode_id"] for r in training}
    results = {}
    for phase in sorted({a["phase"] for a, _ in pairs}):
        phase_pairs = [p for p in pairs if p[0]["phase"] == phase]
        output_spans = []
        for a, _ in phase_pairs:
            own = [r for r in spans.get(a["episode_id"], []) if r["segment_index"] == a["segment_index"]]
            if sum(r["n"] for r in own) != a["stats"][0]:
                raise ValueError("Token-time score reference does not align with actor tokens")
            output_spans.extend(own)
        problems = {r["problem_id"] for r in output_spans}
        if phase == "heldout" and train_episodes & {r["episode_id"] for r in output_spans}:
            raise ValueError("Score baseline heldout episodes overlap training episodes")
        result = dict(feature="token_time_visible_best_score", spans=len(output_spans), variants={})
        for variant in ("full", "segment", "summary"):
            prediction = np.empty(len(output_spans))
            sufficient = bool(training)
            owners = np.array([r["problem_id"] for r in output_spans])
            for problem in sorted(problems):
                fit = [r for r in training if phase != "pre_update" or r["problem_id"] != problem]
                test = owners == problem
                if not fit:
                    sufficient = False
                    break
                prediction[test] = linear_predict(
                    [[r[variant]] for r in fit],
                    [r["reward"] for r in fit],
                    [[output_spans[i][variant]] for i in np.flatnonzero(test)],
                    [r["n"] for r in fit],
                )
            if not sufficient:
                result["variants"][variant] = dict(status="insufficient_training_problems")
                continue
            for name in BUCKETS:
                total = np.zeros(6)
                for row, pred in zip(output_spans, prediction, strict=True):
                    if name == "all" or bucket(row["segment_index"]) == name:
                        total += row["n"] * statistics([pred], row["reward"])
                result["variants"].setdefault(variant, dict(status="ok", by_compactions={}))["by_compactions"][name] = (
                    metrics(total)
                )
        results[phase] = result
    return results


def gradient_metrics(pairs, bootstrap=2000, seed=0):
    for pair in pairs:
        for row in pair:
            gradient = row.get("head_gradient")
            if gradient is None:
                raise ValueError("Missing head_gradient: regenerate heldout values with gradient recording enabled")
            gradient = np.asarray(gradient, float)
            if gradient.ndim != 1 or not len(gradient) or not np.isfinite(gradient).all():
                raise ValueError("Invalid head_gradient vector")
            row["head_gradient"] = gradient
    dimensions = {len(row["head_gradient"]) for pair in pairs for row in pair}
    if len(dimensions) > 1:
        raise ValueError("Head gradient dimensions differ")

    def blocks(rows):
        grouped = defaultdict(list)
        for pair in rows:
            grouped[pair[0]["problem_id"]].append(pair)
        scalar, vectors = [], []
        for problem in sorted(grouped):
            sides, sums = [], []
            own = grouped[problem]
            for side in range(2):
                gradients = np.stack([pair[side]["head_gradient"] for pair in own])
                norms = np.linalg.norm(gradients, axis=1)
                episode_vectors = defaultdict(list)
                for pair, gradient, norm in zip(own, gradients, norms, strict=True):
                    if norm > 1e-14:
                        episode_vectors[pair[0]["episode_id"]].append(gradient / norm)
                episode_sums = [np.sum(v, axis=0) for v in episode_vectors.values()]
                counts = [len(v) for v in episode_vectors.values()]
                numerator, count = 0.0, 0
                if episode_sums:
                    total = np.sum(episode_sums, axis=0)
                    numerator = (float(total @ total) - sum(float(v @ v) for v in episode_sums)) / 2
                    count = (sum(counts) ** 2 - sum(n * n for n in counts)) // 2
                sides.append([len(own), float(norms.sum()), float(norms @ norms), numerator, count])
                sums.append(gradients.sum(axis=0))
            scalar.append(sides)
            vectors.append(sums)
        scalar, vectors = np.asarray(scalar), np.asarray(vectors)
        # Squared norm of weighted cluster gradient sums = w.T @ Gram @ w.
        # Bootstrap never reconstructs per-segment D-dimensional arrays.
        gram = np.stack([vectors[:, side] @ vectors[:, side].T for side in range(2)])
        return scalar, gram

    def estimates(counts, scalar, gram):
        totals = np.einsum("bp,psm->bsm", counts, scalar)
        n, norm, energy, numerator, denominator = np.moveaxis(totals, -1, 0)
        mean_norm = norm / n
        signal = np.stack([np.sum((counts @ gram[side]) * counts, axis=1) for side in range(2)], axis=1)
        gsnr = np.divide(signal, n * energy, out=np.full_like(signal, np.nan), where=energy / n > 1e-14)
        cosine = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        return np.stack([mean_norm, np.clip(gsnr, 0.0, 1.0), np.clip(cosine, -1.0, 1.0)], axis=-1)

    names = ("norm", "gsnr", "same_problem_different_episode_cosine")
    out = {}
    for name in BUCKETS:
        rows = [p for p in pairs if name == "all" or bucket(p[0]["segment_index"]) == name]
        result = dict(segments=len(rows), metrics={})
        out[name] = result
        if not rows:
            continue
        scalar, gram = blocks(rows)
        n_problems = len(scalar)
        point = estimates(np.ones((1, n_problems)), scalar, gram)[0]
        differences = []
        rng = np.random.default_rng(seed)
        for start in range(0, bootstrap, 128):
            size = min(128, bootstrap - start)
            sampled = rng.integers(n_problems, size=(size, n_problems))
            counts = np.zeros((size, n_problems))
            np.add.at(counts, (np.repeat(np.arange(size), n_problems), sampled.ravel()), 1.0)
            draws = estimates(counts, scalar, gram)
            differences.extend(draws[:, 1] - draws[:, 0])
        differences = np.asarray(differences)
        for index, metric in enumerate(names):
            a, b = (float(v) if np.isfinite(v) else None for v in point[:, index])
            result["metrics"][metric] = dict(
                actor_summary=a,
                full_history=b,
                full_minus_actor=dict(
                    value=None if a is None or b is None else b - a,
                    **interval(differences[:, index] if len(differences) else [], n_problems),
                ),
            )
    return out


def input_slices(pairs, bootstrap=2000, seed=0):
    rows = [p for p in pairs if p[0]["segment_index"] > 0]
    cuts = np.quantile([a["removed_cumulative"] for a, _ in rows], [1 / 3, 2 / 3]).tolist() if rows else []
    removed = {}
    for index, name in enumerate(("lowest_third", "middle_third", "highest_third")):
        selected = [p for p in rows if sum(p[0]["removed_cumulative"] > cut for cut in cuts) == index]
        removed[name] = paired_metrics(selected, bootstrap, seed)
    positions = {}
    for index, (lo, hi) in enumerate(POSITIONS):
        selected = []
        for pair in rows:
            copies = tuple(dict(row, stats=row["position_stats"][index]) for row in pair)
            if copies[0]["stats"][0]:
                selected.append(copies)
        positions[f"{lo}-{hi - 1}" if hi else f"{lo}+"] = paired_metrics(selected, bootstrap, seed)
    return dict(removed_cuts=cuts, cumulative_removed_thirds=removed, actor_positions=positions)


def zero_compaction_check(actor_paths, full_paths):
    """Compare original token values with bounded memory and an offset index."""
    offsets = {}
    for path in paths_to_files(full_paths):
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["segment_index"] == 0:
                    offsets[_key(row)] = (path, offset)
    segment_differences, token_error_sum, tokens, maximum_token = [], 0.0, 0, 0.0
    for path in paths_to_files(actor_paths):
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["segment_index"] != 0:
                    continue
                full_path, offset = offsets[_key(row)]
                with full_path.open("rb") as full_stream:
                    full_stream.seek(offset)
                    other = json.loads(full_stream.readline())
                difference = np.asarray(other["values"]) - np.asarray(row["values"])
                segment_differences.append(abs(float(difference.mean())))
                token_error_sum += float(np.abs(difference).sum())
                tokens += len(difference)
                maximum_token = max(maximum_token, float(np.abs(difference).max()))
    maximum_mean = max(segment_differences, default=None)
    mean_token = token_error_sum / tokens if tokens else None
    return dict(
        segments=len(segment_differences),
        tokens=tokens,
        max_segment_mean_difference=maximum_mean,
        mean_token_absolute_difference=mean_token,
        max_token_absolute_difference=maximum_token if tokens else None,
        tolerance=0.01,
        passed=bool(tokens) and maximum_mean <= 0.01 and mean_token <= 0.01,
        definition="same trained critic; segment mean and mean token absolute difference <= tolerance",
    )


def run(
    actor_paths,
    full_paths,
    episodes,
    *,
    bootstrap=2000,
    seed=0,
    allow_input_swap_diagnostic=False,
    require_gradients=True,
    primary_last_batches=30,
):
    if min(bootstrap, primary_last_batches) < 1:
        raise ValueError("bootstrap and primary_last_batches must be positive")
    a = load_values(actor_paths, "actor_summary", allow_input_swap_diagnostic=allow_input_swap_diagnostic)
    b = load_values(full_paths, "full_history", allow_input_swap_diagnostic=allow_input_swap_diagnostic)
    pairs = pair_records(a, b)
    if any(a["phase"] == "pre_update" for a, _ in pairs):
        batches = sorted({a["batch"] for a, _ in pairs if a["phase"] == "pre_update"})
        selected = set(batches[-primary_last_batches:])
        pairs = [p for p in pairs if p[0]["phase"] != "pre_update" or p[0]["batch"] in selected]
    attach_history(pairs, episodes)
    enrich_raw(a, actor_paths)
    enrich_raw(b, full_paths)
    swapped = any(row["trained_mode"] != row["input_mode"] for pair in pairs for row in pair)
    if swapped and any(x["trained_mode"] != y["trained_mode"] for x, y in pairs):
        raise ValueError("Input swap requires the same trained mode on both sides")
    if swapped:
        for x, y in pairs:
            for field in ("checkpoint_run_id", "checkpoint_batch"):
                if field not in x or field not in y or x[field] != y[field]:
                    raise ValueError(f"Input swap requires identical checkpoint provenance: {field}")
    result = dict(
        settings=dict(bootstrap=bootstrap, seed=seed, primary_last_batches=primary_last_batches),
        comparison_kind="input_swap_diagnostic" if swapped else "matched_trained_critics",
        phases={},
        score_only_baseline=score_baseline(pairs, episodes),
    )
    for phase in sorted({x["phase"] for x, _ in pairs}):
        rows = [p for p in pairs if p[0]["phase"] == phase]
        data = dict(
            primary_batches=sorted({a["batch"] for a, _ in rows}),
            residual_history=residual_history(rows, bootstrap, seed),
            slices=input_slices(rows, bootstrap, seed),
        )
        if phase == "heldout" and require_gradients:
            data["gradients"] = gradient_metrics(rows, bootstrap, seed)
        if swapped:
            data["zero_compaction_check"] = zero_compaction_check(actor_paths, full_paths)
        result["phases"][phase] = data
    return result


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = [
        "# Extended offline diagnostics",
        "",
        "Paired differences resample problems jointly. Residual regressions use leave-one-problem-out "
        "prediction and train-fold standardization. Confidence intervals resample the fixed out-of-fold "
        "predictions. Undefined metrics are null.",
        "",
        "The score-only baseline follows visible best scores at actor-event token spans. "
        "Heldout predictions fit on training episodes only; same-problem validation uses independent episodes. "
        "Training readouts leave the target problem out of the reference fit. The summary variant carries "
        "the previous best only when its printed numeric form occurs in the summary final answer; "
        "this is a visibility heuristic, not a semantic annotation.",
        "",
        "Recorded head gradients are mean((V-G)*h) for value-head weights only, per segment. "
        "They exclude bias, PPO clipping, problem weights, and minibatch loss scaling. "
        "Input-swap diagnostics require identical source checkpoint_run_id and checkpoint_batch.",
        "",
        "## Settings and score-only references",
        "",
        "```json",
        json.dumps(
            {k: result[k] for k in ("settings", "comparison_kind", "score_only_baseline")},
            indent=2,
            allow_nan=False,
        ),
        "```",
        "",
        "Gradient GSNR is squared norm of the mean divided by mean squared norm. Conflict cosine excludes "
        "pairs from the same episode. Bootstrap copies repeat internal cluster pairs "
        "without creating cross-copy pairs. "
        "Low GSNR alone does not establish critic failure.",
        "",
    ]
    for phase, data in result["phases"].items():
        report.extend([f"## {phase}", "", "```json", json.dumps(data, indent=2, allow_nan=False), "```", ""])
    (output / "diagnostics-report.md").write_text("\n".join(report))
    write_plots(result, output)
    with (output / "diagnostics-report.md").open("a") as stream:
        for phase in result["phases"]:
            stream.write(f"\n![{phase} diagnostics]({phase}-diagnostics.png)\n")


def write_plots(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = ("actor_summary", "full_history")
    for phase, data in result["phases"].items():
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
        axes = axes.ravel()
        for index, metric in enumerate(("r2", "squared_error_explained")):
            row = data["residual_history"]["metrics"].get(metric, {})
            for side, mode in enumerate(modes):
                value = row.get(mode)
                if value is not None:
                    axes[index].bar(side, value, label=mode)
            axes[index].set_xticks([0, 1], modes, rotation=15)
            axes[index].set_title("History residual " + metric)
        for index, metric in enumerate(("norm", "gsnr", "same_problem_different_episode_cosine"), start=2):
            groups = data.get("gradients", {})
            for mode in modes:
                axes[index].plot(
                    BUCKETS,
                    [groups.get(b, {}).get("metrics", {}).get(metric, {}).get(mode) for b in BUCKETS],
                    marker="o",
                    label=mode,
                )
            axes[index].set_title("Head gradient " + metric.replace("_", " "))
        for index, grouping in enumerate(("cumulative_removed_thirds", "actor_positions"), start=5):
            groups = data["slices"][grouping]
            for mode in modes:
                axes[index].plot(list(groups), [r[mode]["ev"] for r in groups.values()], marker="o", label=mode)
            axes[index].set_title("EV by " + grouping.replace("_", " "))
            axes[index].tick_params(axis="x", rotation=15)
        references = result["score_only_baseline"][phase]["variants"]
        for variant, reference in references.items():
            if reference["status"] == "ok":
                axes[7].plot(
                    BUCKETS, [reference["by_compactions"][b]["ev"] for b in BUCKETS], marker="o", label=variant
                )
        axes[7].set_title("Token-time score-only EV")
        for ax in axes:
            ax.grid(alpha=0.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)
            lines_defined = any(np.isfinite(line.get_ydata(orig=False)).any() for line in ax.lines)
            if not ax.patches and not lines_defined:
                ax.text(0.5, 0.5, "Insufficient data / undefined", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output / f"{phase}-diagnostics.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-summary", nargs="+", required=True)
    parser.add_argument("--full-history", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primary-last-batches", type=int, default=30)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--span", type=int, default=256)
    parser.add_argument("--bin-size", type=int, default=16)
    parser.add_argument("--allow-input-swap-diagnostic", action="store_true")
    parser.add_argument("--skip-gradients", action="store_true", help="Explicitly omit heldout gradient diagnostics")
    args = parser.parse_args()
    result = run(
        args.actor_summary,
        args.full_history,
        args.episodes,
        bootstrap=args.bootstrap,
        seed=args.seed,
        allow_input_swap_diagnostic=args.allow_input_swap_diagnostic,
        require_gradients=not args.skip_gradients,
        primary_last_batches=args.primary_last_batches,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_outputs(result, output)


if __name__ == "__main__":
    main()
