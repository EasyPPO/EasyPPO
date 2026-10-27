"""Exact, whole-parameter gradient attribution for fixed token-loss windows."""

import math
import random
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


def select_windows(count, *, size=64, controls=8, seed=1234):
    """Contiguous in generated-token order; tools remain in the model context.

    Controls exclude the entire target window. Distinct starts are sampled
    without replacement; controls may overlap each other. No short windows.
    """
    if min(size, controls) < 1:
        raise ValueError("Window size and control count must be positive")
    if count < size:
        return None, "short_target"
    candidates = range(size, count - size + 1)
    if not candidates:
        return None, "no_control_window"
    starts = random.Random(seed).sample(candidates, min(controls, len(candidates)))
    return [list(range(size))] + [list(range(start, start + size)) for start in starts], None


@contextmanager
def probe_mode(model):
    """Enable differentiation, not updates; restore flags and preserve .grad."""
    training = [(module, module.training) for module in model.modules()]
    requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.eval().requires_grad_(True)
        yield
    finally:
        for p, flag in requires_grad:
            p.requires_grad_(flag)
        for module, flag in training:
            module.training = flag


def _products(gradients, reference=None):
    """Reduce window/rest products after subtracting FP64 parameter chunks."""
    device = next((g.device for g in gradients if g is not None), torch.device("cpu"))
    norm2 = torch.zeros((), dtype=torch.float64, device=device)
    rest_norm2 = torch.zeros_like(norm2)
    dot = torch.zeros_like(norm2)
    for index, gradient in enumerate(gradients):
        ref = None if reference is None else reference[index]
        if gradient is None and ref is None:
            continue
        flat = None if gradient is None else gradient.detach().reshape(-1)
        length = ref.numel() if flat is None else flat.numel()
        for start in range(0, length, 1 << 20):
            chunk = 0.0 if flat is None else flat[start : start + (1 << 20)].double()
            if flat is not None:
                norm2 += chunk.square().sum()
            if reference is None:
                continue
            other = (
                0.0
                if ref is None
                else ref.reshape(-1)[start : start + (1 << 20)].to(device=device, dtype=torch.float64)
            )
            rest = other - chunk
            rest_norm2 += rest.square().sum()
            if flat is not None:
                dot += (chunk * rest).sum()
    a, b, c = float(norm2), float(rest_norm2), float(dot)
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("Nonfinite gradients; retry in float32 or inspect the checkpoint")
    return a, b, c


def measure_gradients(losses, parameters, windows, *, token_count=None):
    """Each loss is one generated token; every subset uses the SAME denominator N.

    One full backward plus one per window. Only the full gradient is retained
    on CPU, in its original dtype. No optimizer, .backward(), or .grad mutation.
    The reference is the current SEGMENT's loss over ALL model parameters.
    """
    parameters = tuple(parameters)  # nn.Module.parameters deduplicates tied weights
    loss_function = losses if callable(losses) else None
    full_losses = loss_function(None) if loss_function else losses
    count = len(full_losses)
    if token_count is not None and token_count != count:
        raise ValueError("Token count disagrees with full loss")
    if full_losses.ndim != 1 or not full_losses.numel() or not torch.isfinite(full_losses).all():
        raise ValueError("Expected a finite, nonempty vector of token losses")
    if not windows or any(not w or len(set(w)) != len(w) or min(w) < 0 or max(w) >= count for w in windows):
        raise ValueError("Windows must contain unique, in-range loss indices")
    full = torch.autograd.grad(full_losses.mean(), parameters, retain_graph=True, allow_unused=True)
    full_norm2, _, _ = _products(full)
    reference = tuple(None if g is None else g.detach().to("cpu", copy=True) for g in full)
    used_parameters = sum(p.numel() for p, g in zip(parameters, full, strict=True) if g is not None)
    del full
    del full_losses
    results = []
    for index, window in enumerate(windows):
        selected_losses = loss_function(window) if loss_function else losses[window]
        gradient = torch.autograd.grad(
            selected_losses.sum() / count,
            parameters,
            retain_graph=index + 1 < len(windows),
            allow_unused=True,
        )
        norm2, rest_norm2, dot = _products(gradient, reference)
        del gradient
        norm = math.sqrt(norm2)
        results.append(
            dict(
                norm=norm,
                # Norm of the 64-token mean, for comparisons across segment lengths.
                window_mean_norm=norm * count / len(window),
                norm_over_full=norm / math.sqrt(full_norm2) if full_norm2 else None,
                rest_norm=math.sqrt(rest_norm2),
                norm_over_rest=norm / math.sqrt(rest_norm2) if rest_norm2 else None,
                projection_on_rest=dot / rest_norm2 if rest_norm2 else None,
                cosine_with_rest=dot / math.sqrt(norm2 * rest_norm2) if norm2 and rest_norm2 else None,
            )
        )
    baseline = sum(row["norm"] for row in results[1:]) / (len(results) - 1) if len(results) > 1 else 0.0
    return dict(
        full_norm=math.sqrt(full_norm2),
        parameter_count=sum(p.numel() for p in parameters),
        used_parameter_count=used_parameters,
        target=results[0],
        controls=results[1:],
        ratio_to_control_mean=results[0]["norm"] / baseline if baseline else None,
        pair_ratios=[results[0]["norm"] / row["norm"] if row["norm"] else None for row in results[1:]],
    )


def summarize(rows, *, seed=1234, bootstrap=2000):
    """Equal problem weights; resample problems, not overlapping windows."""
    if bootstrap < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    groups = defaultdict(list)
    measured = [row for row in rows if "skip" not in row]
    for row in measured:
        ratio = row["ratio_to_control_mean"]
        if ratio is not None and ratio > 0 and math.isfinite(ratio):
            groups[row["problem_id"]].append(math.log(ratio))
    means = np.array([np.mean(groups[key]) for key in sorted(groups)])
    rng = np.random.default_rng(seed)
    ci = None
    if len(means) > 1:
        samples = [float(rng.choice(means, len(means), replace=True).mean()) for _ in range(bootstrap)]
        ci = np.exp(np.quantile(samples, [0.025, 0.975])).tolist()
    return dict(
        measured_boundaries=len(measured),
        skipped_boundaries=len(rows) - len(measured),
        valid_log_ratio_boundaries=sum(map(len, groups.values())),
        excluded_zero_or_undefined_ratios=len(measured) - sum(map(len, groups.values())),
        problems=len(groups),
        problem_balanced_geometric_ratio=float(np.exp(means.mean())) if len(means) else None,
        problem_cluster_bootstrap_95ci=ci,
    )
