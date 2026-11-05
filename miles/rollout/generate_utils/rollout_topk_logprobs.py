"""Record the sampler's top-k candidate log-probs at generation time."""

import math
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample

_MIN_FLOAT32_LOGPROB = -float(np.finfo(np.float32).max)


def configure_rollout_topk_logprobs_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    """Request candidate probabilities, validating the actual per-call settings.

    The rollout args own ``top_logprobs_num`` / ``top_logprobs`` and ``sampling_logprobs_mode``
    on training requests and silently override client-supplied values.
    """
    k = args.rollout_top_logprobs_num
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_rollout_topk_logprobs_sampling(sampling, temperature=args.rollout_temperature, candidate_count=k)
    if args.rollout_sampling_logprobs_mode == "support":
        # SGLang's support mode returns the actual post-filter behavior distribution.
        request.pop("top_logprobs", None)
        request.pop("top_logprobs_num", None)
        request["sampling_logprobs_mode"] = "support"
    elif openai:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs"] = k
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs_num"] = k


def _validate_wire_token_logprob(token: Any, logp: Any, *, candidate: bool = False) -> None:
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)) or not 0 <= token < 2**31:
        raise ValueError("Rollout server returned an invalid token ID; expected a nonnegative int32 ID")
    if isinstance(logp, (bool, np.bool_)) or not isinstance(logp, (int, float, np.integer, np.floating)):
        raise ValueError("Rollout server returned an invalid log probability")
    try:
        finite = math.isfinite(logp)
    except OverflowError:
        finite = False
    if not finite or logp > 0 or (candidate and logp < _MIN_FLOAT32_LOGPROB):
        raise ValueError("Rollout server returned an invalid log probability; expected a finite nonpositive value")


def extract_rollout_topk_logprobs(
    meta: Mapping[str, Any], k: int, *, sampling_logprobs_mode: str = "selected"
) -> tuple[np.ndarray, np.ndarray]:
    """Pack wire candidates without silently coercing invalid IDs or probabilities.

    Selected mode keeps the first k candidates, matching the existing request
    contract. Distribution checks such as duplicate IDs belong to CI validation.
    """
    items = meta.get("output_token_logprobs") or []
    if not isinstance(items, (list, tuple)):
        raise ValueError("Rollout server returned an invalid sampled-token log probability array")
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("Rollout server returned an invalid sampled-token log probability entry")
        _validate_wire_token_logprob(item[1], item[0])
    n = len(items)
    support_mode = sampling_logprobs_mode == "support"
    field = "output_token_sampling_logprobs" if support_mode else "output_top_logprobs"
    rows = meta.get(field)
    if rows is not None and not isinstance(rows, (list, tuple)):
        raise ValueError(f"Rollout server returned an invalid {field} array")
    if (n and rows is None) or (rows is not None and len(rows) != n):
        raise ValueError(f"Rollout top-k logprobs collection requires SGLang {field} for every generated token")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows or []):
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"Rollout server returned an invalid {field} row")
        if support_mode:
            if len(entries) > k:
                raise ValueError("Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num")
            token_ids, probabilities = meta["output_token_sampling_mask"][i], entries
            if len(token_ids) != len(probabilities):
                raise ValueError("Rollout sampling support IDs and log probabilities must have matching lengths")
        else:
            if any(not isinstance(entry, (list, tuple)) or len(entry) < 2 for entry in entries[:k]):
                raise ValueError("Rollout server returned an invalid top-k candidate entry")
            token_ids = [entry[1] for entry in entries[:k]]
            probabilities = [entry[0] for entry in entries[:k]]
        for token, logp in zip(token_ids, probabilities, strict=True):
            _validate_wire_token_logprob(token, logp, candidate=True)
        ids[i, : len(token_ids)] = token_ids
        logps[i, : len(token_ids)] = probabilities
    return ids, logps


def append_rollout_topk_logprobs(
    sample: Sample, meta: Mapping[str, Any], k: int, *, sampling_logprobs_mode: str = "selected"
) -> None:
    """Append compact [response, k] arrays after appending generated tokens.

    No rescoring is allowed: with stale rollouts it would replace the behavior
    policy. Slots unavailable from the server are represented by ID -1/logp -inf.
    append_sampling_metadata validates the sampling support on the same response,
    and CI checks the whole sample with validate_rollout_topk_logprobs_sample.
    """
    if not k:
        return
    ids, logps = extract_rollout_topk_logprobs(meta, k, sampling_logprobs_mode=sampling_logprobs_mode)
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, name)
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def _support_membership(ids: np.ndarray, support: RolloutSamplingMask) -> tuple[np.ndarray, np.ndarray]:
    """Match candidate IDs to ragged support IDs without a per-token Python loop."""
    n = len(ids)
    if not n:
        return np.zeros_like(ids, dtype=bool), np.empty(0, dtype=np.int64)
    flat_ids, offsets = support._as_tensors()
    flat_ids = flat_ids.numpy()
    offsets = offsets.numpy()
    lengths = np.diff(offsets[: n + 1])
    support_ids = flat_ids[: offsets[n]]
    stride = 1 << 32  # Token IDs are nonnegative int32; rows remain distinct.
    support_rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    support_keys = np.sort(support_rows * stride + support_ids.astype(np.int64))
    candidate_keys = np.arange(n, dtype=np.int64)[:, None] * stride + ids.astype(np.int64)
    positions = np.searchsorted(support_keys, candidate_keys)
    membership = (ids >= 0) & (support_keys[np.minimum(positions, len(support_keys) - 1)] == candidate_keys)
    return membership, lengths


def pad_rollout_topk_logprobs(sample: Sample, count: int) -> None:
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_rollout_topk_logprobs_sample(sample: Sample, k: int) -> None:
    """Validate candidate distributions, including custom producers and restored rollouts.

    Full-sample sorting and support matching have significant CPU and memory costs;
    training callers must enable this validation only under --ci-test.
    """
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError(
            "Rollout top-k logprobs collection requires candidate and sampled-token logprobs on every rollout"
        )
    if ids.shape[1] != k:
        raise ValueError("Rollout top-k logprobs candidates must have --rollout-top-logprobs-num columns")
    if not np.issubdtype(ids.dtype, np.integer) or ((ids < -1) | (ids > np.iinfo(np.int32).max)).any():
        raise ValueError("Rollout top-k candidate IDs must be nonnegative int32 IDs or -1 padding")
    valid = ids >= 0
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate rollout top-k logprobs candidate token IDs")
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (active & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs rollout top-k logprobs candidates")
    if not (np.issubdtype(logps.dtype, np.floating) or np.issubdtype(logps.dtype, np.integer)):
        raise ValueError("Rollout top-k candidate log probabilities must be numeric")
    if not np.isfinite(logps[valid]).all() or (logps[valid] > 0).any() or not np.isneginf(logps[~valid]).all():
        raise ValueError("Rollout top-k log probabilities must be finite and nonpositive; padding is -inf")
    if (logps[valid] < _MIN_FLOAT32_LOGPROB).any():
        raise ValueError("Rollout top-k candidate log probabilities must be representable in float32")
    if (np.exp(np.where(valid, logps, -np.inf)).sum(-1) > 1.00001).any():
        raise ValueError("Rollout top-k candidate probability mass exceeds one")
    if sample.rollout_sampling_mask is not None:
        in_support, lengths = _support_membership(ids, sample.rollout_sampling_mask)
        if ((valid != in_support) & active[:, None]).any() or ((valid.sum(-1) != lengths) & active).any():
            raise ValueError("Rollout top-k logprobs candidates must equal the sampling support")
    sampled = np.asarray(sample.rollout_log_probs)
    response_tokens = sample.tokens[-sample.response_length :] if sample.response_length else []
    for token, logp in zip(response_tokens, sampled, strict=True):
        _validate_wire_token_logprob(token, logp)
    tokens = np.asarray(response_tokens, dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def merge_rollout_topk_logprobs_arrays(
    a: np.ndarray | None, b: np.ndarray | None, *, field: str, gap: int
) -> np.ndarray | None:
    """Merge candidate arrays with masked padding over a tool/observation gap."""
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for rollout top-k logprobs collection")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))


def merge_rollout_topk_logprobs_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    return merge_rollout_topk_logprobs_arrays(getattr(first, field), getattr(second, field), field=field, gap=gap)
