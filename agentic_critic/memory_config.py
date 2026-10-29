"""Resolve critic memory settings once, identically for value and training passes."""

import math

DEFAULT_QUERY_COUNT = 400
KV_DEFAULT = {"ratio": 0.1, "method": "am", "queries": "hindsight", "query_count": DEFAULT_QUERY_COUNT, "ridge": 1e-4}


def resolve_memory_config(mode: str, kv: dict | None) -> dict | None:
    if mode not in ("kv", "attention_matching"):
        if kv:
            raise ValueError(f"memory settings are not used by mode {mode!r}")
        return None
    defaults = {**KV_DEFAULT}
    if mode == "attention_matching":
        defaults.update(method="am_ridge", queries="summary")
    supplied = {k: v for k, v in (kv or {}).items() if v is not None}
    if unknown := supplied.keys() - defaults.keys():
        raise ValueError(f"unknown memory settings: {sorted(unknown)}")
    cfg = {**defaults, **supplied}
    if cfg["queries"] in ("summary", "causal") and "query_count" not in supplied:
        # Summary probes always use every original summary token; only hindsight has a budget.
        cfg["query_count"] = 0
    if mode == "attention_matching" and (cfg["method"] != "am_ridge" or cfg["queries"] != "summary"):
        raise ValueError("attention_matching requires method=am_ridge and queries=summary; use kv for ablations")
    if not math.isfinite(cfg["ratio"]) or not 0 < cfg["ratio"] <= 1:
        raise ValueError("kv ratio must be finite and in (0, 1]")
    if cfg["method"] not in ("evict", "am", "am_cv", "am_ridge"):
        raise ValueError("kv method must be evict, am, am_cv or am_ridge")
    if cfg["queries"] not in ("summary", "hindsight", "causal"):
        raise ValueError("kv queries must be summary, hindsight or causal")
    if cfg["method"] == "am_cv" and cfg["queries"] != "causal":
        raise ValueError("am_cv requires causal queries")
    if not isinstance(cfg["query_count"], int) or cfg["query_count"] < 0:
        raise ValueError("kv query_count must be a nonnegative integer")
    if cfg["queries"] != "hindsight" and cfg["query_count"]:
        raise ValueError("kv query_count applies only to hindsight queries; summary probes use all tokens")
    if not math.isfinite(cfg["ridge"]) or cfg["ridge"] <= 0:
        raise ValueError("kv ridge must be finite and positive")
    return cfg
