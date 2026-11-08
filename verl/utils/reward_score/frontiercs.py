# Copyright 2025 - Frontier-CS reward for VERL RL training
"""Score FrontierSmith and Frontier-CS C++ responses with the bundled HTTP judge."""

import os
import re
import time
from typing import Any, Optional

import requests

DEFAULT_JUDGE_URL = "http://localhost:8082"
POLL_INTERVAL = 2.0
MAX_WAIT = 300.0
FRONTIERCS_REWARD_SCALE = 100.0


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_cpp(response_text: str) -> str:
    """Extract C++ code from a raw model response."""
    if not response_text:
        return ""

    code = strip_think(response_text).strip()
    if not code:
        return ""

    matches = re.findall(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    if code.startswith("```cpp") or code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def normalize_frontiercs_reward(score_100: float) -> float:
    """Scale a Frontier-CS score from [0, 100] to the training range [0, 1]."""
    clipped_score = min(max(float(score_100), 0.0), FRONTIERCS_REWARD_SCALE)
    return clipped_score / FRONTIERCS_REWARD_SCALE


def _format_reward(raw_score: float) -> dict[str, float]:
    return {
        "score": normalize_frontiercs_reward(raw_score),
        "acc": raw_score,
        "raw_score": raw_score,
        "judge_raw_score": raw_score,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    judge_url: Optional[str] = None,
    **kwargs,
) -> dict[str, float]:
    """Return reward in [0, 1] and judge metrics in [0, 100].

    ``frontiersmith_*`` problem IDs select FrontierSmith; ``frontiercs_val_*`` selects
    Frontier-CS. Empty code, failed submissions and judge timeouts score zero.
    """
    if data_source != "frontiercs":
        raise ValueError(f"data_source must be 'frontiercs', got {data_source}")
    code = extract_cpp(solution_str)
    if not code:
        return _format_reward(0.0)

    extra_info = extra_info or {}
    url = (
        judge_url or extra_info.get("judge_url") or os.environ.get("FRONTIERCS_JUDGE_URL") or DEFAULT_JUDGE_URL
    ).rstrip("/")
    try:
        response = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": str(ground_truth), "lang": "cpp"},
            timeout=30,
        )
        response.raise_for_status()
        sid = response.json().get("sid")
        if not sid:
            return _format_reward(0.0)

        start = time.monotonic()
        while time.monotonic() - start < MAX_WAIT:
            response = requests.get(f"{url}/result/{sid}", timeout=10)
            if response.status_code != 404:
                response.raise_for_status()
                result = response.json()
                if result.get("status") == "done":
                    return _format_reward(float(result.get("score", 0)))
                if result.get("status") == "error":
                    return _format_reward(0.0)
            time.sleep(POLL_INTERVAL)
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return _format_reward(0.0)
    return _format_reward(0.0)
