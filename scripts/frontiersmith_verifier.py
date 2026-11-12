"""Harbor verifier copied into prepared FrontierSmith tasks.

Uses the shared judge's score / 100, configured by task verifier JUDGE_URL.
Infrastructure failures leave no
reward file, so Harbor rejects the trial instead of training on a false zero.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _wait_ready(judge_url: str, *, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            _request(f"{judge_url}/problems")
            return
        except (urllib.error.URLError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError("FrontierSmith judge did not become ready") from None
            time.sleep(2)


def _judge(judge_url: str, problem_id: str, code: str, *, timeout: float) -> dict:
    _wait_ready(judge_url)
    submission = _request(f"{judge_url}/submit", {"pid": problem_id, "lang": "cpp", "code": code})
    deadline = time.monotonic() + timeout
    while True:
        result = _request(f"{judge_url}/result/{submission['sid']}")
        if result.get("status") in ("done", "error"):
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FrontierSmith judge timed out on {problem_id}")
        time.sleep(2)


def _reward(result: dict) -> float:
    if result.get("status") == "error":
        # At the pinned upstream revision contestant C++ errors have this exact
        # prefix; checker compilation and service failures have other prefixes.
        if str(result.get("error", "")).startswith("Error: compile failed: "):
            return 0.0
        raise RuntimeError(f"FrontierSmith judge failure: {result}")
    if result.get("status") != "done":
        raise ValueError("FrontierSmith judge returned no final verdict")
    score = result.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError(f"Invalid FrontierSmith judge score: {score!r}")
    return score / 100.0


def _verify(*, solution: Path, logs: Path, judge_url: str, problem_id: str, timeout: float) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    # A failed repeated invocation must not retain an earlier successful reward.
    for name in ("reward.txt", "reward.json"):
        (logs / name).unlink(missing_ok=True)
    code = solution.read_text() if solution.is_file() else ""
    if not code.strip():
        result = {"status": "missing_solution"}
    else:
        result = _judge(judge_url.rstrip("/"), problem_id, code, timeout=timeout)
    (logs / "judge-result.json").write_text(json.dumps(result, indent=2))
    reward = _reward(result) if code.strip() else 0.0
    # Harbor's reward JSON accepts only numeric values. Keep diagnostics separate.
    (logs / "reward.json").write_text(json.dumps({"reward": reward}))
    (logs / "reward.txt").write_text(str(reward))


def main() -> None:
    _verify(
        solution=Path("/app/solution.cpp"),
        logs=Path("/logs/verifier"),
        judge_url=os.environ["JUDGE_URL"],
        problem_id=os.environ["PROBLEM_ID"],
        timeout=float(os.environ.get("MAX_POLL_TIME", "600")),
    )


if __name__ == "__main__":
    main()
