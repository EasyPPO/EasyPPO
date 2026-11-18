"""Harbor verifier copied into prepared FrontierSmith tasks.

Iterative tasks return the highest shared-judge submission score / 100.
Legacy final-file tasks submit once. A judge error verdict scores zero.
Iterative submissions with judge request failures score zero.
"""

import argparse
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("FRONTIERCS_JUDGE_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _wait_ready(judge_url: str, *, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            _request(f"{judge_url}/health")
            return
        except (urllib.error.URLError, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError("FrontierSmith judge did not become ready") from None
            time.sleep(2)


def submit_solution(
    judge_url: str, problem_id: str, code: str, *, timeout: float, wait_ready: bool = True, poll_s: float = 2
) -> dict:
    if not code.strip():
        return {"status": "error", "error": "Error: compile failed: empty program"}
    judge_url = judge_url.rstrip("/")
    if wait_ready:
        _wait_ready(judge_url)
    submission = _request(f"{judge_url}/submit", {"pid": problem_id, "lang": "cpp", "code": code})
    deadline = time.monotonic() + timeout
    while True:
        result = _request(f"{judge_url}/result/{submission['sid']}")
        if result.get("status") in ("done", "error"):
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FrontierSmith judge timed out on {problem_id}")
        time.sleep(poll_s)


def _reward(result: dict) -> float:
    if result.get("status") == "error":
        return 0.0
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
        result = submit_solution(judge_url, problem_id, code, timeout=timeout)
    (logs / "judge-result.json").write_text(json.dumps(result, indent=2))
    reward = _reward(result) if code.strip() else 0.0
    # Harbor's reward JSON accepts only numeric values. Keep diagnostics separate.
    (logs / "reward.json").write_text(json.dumps({"reward": reward}))
    (logs / "reward.txt").write_text(str(reward))


def _submit(*, solution: Path, ledger: Path, judge_url: str, problem_id: str, timeout: float) -> dict:
    code = solution.read_text(encoding="utf-8")
    try:
        result = submit_solution(judge_url, problem_id, code, timeout=timeout)
        _reward(result)
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {"status": "error", "error": f"Judge failure: {type(exc).__name__}: {exc}"}
    submissions = json.loads(ledger.read_text(encoding="utf-8"))
    submissions.append(result)
    pending = ledger.with_suffix(".tmp")
    pending.write_text(json.dumps(submissions), encoding="utf-8")
    pending.replace(ledger)
    return result


def _verify_submissions(*, ledger: Path, logs: Path) -> None:
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("reward.txt", "reward.json"):
        (logs / name).unlink(missing_ok=True)
    submissions = json.loads(ledger.read_text(encoding="utf-8"))
    write_submission_rewards(submissions, logs)


def write_submission_rewards(submissions: list[dict], logs: Path) -> dict:
    if not isinstance(submissions, list):
        raise ValueError("FCS submissions must be a list")
    reward = max((_reward(result) for result in submissions), default=0.0)
    (logs / "judge-result.json").write_text(json.dumps(submissions))
    (logs / "reward.json").write_text(json.dumps({"reward": reward}))
    (logs / "reward.txt").write_text(str(reward))
    return {"reward": reward}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    ledger = Path("/app/fcs-submissions.json")
    if args.submit:
        result = _submit(
            solution=Path("/app/solution.cpp"),
            ledger=ledger,
            judge_url=os.environ["JUDGE_URL"],
            problem_id=os.environ["PROBLEM_ID"],
            timeout=float(os.environ.get("MAX_POLL_TIME", "600")),
        )
        print(json.dumps(result))
        return
    if os.environ.get("FCS_ITERATIVE_SUBMISSIONS") == "1":
        _verify_submissions(ledger=ledger, logs=Path("/logs/verifier"))
        return
    _verify(
        solution=Path("/app/solution.cpp"),
        logs=Path("/logs/verifier"),
        judge_url=os.environ["JUDGE_URL"],
        problem_id=os.environ["PROBLEM_ID"],
        timeout=float(os.environ.get("MAX_POLL_TIME", "600")),
    )


if __name__ == "__main__":
    main()
