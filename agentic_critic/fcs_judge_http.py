"""Client for the official Frontier-CS judge server (go-judge based), running in Docker.

Returns the same `JudgeResult` shape as the local judge in `fcs_judge.py`, so the episode loop
does not care which one is behind it. Model-written code is only ever executed inside the
judge container's sandbox.
"""

import json
import time
import urllib.error
import urllib.request

from agentic_critic.fcs_judge import CaseResult, JudgeResult

MSG_CHARS = 2000


def _call(url: str, payload: dict | None = None, timeout: float = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def healthy(base_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def judge(base_url: str, problem_id: str, code: str, *, poll_s: float = 1.0, timeout_s: float = 900) -> JudgeResult:
    """Submit C++ code for `problem_id` and wait for the verdict."""
    base_url = base_url.rstrip("/")
    if not code.strip():
        # The server answers an empty program with HTTP 400, which used to end the whole episode with an error.
        # An empty file does not compile; the submission counts and scores 0.
        return JudgeResult(0.0, "Compile Error", compile_msg="empty program")
    sid = _call(f"{base_url}/submit", {"pid": problem_id, "lang": "cpp", "code": code})["sid"]
    deadline = time.monotonic() + timeout_s
    while True:
        result = _call(f"{base_url}/result/{sid}")
        if result.get("status") in ("done", "error"):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"judge did not finish submission {sid} for {problem_id} within {timeout_s}s")
        time.sleep(poll_s)

    if result["status"] == "error":
        # The server reports compile failures (and its own internal failures) this way.
        return JudgeResult(0.0, "Compile Error", compile_msg=str(result.get("error", ""))[-MSG_CHARS:])
    cases = [
        CaseResult(
            name=str(i + 1),
            ratio=float(c.get("scoreRatio") or 0.0),
            status="Accepted" if c.get("scoreRatio") == 1.0 else "Wrong Answer",
            time_s=float(c.get("time") or 0) / 1e9,
            msg=str(c.get("msg") or "")[-MSG_CHARS:],
        )
        for i, c in enumerate(result.get("cases") or [])
    ]
    return JudgeResult(float(result.get("score") or 0.0), "done", cases=cases)
