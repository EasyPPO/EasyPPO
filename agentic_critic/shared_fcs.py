"""Run submit-only Frontier-CS episodes against the official shared judge API."""

import asyncio
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from harbor.models.task.task import Task

from agentic_critic.fcs_terminal import FcsSubmissionAgent, judge_feedback
from miles.utils.http_utils import GeneralHttpClientProvider
from scripts.frontiersmith_verifier import write_submission_rewards


async def _submit(url: str, problem_id: str, code: str, timeout: float) -> dict:
    if not code.strip():
        return {"status": "error", "error": "Error: compile failed: empty program"}
    client = GeneralHttpClientProvider.client()
    token = os.environ.get("FRONTIERCS_JUDGE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with asyncio.timeout(timeout):
        # A submit can succeed even if its response is lost. Issue exactly one POST.
        response = await client.post(
            f"{url}/submit",
            json={"pid": problem_id, "lang": "cpp", "code": code},
            headers=headers,
            follow_redirects=False,
            timeout=10,
        )
        response.raise_for_status()
        sid = response.json()["sid"]
        if type(sid) is not int or sid < 0:
            raise ValueError("Frontier-CS judge returned an invalid submission ID")
        while True:
            response = await client.get(f"{url}/result/{sid}", headers=headers, follow_redirects=False, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result["status"] in ("done", "error"):
                return result
            if result["status"] != "queued":
                raise ValueError(f"Unexpected Frontier-CS judge status: {result['status']!r}")
            await asyncio.sleep(2)


def _persist_episode(
    trial_dir: Path, agent: FcsSubmissionAgent, verdicts: list[dict], timing: dict, *, verified: bool
):
    agent_dir, verifier_dir = trial_dir / "agent", trial_dir / "verifier"
    agent_dir.mkdir(parents=True)
    verifier_dir.mkdir()
    agent.trace.write_trajectory(agent.messages, agent_dir)
    (trial_dir / "timing.json").write_text(json.dumps(timing))
    if verified:
        return write_submission_rewards(verdicts, verifier_dir)
    (verifier_dir / "judge-result.json").write_text(json.dumps(verdicts))
    return None


async def _finish_write(write):
    # Cancellation must drain the worker thread before the rollout owner can exit.
    task = asyncio.create_task(write)
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception:
            # Read the worker exception below, after resolving any owner cancellation.
            break
    try:
        result = task.result()
    except Exception as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def run_shared_judge_episode(
    *,
    task_path: Path,
    trials_dir: Path,
    judge_url: str | None,
    tokenizer,
    decode,
    config,
    max_submissions: int,
    max_turns: int,
):
    origin = urlsplit(judge_url or "")
    if (
        origin.scheme not in ("http", "https")
        or not origin.hostname
        or origin.username is not None
        or origin.password is not None
        or origin.query
        or origin.fragment
    ):
        raise ValueError("shared-judge requires --fcs-judge-url with an HTTP(S) base URL")
    url = judge_url.rstrip("/")
    task = await asyncio.to_thread(Task, task_path)
    env = task.config.verifier.env
    if env.get("FCS_ITERATIVE_SUBMISSIONS") != "1":
        raise ValueError("Reprepare FCS tasks for iterative submissions before training")
    problem_id = env["PROBLEM_ID"]
    timeout = float(env["MAX_POLL_TIME"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("MAX_POLL_TIME must be finite and positive")

    verdicts = []
    judge_seconds = 0.0

    async def submit(code):
        nonlocal judge_seconds
        started = time.monotonic()
        try:
            verdict = await _submit(url, problem_id, code, timeout)
            feedback = judge_feedback(verdict)
            verdicts.append(verdict)
            return feedback
        except Exception as exc:
            exc.add_note(f"Frontier-CS problem {problem_id}, submission {len(verdicts) + 1}")
            raise
        finally:
            judge_seconds += time.monotonic() - started

    agent = FcsSubmissionAgent(
        tokenizer=tokenizer,
        decode=decode,
        config=config,
        submit=submit,
        max_submissions=max_submissions,
        max_turns=max_turns,
    )
    trial_dir = trials_dir / f"{task_path.name}__{uuid4().hex}"
    started_at = datetime.now(UTC)
    deadline = asyncio.timeout(task.config.agent.timeout_sec)
    try:
        async with deadline:
            await agent.run_async(task.instruction)
    except TimeoutError:
        if not deadline.expired():
            agent.trace.exit_status = "AgentError"
            raise
        agent.trace.exit_status = "AgentTimeoutError"
        agent.trace.truncated = True
    except asyncio.CancelledError:
        agent.trace.exit_status = "Cancelled"
        agent.trace.truncated = True
        raise
    except Exception:
        agent.trace.exit_status = "AgentError"
        raise
    finally:
        finished_at = datetime.now(UTC)
        write = asyncio.to_thread(
            _persist_episode,
            trial_dir,
            agent,
            verdicts,
            {
                "problem_id": problem_id,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "judge_seconds": judge_seconds,
                "exit_status": agent.trace.exit_status,
            },
            verified=agent.trace.exit_status not in ("AgentError", "Cancelled"),
        )
        rewards = await _finish_write(write)
    return agent.trace, rewards, str(trial_dir)
