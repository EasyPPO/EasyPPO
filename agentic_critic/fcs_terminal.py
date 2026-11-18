"""Submit-only Frontier-CS interaction on the terminal trajectory recorder."""

from dataclasses import replace

from minisweagent.agents.default import Submitted

from agentic_critic.fcs_env import (
    COMPACT_PROMPT,
    NEXT_PROMPT,
    NO_CODE_PROMPT,
    RESUME_PROMPT,
    TASK_PROMPT,
    answer_text,
    extract_code,
    format_feedback,
)
from agentic_critic.fcs_judge import CaseResult, JudgeResult
from agentic_critic.terminal_env import TerminalAgent
from scripts.frontiersmith_verifier import _reward


def judge_feedback(response: dict) -> JudgeResult:
    score = 100 * _reward(response)
    if response["status"] == "error":
        message = str(response.get("error", ""))
        if message.startswith("Error: compile failed: "):
            return JudgeResult(score, "Compile Error", compile_msg=message)
        return JudgeResult(score, "Judge Error")
    cases = [
        CaseResult(
            name=str(i),
            ratio=float(case.get("scoreRatio") or 0),
            status="Accepted" if case.get("scoreRatio") == 1 else "Wrong Answer",
            time_s=float(case.get("time") or 0) / 1e9,
        )
        for i, case in enumerate(response.get("cases") or [], 1)
    ]
    return JudgeResult(score, "done", cases=cases)


class FcsSubmissionAgent(TerminalAgent):
    def __init__(self, *, tokenizer, decode, submit, config, max_submissions: int, max_turns: int):
        if max_submissions < 1 or max_turns < 1:
            raise ValueError("FCS submission and turn limits must be positive")
        super().__init__(tokenizer=tokenizer, decode=decode, execute=None, config=replace(config, max_steps=max_turns))
        self.submit = submit
        self.max_submissions = max_submissions
        self.turns = 0

    def _initial_messages(self, task: str) -> list[dict]:
        self.task_prompt = TASK_PROMPT.format(statement=task)
        return [{"role": "user", "content": self.task_prompt}]

    def _summary_request(self) -> list[dict]:
        # Replace the next-solution instruction with the reference summary prompt.
        feedback = self.messages[-1]["content"].removesuffix("\n" + NEXT_PROMPT)
        return self.messages[:-1] + [{"role": "user", "content": feedback + "\n\n" + COMPACT_PROMPT}]

    def _should_compact(self, current_size: int, reserve: int) -> bool:
        # The reference triggers strictly above the soft threshold. Preserve
        # the harness's hard-context reserve even when the threshold is higher.
        return (
            current_size > self.options.compact_threshold or current_size >= self.options.max_context_tokens - reserve
        )

    def _summary_resume(self, summary: str) -> list[dict]:
        summary = answer_text(summary)
        if eos := getattr(self.tokenizer, "eos_token", None):
            summary = summary.replace(eos, "")
        return [{"role": "user", "content": RESUME_PROMPT.format(task=self.task_prompt, summary=summary.strip())}]

    async def _step(self):
        await self._compact()
        response = await self._complete(self.messages, limit=self.options.max_new_tokens, kind="action")
        self.add_message("assistant", response.text)
        self.turns += 1
        code = extract_code(response.text)
        if code is None:
            self.trace.n_format_errors += 1
            self.trace.calls[-1]["action_status"] = "format_error"
            feedback = NO_CODE_PROMPT
        else:
            result = await self.submit(code)
            self.trace.submission_scores.append(result.score)
            self.trace.calls[-1]["action_status"] = "submitted"
            feedback = format_feedback(
                result, len(self.trace.submission_scores), self.max_submissions, max(self.trace.submission_scores)
            )
        if len(self.trace.submission_scores) >= self.max_submissions or self.turns >= self.options.max_steps:
            raise Submitted()
        self.add_message("user", feedback + "\n" + NEXT_PROMPT)
