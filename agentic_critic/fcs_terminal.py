"""Submit-only Frontier-CS interaction on the terminal trajectory recorder."""

from dataclasses import replace

from minisweagent.agents.default import Submitted

from agentic_critic.fcs_env import (
    FCS_REASONING_EFFORT,
    FCS_SUMMARY_EFFORT,
    NEXT_PROMPT,
    NO_CODE_PROMPT,
    RESUME_PROMPT,
    TASK_PROMPT,
    TRUNCATED_NO_SUBMISSION_PROMPT,
    compact_messages,
    extract_code,
    format_feedback,
    summary_text,
)
from agentic_critic.terminal_env import Completion, TerminalAgent


class FcsSubmissionAgent(TerminalAgent):
    def __init__(self, *, tokenizer, decode, submit, config, max_submissions: int, max_turns: int):
        if max_submissions < 1 or max_turns < 1:
            raise ValueError("FCS submission and turn limits must be positive")
        super().__init__(tokenizer=tokenizer, decode=decode, execute=None, config=replace(config, max_steps=max_turns))
        self.submit = submit
        self.max_submissions = max_submissions
        self.turns = 0

    def _template_options(self, kind: str) -> dict:
        return {
            "enable_thinking": self.options.fcs_summary_thinking if kind == "summary" else self.options.enable_thinking,
            "reasoning_effort": FCS_SUMMARY_EFFORT if kind == "summary" else FCS_REASONING_EFFORT,
        }

    def _prompt(self, messages: list[dict], *, kind: str = "action") -> list[int]:
        prompt = super()._prompt(messages)
        if kind == "summary":
            # Qwen encodes low effort as a system instruction. Apply it to the
            # two new user messages so that previously sampled IDs remain an exact prefix.
            turn = self._encode(messages[-2:])
            if prompt[-len(turn):] != turn:
                raise ValueError("Cannot locate the FCS summary turn in the chat template")
            return prompt[:-len(turn)] + self._encode(messages[-2:], kind="summary")
        return prompt

    def _record_completion(self, messages: list[dict], prompt: list[int], result: Completion, *, kind: str):
        previous = self._completed_prefix
        if previous is not None and prompt[: len(previous.sampled)] != previous.sampled:
            # The native 9B template rewrites earlier assistant thinking after
            # new feedback. Only that feedback and its generation glue are new
            # to the full-history critic. Verify both ends against the actor IDs.
            turn = self._encode(messages[-2:] if kind == "summary" else messages[-1:], kind=kind)
            start = len(prompt) - len(turn)
            if start > 0 and prompt[start:] == turn and messages[-1]["role"] == "user":
                end = next((i for i in range(start - 1, -1, -1) if prompt[i] in self._eos_token_ids), None)
                if end is not None and not self.tokenizer.decode(prompt[end + 1 : start]).strip():
                    self.trace.next_critic_prompt = prompt[end + 1 :]
        super()._record_completion(messages, prompt, result, kind=kind)

    def _initial_messages(self, task: str) -> list[dict]:
        self.task_prompt = TASK_PROMPT.format(statement=task, max_submissions=self.max_submissions)
        return [{"role": "user", "content": self.task_prompt}]

    def _summary_request(self) -> list[dict]:
        # Replace the next-solution instruction with the reference summary prompt.
        feedback = self.messages[-1]["content"].removesuffix("\n" + NEXT_PROMPT)
        return self.messages[:-1] + compact_messages(feedback)

    def _should_compact(self, current_size: int, reserve: int) -> bool:
        # The reference triggers strictly above the soft threshold. Preserve
        # the harness's hard-context reserve even when the threshold is higher.
        return (
            current_size > self.options.compact_threshold or current_size >= self.options.max_context_tokens - reserve
        )

    def _summary_resume(self, summary: str) -> list[dict]:
        # With thinking disabled the template already closed the block in the prompt.
        if self.options.fcs_summary_thinking:
            summary = summary_text(summary)
        if eos := getattr(self.tokenizer, "eos_token", None):
            summary = summary.replace(eos, "")
        return [{"role": "user", "content": RESUME_PROMPT.format(task=self.task_prompt, summary=summary.strip())}]

    async def _step(self):
        await self._compact()
        response = await self._complete(self.messages, limit=self.options.max_new_tokens, kind="action")
        self.add_message("assistant", response.text)
        self.turns += 1
        if response.finish_reason == "length":
            self.trace.calls[-1]["action_status"] = "truncated_no_submission"
            feedback = TRUNCATED_NO_SUBMISSION_PROMPT
        elif (code := extract_code(response.text)) is None:
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
