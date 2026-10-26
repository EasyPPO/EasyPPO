"""mini-swe-agent terminal episodes with token-exact, trainable compaction.

Harbor owns sandboxing and grading. This module has no Miles/GPU dependency;
its decoder returns the actual sampled token IDs and log probabilities.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import DefaultAgent, FormatError, NonTerminatingException, Submitted

from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode

# Format example from mini-swe-agent v1.17.1 config/mini.yaml.
_SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.

Each command runs in a new shell; use explicit paths or cd in each command.
After the bash block, stop and wait for the actual command output before deciding the next action.
To finish, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Do not combine this submission command with any other command."""
# mini-swe-agent v1.17.1 config/mini.yaml; keep its existing bash protocol.
_FORMAT_ERROR = """Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.
If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
Else, please format your response exactly as follows:

<response_example>
Here are some thoughts about why you want to perform the action.

```bash
<action>
```
</response_example>

Note: In rare cases, if you need to reference a similar format in your command, you might have
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ```bash.
"""
_SUMMARIZE = """Summarize progress for a fresh context continuing this task. Include modified files,
commands and results, failures, and remaining work. Do not execute a command or
claim the task is complete. Return only the summary."""
_OBSERVATION = """<returncode>{{output.returncode}}</returncode>
<output>{{output.output}}</output>"""


@dataclass(frozen=True)
class TerminalConfig:
    enable_thinking: bool = True
    max_steps: int = 100
    max_context_tokens: int = 32768
    max_new_tokens: int = 4096
    max_total_tokens: int = 65536
    compaction: Literal["summary", "keep_last_turns", "none"] = "summary"
    compact_threshold: int = 24576
    summary_tokens: int = 2048
    keep_last_turns: int = 2
    command_timeout: int = 120
    max_output_chars: int = 16000

    def __post_init__(self):
        if type(self.enable_thinking) is not bool:
            raise ValueError("terminal enable_thinking must be a boolean")
        for name in (
            "max_steps",
            "max_context_tokens",
            "max_new_tokens",
            "max_total_tokens",
            "compact_threshold",
            "summary_tokens",
            "keep_last_turns",
            "command_timeout",
            "max_output_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"terminal {name} must be a positive integer")
        if self.compaction not in ("summary", "keep_last_turns", "none"):
            raise ValueError(f"Unknown terminal compaction: {self.compaction}")
        if self.compact_threshold >= self.max_context_tokens:
            raise ValueError("terminal compact_threshold must be smaller than max_context_tokens")
        if self.max_new_tokens >= self.max_context_tokens or self.summary_tokens >= self.max_context_tokens:
            raise ValueError("terminal generation limits must be smaller than max_context_tokens")

    def with_overrides(self, overrides: dict) -> "TerminalConfig":
        # Per-episode values win over launch defaults; unknown fields are errors.
        return replace(self, **overrides)


@dataclass(frozen=True)
class Completion:
    tokens: list[int]
    logprobs: list[float]
    text: str
    finish_reason: str = "stop"


@dataclass
class TerminalTrace:
    segments: list[Segment] = field(default_factory=list)
    logprobs: list[list[float]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    n_compactions: int = 0
    n_format_errors: int = 0
    n_commands: int = 0
    generated_tokens: int = 0
    exit_status: str = "Running"
    truncated: bool = False

    def record(self, prompt: list[int], result: Completion, *, kind: str):
        if len(result.tokens) != len(result.logprobs) or not result.tokens:
            raise ValueError("Decoder must return nonempty, aligned token IDs and log probabilities")
        if any(lp is None or not math.isfinite(lp) for lp in result.logprobs):
            raise ValueError("Decoder log probabilities must be finite")
        # Merge only exact token prefixes. Templates that remove old thinking
        # start a new segment; never retokenize sampled tokens for training.
        previous = self.segments[-1] if self.segments else None
        if previous is not None and prompt[: len(previous.tokens)] == previous.tokens:
            gap = prompt[len(previous.tokens) :]
            previous.tokens.extend(gap + result.tokens)
            previous.response_length += len(gap) + len(result.tokens)
            previous.loss_mask.extend([0] * len(gap) + [1] * len(result.tokens))
            self.logprobs[-1].extend([0.0] * len(gap) + result.logprobs)
        else:
            self.segments.append(Segment(prompt + result.tokens, len(result.tokens), [1] * len(result.tokens)))
            self.logprobs.append(list(result.logprobs))
        self.generated_tokens += len(result.tokens)
        self.calls.append(
            {
                "kind": kind,
                "prompt_tokens": len(prompt),
                "output_tokens": len(result.tokens),
                "finish_reason": result.finish_reason,
            }
        )

    def sample_fields(self) -> list[dict]:
        if not self.segments:
            return []
        critic, spans = critic_episode(self.segments)
        return [
            dict(
                tokens=seg.tokens,
                response_length=seg.response_length,
                loss_mask=seg.loss_mask,
                rollout_log_probs=lp,
                critic_tokens=critic.tokens,
                critic_response_length=critic.response_length,
                critic_loss_mask=critic.loss_mask,
                critic_span=span,
            )
            for seg, lp, span in zip(self.segments, self.logprobs, spans, strict=True)
        ]


class EpisodeLimit(Exception):
    """An agent budget was reached; verify the work completed so far."""


class TerminalAgent(DefaultAgent):
    """Async orchestration around mini's templates, parser, recovery and submission.

    Async execution lets Harbor cancel model/command calls before tearing down
    the sandbox; no background agent thread survives a timeout.
    """

    def __init__(self, *, tokenizer, decode, execute, config: TerminalConfig):
        super().__init__(
            model=None,
            env=None,
            system_template=_SYSTEM,
            cost_limit=0,
            instance_template="{{task}}",
            action_observation_template=_OBSERVATION,
            format_error_template=_FORMAT_ERROR,
        )
        self.tokenizer = tokenizer
        self.decode = decode
        self.execute = execute
        self.options = config
        self.trace = TerminalTrace()
        self.original_messages: list[dict] = []

    def _encode(self, messages: list[dict]) -> list[int]:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        return list(
            self.tokenizer.apply_chat_template(
                clean,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=self.options.enable_thinking,
            )
        )

    def render_template(self, template: str, **kwargs) -> str:
        # DefaultAgent's renderer asks model/environment for system metadata;
        # our async adapters deliberately expose only decoding and execution.
        return Template(template, undefined=StrictUndefined).render(**kwargs)

    async def _complete(self, messages: list[dict], *, limit: int, kind: str) -> Completion:
        prompt = self._encode(messages)
        limit = min(
            limit,
            self.options.max_context_tokens - len(prompt),
            self.options.max_total_tokens - self.trace.generated_tokens,
        )
        if limit <= 0:
            raise EpisodeLimit("ContextOrTokenLimit")
        result = await self.decode(prompt, limit)
        if len(result.tokens) > limit:
            raise ValueError("Decoder exceeded the requested token limit")
        self.trace.record(prompt, result, kind=kind)
        if result.finish_reason == "length":
            raise EpisodeLimit("GenerationLimit")
        if result.finish_reason != "stop":
            raise RuntimeError(f"Decoder stopped unexpectedly: {result.finish_reason}")
        return result

    async def _compact(self):
        if self.options.compaction == "none" or len(self.messages) <= 2:
            return
        current_size = len(self._encode(self.messages))
        # Trigger early enough for another full action or the summary request.
        request = self.messages + [{"role": "user", "content": _SUMMARIZE}]
        summary_overhead = len(self._encode(request)) - current_size
        reserve = max(self.options.max_new_tokens, self.options.summary_tokens + summary_overhead)
        threshold = min(self.options.compact_threshold, self.options.max_context_tokens - reserve)
        if current_size < threshold:
            return
        if self.options.compaction == "summary":
            summary = await self._complete(request, limit=self.options.summary_tokens, kind="summary")
            replacement = self.original_messages + [{"role": "user", "content": "Progress summary:\n" + summary.text}]
        else:
            # Retain whole assistant/observation turns, never a dangling tool result.
            starts = [i for i, m in enumerate(self.messages[2:], 2) if m["role"] == "assistant"]
            if not starts:
                raise EpisodeLimit("CannotCompact")
            first = starts[max(0, len(starts) - self.options.keep_last_turns)]
            replacement = self.original_messages + self.messages[first:]
        if len(self._encode(replacement)) >= current_size:
            raise EpisodeLimit("CompactionDidNotShrink")
        self.messages = replacement
        self.trace.n_compactions += 1

    async def _step(self):
        await self._compact()
        response = await self._complete(self.messages, limit=self.options.max_new_tokens, kind="action")
        # Reasoning is retained in the recorded tokens, but is not executable.
        content = response.text.split("</think>")[-1]
        self.add_message("assistant", response.text)
        try:
            action = self.parse_action({"content": content})
        except FormatError:
            self.trace.n_format_errors += 1
            self.trace.calls[-1]["action_status"] = "format_error"
            raise
        self.trace.n_commands += 1
        self.trace.calls[-1]["action_status"] = "executed"
        try:
            output = await self.execute(action["action"], self.options.command_timeout)
        except TimeoutError:
            self.add_message("user", "The command timed out. Try a shorter, non-interactive command.")
            return
        self.has_finished(output)
        text = output["output"]
        cap = self.options.max_output_chars
        if len(text) > cap:
            tail = text[-(cap // 2) :] if cap // 2 else ""
            text = text[: (cap + 1) // 2] + "\n[output truncated]\n" + tail
        self.add_message("user", self.render_template(_OBSERVATION, output={**output, "output": text}))

    async def run_async(self, task: str) -> TerminalTrace:
        self.messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": task}]
        self.original_messages = list(self.messages)
        try:
            for _ in range(self.options.max_steps):
                try:
                    await self._step()
                except NonTerminatingException as exc:
                    self.add_message("user", str(exc))
            raise EpisodeLimit("StepLimit")
        except Submitted:
            self.trace.exit_status = "Submitted"
        except EpisodeLimit as exc:
            self.trace.exit_status = str(exc)
            self.trace.truncated = True
        return self.trace
