"""mini-swe-agent terminal episodes with token-exact, trainable compaction.

Harbor owns sandboxing and grading. Standalone terminal use needs no Miles/GPU
dependency; optional candidate collection uses Miles validation when enabled.
The decoder returns the actual sampled token IDs and log probabilities.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode
from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import DefaultAgent, FormatError, NonTerminatingException, Submitted

if TYPE_CHECKING:
    import numpy as np

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

    def with_overrides(self, overrides: dict) -> TerminalConfig:
        # Per-episode values win over launch defaults; unknown fields are errors.
        return replace(self, **overrides)


@dataclass(frozen=True)
class Completion:
    tokens: list[int]
    logprobs: list[float]
    text: str
    finish_reason: str = "stop"
    topk_token_ids: np.ndarray | None = None
    topk_log_probs: np.ndarray | None = None


@dataclass(frozen=True)
class _CompletedPrefix:
    canonical: list[int]
    sampled: list[int]


@dataclass
class TerminalTrace:
    segments: list[Segment] = field(default_factory=list)
    # The critic's chronological stream. Only explicitly compacted actor
    # resume prompts are replaced; template-induced segment breaks retain
    # their entire prompt because it can contain previously unseen observations.
    critic_segments: list[Segment] = field(default_factory=list)
    next_critic_prompt: list[int] | None = None
    critic_memory_exact: bool = True
    logprobs: list[list[float]] = field(default_factory=list)
    topk_token_ids: list[np.ndarray] | None = None
    topk_log_probs: list[np.ndarray] | None = None
    calls: list[dict] = field(default_factory=list)
    n_compactions: int = 0
    n_format_errors: int = 0
    n_commands: int = 0
    generated_tokens: int = 0
    exit_status: str = "Running"
    truncated: bool = False

    def _completion_topk(self, result: Completion) -> tuple[np.ndarray, np.ndarray] | None:
        ids, logps = result.topk_token_ids, result.topk_log_probs
        if ids is None and logps is None:
            if self.topk_token_ids is not None:
                raise ValueError("Every terminal completion must carry the same top-k candidate side channel")
            return None
        if ids is None or logps is None:
            raise ValueError("Terminal top-k candidate IDs and log probabilities must both be present")
        if self.segments and self.topk_token_ids is None:
            raise ValueError("Terminal top-k candidates cannot be enabled partway through an episode")
        # NumPy and Miles are optional until the top-k feature is requested.
        import numpy as np

        ids, logps = np.asarray(ids), np.asarray(logps)
        if ids.ndim != 2 or ids.shape != logps.shape or ids.shape[0] != len(result.tokens) or ids.shape[1] <= 0:
            raise ValueError("Terminal top-k candidate arrays must have shape [generated tokens, candidate count]")
        if not np.issubdtype(ids.dtype, np.integer) or ((ids < -1) | (ids > np.iinfo(np.int32).max)).any():
            raise ValueError("Terminal top-k candidate IDs must be nonnegative int32 IDs or -1 padding")
        if any(
            isinstance(token, bool)
            or not isinstance(token, (int, np.integer))
            or not 0 <= token <= np.iinfo(np.int32).max
            for token in result.tokens
        ):
            raise ValueError("Terminal top-k collection requires nonnegative sampled int32 token IDs")
        if any(logp > 0 for logp in result.logprobs):
            raise ValueError("Terminal sampled-token log probabilities must be nonpositive")
        if not (np.issubdtype(logps.dtype, np.floating) or np.issubdtype(logps.dtype, np.integer)):
            raise ValueError("Terminal top-k candidate log probabilities must be numeric")
        valid = ids >= 0
        if not np.isfinite(logps[valid]).all() or (logps[valid] > 0).any() or not np.isneginf(logps[~valid]).all():
            raise ValueError(
                "Terminal top-k candidate log probabilities must be finite and nonpositive; padding is -inf"
            )
        if (np.exp(np.where(valid, logps, -np.inf)).sum(-1) > 1.00001).any():
            raise ValueError("Terminal top-k candidate probability mass exceeds one")
        if self.topk_token_ids is not None and ids.shape[1] != self.topk_token_ids[0].shape[1]:
            raise ValueError("Terminal top-k candidate count changed during the episode")
        # The candidate path is a Miles training feature. Keep its optional
        # framework dependency out of standalone terminal use when disabled.
        from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
        from miles.utils.types import Sample

        validate_rollout_topk_logprobs_sample(
            Sample(
                tokens=result.tokens,
                response_length=len(result.tokens),
                loss_mask=[1] * len(result.tokens),
                rollout_log_probs=result.logprobs,
                rollout_topk_token_ids=ids,
                rollout_topk_log_probs=logps,
            ),
            ids.shape[1],
        )
        with np.errstate(over="ignore"):
            packed_logps = logps.astype(np.float32, copy=True)
        if not np.isfinite(packed_logps[valid]).all():
            raise ValueError("Terminal top-k candidate log probabilities must be representable in float32")
        return ids.astype(np.int32, copy=True), packed_logps

    def _record_topk(self, candidates: tuple[np.ndarray, np.ndarray] | None, *, gap: int | None) -> None:
        if candidates is None:
            return
        import numpy as np

        if self.topk_token_ids is None:
            self.topk_token_ids, self.topk_log_probs = [], []
        for segments, values, fill in zip(
            (self.topk_token_ids, self.topk_log_probs), candidates, (-1, -np.inf), strict=True
        ):
            if gap is None:
                segments.append(values)
            else:
                padding = np.full((gap, values.shape[1]), fill, dtype=values.dtype)
                segments[-1] = np.concatenate((segments[-1], padding, values))

    def record(self, prompt: list[int], result: Completion, *, kind: str):
        if len(result.tokens) != len(result.logprobs) or not result.tokens:
            raise ValueError("Decoder must return nonempty, aligned token IDs and log probabilities")
        if any(lp is None or not math.isfinite(lp) for lp in result.logprobs):
            raise ValueError("Decoder log probabilities must be finite")
        candidates = self._completion_topk(result)
        # Merge only exact token prefixes. Templates that remove old thinking
        # start a new segment; never retokenize sampled tokens for training.
        previous = self.segments[-1] if self.segments else None
        merged = previous is not None and prompt[: len(previous.tokens)] == previous.tokens
        if merged:
            gap = prompt[len(previous.tokens) :]
            previous.tokens.extend(gap + result.tokens)
            previous.response_length += len(gap) + len(result.tokens)
            previous.loss_mask.extend([0] * len(gap) + [1] * len(result.tokens))
            self.logprobs[-1].extend([0.0] * len(gap) + result.logprobs)
        else:
            if previous is not None and self.next_critic_prompt is None:
                # Without a recorded compaction boundary this prompt can contain
                # both repeated history and unseen observations. Keep the actor
                # sample intact; full_detach rejects this unauditable replay.
                self.critic_memory_exact = False
            self.segments.append(Segment(prompt + result.tokens, len(result.tokens), [1] * len(result.tokens)))
            self.logprobs.append(list(result.logprobs))
        if len(self.critic_segments) < len(self.segments):
            critic_prompt = prompt if self.next_critic_prompt is None else self.next_critic_prompt
            self.critic_segments.append(
                Segment(list(critic_prompt) + result.tokens, len(result.tokens), [1] * len(result.tokens))
            )
        else:
            own = self.critic_segments[-1]
            own.tokens.extend(gap + result.tokens)
            own.response_length += len(gap) + len(result.tokens)
            own.loss_mask.extend([0] * len(gap) + [1] * len(result.tokens))
        self.next_critic_prompt = None
        self._record_topk(candidates, gap=len(gap) if merged else None)
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
        memory_segments = [asdict(s) for s in self.critic_segments]
        samples = [
            dict(
                tokens=seg.tokens,
                response_length=seg.response_length,
                loss_mask=seg.loss_mask,
                rollout_log_probs=lp,
                critic_tokens=critic.tokens,
                critic_response_length=critic.response_length,
                critic_loss_mask=critic.loss_mask,
                critic_span=span,
                critic_segments=memory_segments,
                critic_memory_exact=self.critic_memory_exact,
            )
            for seg, lp, span in zip(self.segments, self.logprobs, spans, strict=True)
        ]
        if self.topk_token_ids is not None:
            for sample, ids, logps in zip(samples, self.topk_token_ids, self.topk_log_probs, strict=True):
                sample.update(rollout_topk_token_ids=ids, rollout_topk_log_probs=logps)
        return samples


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
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_ids = frozenset([eos] if isinstance(eos, int) else eos or ())
        self._completed_prefix: _CompletedPrefix | None = None

    def _encode(self, messages: list[dict], *, add_generation_prompt: bool = True) -> list[int]:
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        return list(
            self.tokenizer.apply_chat_template(
                clean,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
                enable_thinking=self.options.enable_thinking,
            )
        )

    def _prompt(self, messages: list[dict]) -> list[int]:
        canonical = self._encode(messages)
        previous = self._completed_prefix
        if previous is not None and canonical[: len(previous.canonical)] == previous.canonical:
            # Preserve the IDs actually sampled, including noncanonical BPE
            # splits. The suffix starts immediately after the completed EOS;
            # template newlines, observations and generation glue stay intact.
            return previous.sampled + canonical[len(previous.canonical) :]
        return canonical

    def _remember_completion(self, messages: list[dict], prompt: list[int], result: Completion):
        self._completed_prefix = None
        if result.tokens[-1] not in self._eos_token_ids:
            return  # No proven completed-message boundary: do not guess one.
        completed = self._encode(
            messages + [{"role": "assistant", "content": result.text}], add_generation_prompt=False
        )
        end = next((i for i in range(len(completed) - 1, -1, -1) if completed[i] in self._eos_token_ids), None)
        if end is None or self.tokenizer.decode(completed[end + 1 :]).strip():
            return  # A supported completed render ends in EOS plus whitespace.
        self._completed_prefix = _CompletedPrefix(completed[: end + 1], prompt + result.tokens)

    def render_template(self, template: str, **kwargs) -> str:
        # DefaultAgent's renderer asks model/environment for system metadata;
        # our async adapters deliberately expose only decoding and execution.
        return Template(template, undefined=StrictUndefined).render(**kwargs)

    async def _complete(self, messages: list[dict], *, limit: int, kind: str) -> Completion:
        prompt = self._prompt(messages)
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
        self._remember_completion(messages, prompt, result)
        return result

    async def _compact(self):
        if self.options.compaction == "none" or len(self.messages) <= 2:
            return
        current_size = len(self._prompt(self.messages))
        # Trigger early enough for another full action or the summary request.
        request = self.messages + [{"role": "user", "content": _SUMMARIZE}]
        summary_overhead = len(self._prompt(request)) - current_size
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
        # A retained history must not receive another system/task/summary copy.
        # In keep-last mode the latest observation has not reached record() yet;
        # preserve its exact suffix before replacing the copied resume prompt.
        unseen = []
        if self.options.compaction == "keep_last_turns" and self.trace.segments:
            prompt = self._prompt(self.messages)
            previous = self.trace.segments[-1].tokens
            if prompt[: len(previous)] != previous:
                raise ValueError("Cannot preserve unseen observations across non-prefix terminal compaction")
            unseen = prompt[len(previous) :]
        continuation = self._encode([{"role": "user", "content": "Continue the task using the preceding history."}])
        self.trace.next_critic_prompt = unseen if unseen else continuation
        self.messages = replacement
        self._completed_prefix = None  # Explicit compaction starts a fresh actor context.
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
        self._completed_prefix = None
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
