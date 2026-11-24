"""Harbor Terminus-2 terminal protocol with EasyPPO's token-exact episode recorder."""

import re
import shlex
from dataclasses import replace

from tenacity import retry_if_not_exception_type

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.models.metric import UsageInfo

from agentic_critic.terminal_env import EpisodeLimit, TerminalAgent


ETL_SMOKE_TASK_ID = "task_002307_b945c419"


class _TerminusSession(TerminalAgent):
    async def _complete(self, messages, *, limit, kind):
        if self.trace.generated_tokens >= self.options.max_total_tokens:
            raise EpisodeLimit("EpisodeTokenLimit")
        try:
            return await super()._complete(messages, limit=limit, kind=kind)
        except EpisodeLimit as exc:
            if str(exc) == "ContextOrTokenLimit":
                raise ContextLengthExceededError("No context left for a completion") from exc
            raise

    def _summary_request(self):
        request = super()._summary_request()
        request[-1]["content"] += (
            " This turn is a handoff to another agent. Describe the current terminal state "
            "and distinguish completed work from proposed next steps. Pause the terminal XML protocol "
            "for this reply and write the handoff in plain text."
        )
        return request

    def _summary_resume(self, summary: str) -> list[dict]:
        # SGLang can include Qwen's reasoning and EOS in its text response.
        visible = summary.partition("<|im_end|>")[0].partition("<|endoftext|>")[0]
        if self.options.enable_thinking and self.trace.calls[-1]["finish_reason"] == "length" and "</think>" not in visible:
            visible = ""
        visible = visible.rsplit("</think>", 1)[-1].strip()
        if visible.startswith("<think>"):
            visible = ""
        if visible.startswith("<response>"):
            parsed = _TerminusXMLParser().parse_response(visible)
            if not parsed.error and (parsed.analysis or parsed.plan):
                visible = "\n".join(filter(None, (parsed.analysis, parsed.plan)))
        return super()._summary_resume(visible)

    def _record_completion(self, messages, prompt, result, *, kind):
        if result.finish_reason == "length" and not result.tokens:
            raise ContextLengthExceededError("Empty completion at the context limit")
        super()._record_completion(messages, prompt, result, kind=kind)


class _TerminusXMLParser(TerminusXMLPlainParser):
    def _find_top_level_tags(self, content):
        # TorchTitan's SafeTerminusXMLParser cleans only diagnostics, preserving keystrokes.
        clean = re.sub(r"<[^>]*>", lambda m: m[0] if m[0][1:-1].strip() else "", content)
        return super()._find_top_level_tags(clean)


class MilesTerminalLLM(BaseLLM):
    def bind(self, tokenizer, decode, config):
        # Record the complete sampled turn before Harbor handles a length stop.
        self.session = _TerminusSession(
            tokenizer=tokenizer, decode=decode, execute=None, config=replace(config, continue_on_length=True)
        )
        self._vocab = tokenizer.get_vocab()
        self._last_request = None
        self._prepared_request = None
        self._episode_budget_notified = False

    def get_model_context_limit(self):
        return self.session.options.max_context_tokens

    def get_model_output_limit(self):
        return self.session.options.max_new_tokens

    def _response_parts(self, tokens):
        # Match parse_qwen35 in PrimeIntellect-ai/renderers@2a7e77eb.
        # Harbor consumes the text block. The recorder keeps every sampled ID.
        vocab = self._vocab
        stop_ids = {vocab.get("<|im_end|>"), vocab.get("<|endoftext|>")}
        stop = next((i for i, token in enumerate(tokens) if token in stop_ids), len(tokens))
        ids = tokens[:stop]
        opening, closing = vocab.get("<think>"), vocab.get("</think>")
        decode = self.session.tokenizer.decode
        reasoning = None
        if closing in ids:
            end = ids.index(closing)
            reasoning = decode([token for token in ids[:end] if token != opening]).strip()
            ids = ids[end + 1:]
        elif opening in ids:
            return "", decode(ids[ids.index(opening) + 1:]).strip()
        tool_call = vocab.get("<tool_call>")
        if tool_call in ids:
            ids = ids[:ids.index(tool_call)]
        return decode(ids).strip(), reasoning

    async def call(self, prompt, message_history=None, **kwargs):
        session = self.session
        history = list(message_history or [])
        turn = {"role": "user", "content": prompt}
        request = history + [turn]
        if self._prepared_request != request:
            previous = self._last_request
            if previous is not None and history[:len(previous)] == previous and (
                len(history) > len(previous) and history[len(previous)]["role"] == "assistant"
                and all(m["role"] != "assistant" for m in history[len(previous) + 1:])
            ):
                # Harbor echoes visible text. Keep the original sampled reasoning and token IDs.
                messages = self._completed_messages + history[len(previous) + 1:] + [turn]
            else:
                messages = request.copy()
            if not session.original_messages:
                session.original_messages = messages.copy()
            session.messages = messages
            await session._compact()
            used = session.trace.generated_tokens
            budget = session.options.max_total_tokens
            if not self._episode_budget_notified and 2 * used >= budget:
                notice = (
                    f"[Episode budget: {used:,} of {budget:,} generated tokens used; "
                    f"{max(0, budget - used):,} remain across all future actions and summaries. "
                    "Prioritize verification and submission.]"
                )
                last = session.messages[-1]
                session.messages[-1] = {**last, "content": last["content"] + "\n\n" + notice}
                if session.options.compaction == "summary" and session.trace.next_critic_prompt is not None:
                    # The full-history critic replaces the actor resume prompt at this boundary.
                    session.trace.next_critic_prompt = session._encode([{"role": "user", "content": notice}])
                self._episode_budget_notified = True
            self._prepared_request = request

        result = await session._complete(session.messages, limit=self.get_model_output_limit(), kind="action")
        session.messages = session.messages + [{"role": "assistant", "content": result.text}]
        self._completed_messages = session.messages
        self._last_request = request
        self._prepared_request = None
        visible, reasoning = self._response_parts(result.tokens)
        recorded = session.trace.calls[-1]
        if result.finish_reason == "length":
            if session.trace.generated_tokens >= session.options.max_total_tokens:
                raise EpisodeLimit("EpisodeTokenLimit")
            if recorded["prompt_tokens"] + len(result.tokens) >= self.get_model_context_limit() - 64:
                raise ContextLengthExceededError("Turn reached the context limit")
            raise OutputLengthExceededError("Turn reached the output limit", truncated_response=visible)
        return LLMResponse(
            content=visible,
            reasoning_content=reasoning,
            usage=UsageInfo(prompt_tokens=recorded["prompt_tokens"], completion_tokens=len(result.tokens), cache_tokens=0, cost_usd=0),
        )


class MilesTerminus2(Terminus2):
    def _init_llm(self, **kwargs):
        return MilesTerminalLLM()

    def bind(self, *, tokenizer, decode, config):
        self._llm.bind(tokenizer, decode, config)
        if type(self._parser) is TerminusXMLPlainParser:
            self._parser = _TerminusXMLParser()
        self.trace = self._llm.session.trace

    async def setup(self, environment):
        await super().setup(environment)
        self._previous_pane = None
        self._session.get_incremental_output = self._observe_terminal

    async def _observe_terminal(self):
        # Read only lines since the preceding observation. Blank deltas, alternate
        # screens and scrollback resets use the current visible screen.
        target = shlex.quote(self._session._session_name)
        previous = self._previous_pane
        capture = f"tmux capture-pane -p -t {target}"
        command = f"""set -e
state=$(tmux display-message -p -t {target} '#{{history_size}} #{{cursor_y}} #{{alternate_on}} #{{pane_id}}')
set -- $state
printf '%s\n' "$state"
"""
        if previous:
            history, cursor, alternate, pane = previous
            if alternate == 0:
                command += f"""if [ "$3" = 0 ] && [ "$4" = {shlex.quote(pane)} ] && [ "$1" -ge {history} ]; then
    delta=$({capture} -S "$(({cursor} - ($1 - {history})))" && printf .) || exit 1
    delta=${{delta%.}}
    case "$delta" in *[![:space:]]*) printf 'new\n%s' "$delta"; exit 0;; esac
fi
"""
        command += f"printf 'screen\n'\n{capture}"
        result = await self._session.environment.exec(command, user=self._session._user)
        if result.return_code != 0:
            raise RuntimeError(f"Terminus could not observe tmux: {result.stderr}")
        header, mode, output = result.stdout.split("\n", 2)
        history, cursor, alternate, pane = header.split()
        self._previous_pane = (int(history), int(cursor), int(alternate), pane)
        label = "New Terminal Output" if mode == "new" else "Current Terminal Screen"
        return f"{label}:\n{output}"

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        query = Terminus2._query_llm
        # Episode budgets are terminal. Keep Harbor's transport retries and XML recovery.
        query = query.retry_with(retry=query.retry.retry & retry_if_not_exception_type(EpisodeLimit))
        response = await query(self, chat, prompt, original_instruction, session)
        if isinstance(response, str):
            # Harbor 0.24 returns a bare string on successful XML salvage, before Chat
            # records this turn. Echo it so the next request extends the sampled tokens.
            chat.messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
            chat.reset_response_chain()
            return LLMResponse(content=response)
        return response

    async def _handle_llm_interaction(self, *args, **kwargs):
        interaction = await super()._handle_llm_interaction(*args, **kwargs)
        commands, is_complete, feedback, *_ = interaction
        self._confirmed_complete = self._pending_completion and is_complete and "ERROR:" not in (feedback or "")
        self.trace.n_commands += len(commands)
        self.trace.n_format_errors += int(not commands and not is_complete)
        return interaction

    async def run(self, instruction, environment, context):
        self._confirmed_complete = False
        try:
            await super().run(instruction, environment, context)
            if self._confirmed_complete:
                self.trace.exit_status = "Submitted"
            else:
                self.trace.exit_status = "TurnLimit" if self._n_episodes >= self._max_episodes else "SessionEnded"
            self.trace.truncated = not self._confirmed_complete
        except ContextLengthExceededError:
            self.trace.exit_status = "ContextLimit"
            self.trace.truncated = True
        except EpisodeLimit as exc:
            self.trace.exit_status = str(exc)
            self.trace.truncated = True
        finally:
            context.n_input_tokens = sum(call["prompt_tokens"] for call in self.trace.calls)
            context.n_output_tokens = self.trace.generated_tokens
            self.trace.write_trajectory(self._llm.session.messages, self.logs_dir)
