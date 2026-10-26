"""CPU contracts for mini's real parser, compaction, budgets and token alignment."""

import asyncio

import pytest

pytest.importorskip("minisweagent")

from agentic_critic.terminal_env import Completion, TerminalAgent, TerminalConfig, TerminalTrace


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def completion(text, finish_reason="stop"):
    tokens = list(map(ord, text + "</assistant>"))
    return Completion(tokens, [-0.5] * len(tokens), text, finish_reason)


def action(command):
    return completion(f"```bash\n{command}\n```")


def make_agent(replies, *, config=None, output="ok"):
    replies = iter(replies)
    prompts, commands = [], []

    async def decode(prompt, limit):
        prompts.append(list(prompt))
        result = next(replies)
        assert len(result.tokens) <= limit
        return result

    async def execute(command, timeout):
        commands.append(command)
        return {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" if command == "submit" else output,
            "returncode": 0,
        }

    config = config or TerminalConfig(
        max_context_tokens=6000, compact_threshold=1200, max_new_tokens=1000, summary_tokens=200
    )
    agent = TerminalAgent(tokenizer=Tokenizer(), decode=decode, execute=execute, config=config)
    return agent, prompts, commands


def test_real_hf_chat_template_returns_token_ids():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    agent = TerminalAgent(tokenizer=tokenizer, decode=None, execute=None, config=TerminalConfig())
    assert agent._encode([{"role": "user", "content": "hello"}]) == [1]


@pytest.mark.parametrize("enabled", [True, False])
def test_thinking_mode_reaches_chat_template(enabled):
    class RecordingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is enabled
            return super().apply_chat_template(messages, **kwargs)

    agent = TerminalAgent(
        tokenizer=RecordingTokenizer(), decode=None, execute=None, config=TerminalConfig(enable_thinking=enabled)
    )
    assert agent._encode([{"role": "user", "content": "task"}])


@pytest.mark.asyncio
async def test_summary_keeps_all_generated_tokens_and_masks_observations():
    agent, prompts, commands = make_agent(
        [action("inspect"), completion("File /tmp/work changed; next submit."), action("submit")], output="x" * 1800
    )
    trace = await agent.run_async("Fix the task")
    assert trace.exit_status == "Submitted" and trace.n_compactions == 1
    assert commands == ["inspect", "submit"]  # summary was never executed
    assert [c["kind"] for c in trace.calls] == ["action", "summary", "action"]
    fields = trace.sample_fields()
    assert len(fields) == 2
    assert sum(sum(f["loss_mask"]) for f in fields) == trace.generated_tokens
    assert 0 in fields[0]["loss_mask"]
    assert "x" * 1800 not in Tokenizer().decode(fields[1]["tokens"])
    for prompt in prompts:
        assert any(f["tokens"][: len(prompt)] == prompt for f in fields)
    critic_response = fields[0]["critic_tokens"][-fields[0]["critic_response_length"] :]
    for f in fields:
        start, length = f["critic_span"]
        assert critic_response[start : start + length] == f["tokens"][-length:]
        assert len(f["loss_mask"]) == len(f["rollout_log_probs"]) == length
        assert all(
            lp == (0.0 if mask == 0 else -0.5) for lp, mask in zip(f["rollout_log_probs"], f["loss_mask"], strict=True)
        )


@pytest.mark.asyncio
async def test_format_recovery_and_reasoning_are_not_executed():
    agent, _, commands = make_agent(
        [
            completion("not an action"),
            completion("<think>```bash\nwrong\n```</think>```bash\nsubmit\n```"),
        ],
        config=TerminalConfig(compaction="none"),
    )
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and commands == ["submit"]
    assert "EXACTLY ONE" in agent.messages[3]["content"]
    assert "```bash" in agent.messages[3]["content"]
    assert "<response_example>" in agent.messages[3]["content"]
    assert trace.n_format_errors == 1 and trace.n_commands == 1
    assert [c["action_status"] for c in trace.calls] == ["format_error", "executed"]
    assert trace.generated_tokens == sum(sum(s.loss_mask) for s in trace.segments)


@pytest.mark.asyncio
async def test_keep_last_turns_drops_only_whole_turns():
    config = TerminalConfig(
        compaction="keep_last_turns",
        keep_last_turns=1,
        compact_threshold=2200,
        max_context_tokens=6000,
        max_new_tokens=500,
        summary_tokens=200,
    )
    agent, _, _ = make_agent([action("first"), action("second"), action("submit")], config=config, output="x" * 650)
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and trace.n_compactions == 1
    assert len(trace.calls) == 3 and all(c["kind"] == "action" for c in trace.calls)
    resumed = Tokenizer().decode(trace.segments[-1].tokens)
    assert "first" not in resumed and "second" in resumed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["action", "summary"])
async def test_length_stop_never_executes_partial_action_or_resumes_partial_summary(kind):
    replies = (
        [completion("partial", "length")] if kind == "action" else [action("inspect"), completion("partial", "length")]
    )
    agent, _, commands = make_agent(replies, output="x" * 1800)
    trace = await agent.run_async("task")
    assert trace.exit_status == "GenerationLimit" and trace.truncated
    assert commands == ([] if kind == "action" else ["inspect"])
    assert trace.n_compactions == 0 and trace.generated_tokens > 0


@pytest.mark.asyncio
async def test_step_limit_and_command_timeout():
    config = TerminalConfig(max_steps=1)
    agent, _, _ = make_agent([action("slow")], config=config)

    async def timeout(command, seconds):
        raise TimeoutError

    agent.execute = timeout
    trace = await agent.run_async("task")
    assert trace.truncated and trace.exit_status == "StepLimit"
    assert "timed out" in agent.messages[-1]["content"]


@pytest.mark.asyncio
async def test_cancellation_propagates_without_more_commands():
    agent, _, _ = make_agent([])
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def decode(prompt, limit):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    agent.decode = decode
    task = asyncio.create_task(agent.run_async("task"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and not agent.trace.calls


def test_template_rewrite_starts_new_segment_without_retokenizing():
    trace = TerminalTrace()
    trace.record([1, 2], Completion([3, 4], [-0.1, -0.2], "a"), kind="action")
    trace.record([1, 7], Completion([8], [-0.3], "b"), kind="action")
    fields = trace.sample_fields()
    assert [f["tokens"] for f in fields] == [[1, 2, 3, 4], [1, 7, 8]]
    assert fields[1]["critic_tokens"] == [1, 2, 3, 4, 1, 7, 8]


@pytest.mark.parametrize(
    "kwargs", [{"max_steps": 0}, {"keep_last_turns": -1}, {"compaction": "bad"}, {"compact_threshold": 32768}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TerminalConfig(**kwargs)


def test_request_overrides_do_not_mutate_defaults():
    defaults = TerminalConfig()
    assert defaults.with_overrides({"max_steps": 4}).max_steps == 4
    assert defaults.max_steps == 100
    with pytest.raises(TypeError):
        defaults.with_overrides({"typo": 4})


@pytest.mark.asyncio
async def test_keep_more_turns_than_exist_stops_cleanly_when_it_cannot_shrink():
    config = TerminalConfig(
        compaction="keep_last_turns",
        keep_last_turns=4,
        compact_threshold=800,
        max_context_tokens=6000,
        max_new_tokens=500,
        summary_tokens=200,
    )
    agent, _, _ = make_agent([action("inspect")], config=config, output="x" * 900)
    trace = await agent.run_async("task")
    assert trace.exit_status == "CompactionDidNotShrink" and trace.truncated


@pytest.mark.asyncio
async def test_total_token_limit_includes_summary():
    replies = [action("inspect"), completion("progress")]
    budget = sum(len(r.tokens) for r in replies)
    config = TerminalConfig(
        max_context_tokens=6000,
        compact_threshold=1200,
        max_new_tokens=1000,
        summary_tokens=200,
        max_total_tokens=budget,
    )
    agent, prompts, commands = make_agent(replies, config=config, output="x" * 1800)
    trace = await agent.run_async("task")
    assert trace.exit_status == "ContextOrTokenLimit" and trace.truncated
    assert trace.generated_tokens == budget and len(prompts) == 2 and commands == ["inspect"]


@pytest.mark.asyncio
async def test_compaction_disabled_and_oversized_initial_context():
    config = TerminalConfig(
        compaction="none", max_context_tokens=6000, compact_threshold=1200, max_new_tokens=1000, summary_tokens=200
    )
    agent, prompts, _ = make_agent([], config=config)
    trace = await agent.run_async("x" * 7000)
    assert trace.truncated and not prompts and not trace.sample_fields()


@pytest.mark.asyncio
async def test_tiny_output_cap_does_not_keep_the_entire_output():
    agent, _, _ = make_agent(
        [action("inspect"), action("submit")],
        config=TerminalConfig(max_output_chars=1),
        output="sensitive-long-output",
    )
    await agent.run_async("task")
    assert "sensitive-long-output" not in agent.messages[3]["content"]


def test_bad_logprobs_fail_before_mutating_trace():
    trace = TerminalTrace()
    with pytest.raises(ValueError, match="finite"):
        trace.record([1], Completion([2], [float("nan")], "bad"), kind="action")
    assert not trace.segments
