"""A 120k generated-token episode survives compaction and training conversion."""

import pytest
from agentic_critic.terminal_env import Completion, TerminalAgent, TerminalConfig
from tests.agentic_critic.test_terminal_env import Tokenizer, completion
from tests.fast.ray.rollout.test_memory_critic_minibatches import _executor, _parallel

from miles.utils.data import process_rollout_data
from miles.utils.types import Sample


@pytest.mark.parametrize("budget", [131072, 262144])
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_cumulative_budget_and_training_pack_include_different_tokens(monkeypatch, mode, budget):
    config = TerminalConfig(
        max_steps=300,
        max_context_tokens=32768,
        compact_threshold=24576,
        max_new_tokens=8192,
        summary_tokens=2048,
        max_total_tokens=120000,
    )

    async def decode(prompt, limit):
        is_summary = "Summarize progress for a fresh context" in Tokenizer().decode(prompt[-1000:])
        suffix = "progress" if is_summary else "\n```bash\ninspect\n```"
        # Responses may finish below their per-call ceiling. Leave room for
        # observations and a complete summary inside the 32k actor window.
        generated = min(limit, 1024 if is_summary else 3072)
        padding = generated - len(completion(suffix).tokens)
        if padding < 0:
            return Completion([ord("x")] * limit, [-0.5] * limit, "x" * limit, "length")
        return completion("x" * padding + suffix)

    async def execute(_command, _timeout):
        return {"output": "observation" * 200, "returncode": 0}

    trace = await TerminalAgent(tokenizer=Tokenizer(), decode=decode, execute=execute, config=config).run_async("task")
    assert trace.generated_tokens == 120000 and trace.truncated and trace.n_compactions > 1
    assert {call["kind"] for call in trace.calls} == {"action", "summary"}
    samples = []
    for segment, fields in enumerate(trace.sample_fields()):
        sample = Sample(
            index=0,
            rollout_id=0,
            group_index=0,
            tokens=fields["tokens"],
            response_length=fields["response_length"],
            loss_mask=fields["loss_mask"],
            reward=1.0,
            status=Sample.Status.TRUNCATED,
            metadata={"segment": segment},
        )
        for key, value in fields.items():
            if key.startswith("critic_"):
                setattr(sample, key, value)
        samples.append(sample)
    assert max(len(sample.tokens) for sample in samples) <= 32768
    assert 131072 < sum(len(sample.tokens) for sample in samples) <= 262144
    executor = _executor(monkeypatch, mode=mode, dp_size=1, balance=False, samples=samples, num_groups=1, fanout=1, critic_steps=1)
    executor.args.num_critic_only_steps = 30
    executor.args.max_tokens_per_gpu = budget
    if mode == "full_detach" and budget == 131072:
        # The chronological critic sequence is indivisible.
        with pytest.raises(ValueError, match="exceeds --max-tokens-per-gpu"):
            await executor.get(0)
        return
    pack = await executor.get(0)
    actor, actor_result = process_rollout_data(executor.args, pack.data_ref, 0, _parallel(), witness_info=None)
    critic, critic_result = process_rollout_data(executor.args, pack.for_critic().data_ref, 0, _parallel(), witness_info=None)
    with actor_result, critic_result:
        for shard in (actor, critic):
            bins = shard["micro_batch_indices"]
            assert (len(bins) > 1) == (budget == 131072)
            assert all(sum(shard["total_lengths"][i] for i in rows) <= budget for rows in bins)
        assert sum(sum(mask) for mask in actor["loss_masks"]) == 120000
        assert sum(sum(mask) for mask in critic["loss_masks"]) == 120000
        responses = [tokens[-length:] for tokens, length in zip(critic["tokens"], critic["response_lengths"], strict=True)]
        shipped = [responses[row][start : start + length] for row, start, length in critic["value_slices"]]
        assert shipped == [tokens[-length:] for tokens, length in zip(actor["tokens"], actor["response_lengths"], strict=True)]


async def test_terminus_summary_keeps_exact_critic_memory_at_12k(tmp_path):
    from agentic_critic.harbor_terminus import MilesTerminus2

    action_text = "<response><commands></commands></response>"
    summary_text = ("private reasoning</think><response><analysis>ETL progress</analysis>"
                    "<plan>Verify the files</plan><commands></commands></response><|im_end|>")
    halfway = sum(len(completion(text).tokens) for text in (action_text, summary_text))
    config = TerminalConfig(max_context_tokens=63488, compact_threshold=12000, max_new_tokens=16384, max_total_tokens=2 * halfway)
    agent = MilesTerminus2(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-9B", parser_name="xml", enable_summarize=False, record_terminal_session=False)
    tokenizer = Tokenizer()

    prompts = []

    async def decode(prompt, limit):
        prompts.append(tokenizer.decode(prompt))
        summary = "Summarize progress for a fresh context" in tokenizer.decode(prompt[-1000:])
        return completion(summary_text if summary else action_text)

    agent.bind(tokenizer=tokenizer, decode=decode, config=config)
    from harbor.llms.chat import Chat

    chat = Chat(agent._llm)
    await chat.chat("Read the configuration files")
    before = agent.trace.generated_tokens
    await chat.chat("o" * 12000)
    assert [call["kind"] for call in agent.trace.calls] == ["action", "summary", "action"]
    assert agent.trace.n_compactions == 1 and agent.trace.critic_memory_exact
    assert agent.trace.generated_tokens == before + sum(c["output_tokens"] for c in agent.trace.calls[1:])
    actor_spans = [s["rollout_log_probs"] for s in agent.trace.sample_fields()]
    assert all(actor_spans)
    assert sum(map(sum, [s.loss_mask for s in agent.trace.segments])) == agent.trace.generated_tokens


    assert all("[Episode budget:" not in prompt for prompt in prompts[:2])
    assert f"[Episode budget: {halfway:,} of {2 * halfway:,}" in prompts[2]
    assert "ETL progress" in prompts[2] and "Verify the files" in prompts[2]
    assert "private reasoning" not in prompts[2] and "<|im_end|>" not in prompts[2]
    assert "<commands>" not in prompts[2]
    assert "[Episode budget:" in tokenizer.decode(agent.trace.critic_segments[-1].tokens)
    await chat.chat("next observation")
    assert prompts[3].count("[Episode budget:") == 1

    agent._llm.session.messages.append({"role": "user", "content": "keep this observation" * 800})
    async def empty_summary(prompt, limit):
        return completion("internal note</think><|im_end|>")
    agent._llm.session.decode = empty_summary
    await agent._llm.session._compact()
    assert agent.trace.n_compactions == 1
    assert any("keep this observation" in msg["content"] for msg in agent._llm.session.messages)


@pytest.mark.parametrize("salvage", [True, False])
async def test_terminus_capped_action_preserves_sampled_prefix(tmp_path, salvage):
    from harbor.llms.chat import Chat
    from agentic_critic.harbor_terminus import MilesTerminus2

    class ChatTokenizer(Tokenizer):
        eos_token_id = 3

        def apply_chat_template(self, messages, *, add_generation_prompt=True, **kwargs):
            text = "".join(f"<{m['role']}>{m['content']}\x03" for m in messages)
            return list(map(ord, text + ("<assistant>" if add_generation_prompt else "")))

    xml = "<response><analysis>ok</analysis><plan>check</plan><commands></commands></response>"
    cap = xml + "extra" if salvage else "<response>"
    requests = []
    async def decode(prompt, limit):
        requests.append(list(prompt))
        if len(requests) == 1:
            return Completion(list(map(ord, cap)), [-0.5] * len(cap), cap, "length")
        raw = xml
        return Completion(list(map(ord, raw)) + [3], [-0.5] * (len(raw) + 1), raw, "stop")

    agent = MilesTerminus2(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-9B", parser_name="xml", enable_summarize=False)
    agent.bind(tokenizer=ChatTokenizer(), decode=decode, config=TerminalConfig(max_new_tokens=512))
    chat = Chat(agent._llm)
    initial = agent._prompt_template.format(instruction="Read report.txt", terminal_state="Current Terminal Screen:\n$")
    response = await agent._query_llm(chat, initial)
    assert response.content == xml
    if salvage:
        await agent._query_llm(chat, "next observation")
    assert len(requests) == 2 and chat.messages[0]["content"] == initial
    assert requests[1][:len(requests[0]) + len(cap)] == requests[0] + list(map(ord, cap))
    assert agent.trace.critic_memory_exact and len(agent.trace.segments) == 1
    assert ("Re-issue this request" in ChatTokenizer().decode(requests[1])) == (not salvage)


async def test_terminus_retries_transport_only(tmp_path):
    from harbor.llms.base import ContextLengthExceededError
    from harbor.llms.chat import Chat
    from agentic_critic.harbor_terminus import MilesTerminus2
    from agentic_critic.terminal_env import EpisodeLimit

    agent = MilesTerminus2(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-9B", parser_name="xml", enable_summarize=False)
    calls = []

    async def decode(prompt, limit):
        calls.append(prompt.copy())
        if len(calls) == 1:
            raise ConnectionError("temporary outage")
        return Completion([], [], "", "length")

    agent.bind(tokenizer=Tokenizer(), decode=decode, config=TerminalConfig())
    chat = Chat(agent._llm)
    with pytest.raises(ContextLengthExceededError):
        await agent._query_llm(chat, "task")
    assert len(calls) == 2 and calls[0] == calls[1] and not chat.messages
    assert not agent.trace.segments
    agent.trace.generated_tokens = agent._llm.session.options.max_total_tokens
    with pytest.raises(EpisodeLimit, match="EpisodeTokenLimit"):
        await agent._query_llm(chat, "task")
    assert len(calls) == 2

def test_terminus_visible_completion_uses_sampled_qwen_markers(tmp_path):
    from agentic_critic.harbor_terminus import MilesTerminus2

    class MarkerTokenizer(Tokenizer):
        def get_vocab(self):
            return {"<think>": 1, "</think>": 2, "<|im_end|>": 3, "<tool_call>": 4}

    agent = MilesTerminus2(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-9B", parser_name="xml")
    agent.bind(tokenizer=MarkerTokenizer(), decode=None, config=TerminalConfig())
    parse = agent._llm._response_parts
    xml = list(map(ord, "  <response/>  "))
    assert parse(xml) == ("<response/>", None)
    assert parse([1, *xml]) == ("", "<response/>")
    assert parse([1, 2, *xml, 4, 120]) == ("<response/>", "")
    assert parse([120, 2, *xml, 3, 121]) == ("<response/>", "x")


async def test_terminus_observations_preserve_tmux_output(tmp_path):
    import asyncio
    import shlex
    import shutil
    import subprocess
    from types import SimpleNamespace

    from agentic_critic.harbor_terminus import MilesTerminus2

    if shutil.which("tmux") is None:
        pytest.skip("tmux is required to exercise terminal observations")
    socket = str(tmp_path / "tmux")
    target = "observation"

    def tmux(*args):
        return subprocess.check_output(["tmux", "-S", socket, *args], text=True)

    class LocalEnvironment:
        async def exec(self, command, user):
            prefix = 'tmux() { command tmux -S ' + shlex.quote(socket) + ' "$@"; }\n'
            result = subprocess.run(["sh", "-c", prefix + command], capture_output=True, text=True)
            return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    tmux("new-session", "-d", "-s", target, "cat")
    try:
        agent = MilesTerminus2(logs_dir=tmp_path, model_name="Qwen/Qwen3.5-9B", parser_name="xml")
        agent.bind(tokenizer=Tokenizer(), decode=None, config=TerminalConfig())
        agent._session = SimpleNamespace(_session_name=target, _user=None, environment=LocalEnvironment())
        agent._previous_pane = None
        first = await agent._observe_terminal()
        assert first == "Current Terminal Screen:\n" + tmux("capture-pane", "-p", "-t", target)
        cursor = agent._previous_pane[1]
        tmux("send-keys", "-t", target, "example", "Enter")
        for _ in range(100):
            if tmux("capture-pane", "-p", "-t", target).count("example") == 2:
                break
            await asyncio.sleep(0.01)
        delta = await agent._observe_terminal()
        assert delta == "New Terminal Output:\n" + tmux("capture-pane", "-p", "-t", target, "-S", str(cursor))
        same = await agent._observe_terminal()
        assert same == "Current Terminal Screen:\n" + tmux("capture-pane", "-p", "-t", target)
    finally:
        tmux("kill-server")
