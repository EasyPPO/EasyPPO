"""Thinking-enabled mini protocol, history retention and sampled-token contracts."""

import os
from dataclasses import replace
from pathlib import Path

import agentic_critic
import pytest
from agentic_critic.terminal_env import _FORMAT_ERROR, _SYSTEM, Completion, TerminalAgent, TerminalConfig

TEMPLATE = Path(agentic_critic.__file__).with_name("qwen3_5_interleaved.jinja")


@pytest.fixture
def tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if path := os.environ.get("TEST_QWEN_TOKENIZER"):
        result = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    else:
        # Byte-level encoding exercises the real template without downloading weights/tokenizers.
        vocab = {s: i for i, s in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
        vocab["ab"] = len(vocab)
        backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, [("a", "b")]))
        backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = tokenizers.decoders.ByteLevel()
        result = transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"]
        )
    result.chat_template = TEMPLATE.read_text()
    return result


def test_prompt_matches_pinned_mini_config():
    import minisweagent
    import yaml

    mini = yaml.safe_load((Path(minisweagent.__file__).parent / "config/mini.yaml").read_text())["agent"]
    assert _SYSTEM.startswith(mini["system_template"].rstrip())
    assert _FORMAT_ERROR == mini["format_error_template"]


@pytest.mark.parametrize(
    "feedback",
    [
        "<returncode>0</returncode><output>/app</output>",
        "Please always provide EXACTLY ONE action.",
        "Summarize progress.",
    ],
)
def test_reasoning_survives_observation_error_and_summary_request(tokenizer, feedback):
    agent = TerminalAgent(tokenizer=tokenizer, decode=None, execute=None, config=TerminalConfig())
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "Inspect the task"},
        {"role": "assistant", "content": "REASONING_SENTINEL\n</think>\n\n```bash\npwd\n```"},
        {"role": "user", "content": feedback},
    ]
    rendered = tokenizer.decode(agent._encode(messages))
    assert "REASONING_SENTINEL" in rendered and "```bash\npwd\n```" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    assert "<tool_call>" not in rendered and "<tools>" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_thinking_enabled_episode_preserves_exact_tokens_and_compaction(tokenizer, compact):
    texts = ["Inspect first.\n</think>\n\n```bash\ninspect\n```"]
    if compact:
        texts.append("Remember the result.\n</think>\n\nProgress: inspected; next submit.")
    texts.append("Ready to finish.\n</think>\n\n```bash\nsubmit\n```")
    replies = iter(texts)
    prompts, sampled, commands = [], [], []

    async def decode(prompt, limit):
        text = next(replies)
        tokens = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
        assert len(tokens) <= limit
        prompts.append(list(prompt))
        sampled.append(tokens)
        return Completion(tokens, [-0.5] * len(tokens), text)

    async def execute(command, timeout):
        commands.append(command)
        return {
            "returncode": 0,
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "z " * 4000,
        }

    config = TerminalConfig(compaction="summary" if compact else "none")
    agent = TerminalAgent(tokenizer=tokenizer, decode=decode, execute=execute, config=config)
    initial_size = len(agent._encode([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "task"}]))
    agent.options = replace(config, compact_threshold=initial_size + 500)
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and trace.n_format_errors == 0
    assert commands == ["inspect", "submit"]
    assert trace.n_compactions == int(compact)
    assert "Inspect first." in tokenizer.decode(prompts[1])
    assert all(tokenizer.decode(p).endswith("<|im_start|>assistant\n<think>\n") for p in prompts)
    if compact:
        resumed = tokenizer.decode(prompts[-1])
        assert "Progress: inspected" in resumed and "z " * 100 not in resumed
    fields = trace.sample_fields()
    assert sum(sum(f["loss_mask"]) for f in fields) == sum(map(len, sampled))
    # Every request/response pair must survive byte-for-byte in one actor segment.
    for prompt, tokens in zip(prompts, sampled, strict=True):
        assert any(f["tokens"][: len(prompt) + len(tokens)] == prompt + tokens for f in fields)
    for f in fields:
        response = f["tokens"][-f["response_length"] :]
        assert len(response) == len(f["loss_mask"]) == len(f["rollout_log_probs"])
        assert all(
            lp == (-0.5 if mask else 0.0) for lp, mask in zip(f["rollout_log_probs"], f["loss_mask"], strict=True)
        )
        start, length = f["critic_span"]
        critic = f["critic_tokens"][-f["critic_response_length"] :]
        assert critic[start : start + length] == response


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_token_in_reuses_noncanonical_sampled_ids_before_next_decode(tokenizer, compact):
    tokenizer.eos_token = "<|im_end|>"
    first = "ab inspect.\n</think>\n\n```bash\ninspect\n```"
    middle = (
        "ab retained progress.\n</think>\n\nSUMMARY_SENTINEL"
        if compact
        else "ab check.\n</think>\n\n```bash\ncheck\n```"
    )
    last = "ab done.\n</think>\n\n```bash\nsubmit\n```"
    replies = iter([first, middle, last])
    prompts, sampled = [], []

    async def decode(prompt, limit):
        text = next(replies)
        # Intentionally choose a valid, noncanonical BPE split. Retokenizing
        # the text would merge "ab" and discard these exact sampled IDs.
        tokens = [token for char in text for token in tokenizer.encode(char, add_special_tokens=False)]
        assert tokens != tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(tokens) == text
        tokens.append(tokenizer.eos_token_id)
        assert len(tokens) <= limit
        prompts.append(list(prompt))
        sampled.append(tokens)
        return Completion(tokens, [-0.5] * len(tokens), text)

    async def execute(command, timeout):
        return {
            "returncode": 0,
            "output": (
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                if command == "submit"
                else ("observation " * 4000 if compact else f"OBSERVATION_{command}_SENTINEL")
            ),
        }

    config = TerminalConfig(compaction="summary" if compact else "none")
    agent = TerminalAgent(tokenizer=tokenizer, decode=decode, execute=execute, config=config)
    task = "TOKEN_IN_TASK_SENTINEL"
    initial = agent._encode([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": task}])
    agent.options = replace(config, compact_threshold=len(initial) + 500)
    trace = await agent.run_async(task)
    assert trace.exit_status == "Submitted" and trace.critic_memory_exact
    assert len(trace.segments) == 1 + int(compact)
    exact_first = prompts[0] + sampled[0]
    assert prompts[1][: len(exact_first)] == exact_first  # What the decoder actually received.
    assert tokenizer.decode(prompts[1][len(exact_first) :]).startswith("\n<|im_start|>user\n")
    if compact:
        assert trace.n_compactions == 1
        assert prompts[2][: len(prompts[1] + sampled[1])] != prompts[1] + sampled[1]
        assert "SUMMARY_SENTINEL" in tokenizer.decode(prompts[2])
    else:
        exact_second = prompts[1] + sampled[1]
        assert prompts[2][: len(exact_second)] == exact_second
        text = tokenizer.decode(trace.segments[0].tokens)
        assert text.count(task) == text.count("OBSERVATION_inspect_SENTINEL") == 1
    critic_text = tokenizer.decode([token for segment in trace.critic_segments for token in segment.tokens])
    assert critic_text.count(task) == 1
    assert sum(sum(segment.loss_mask) for segment in trace.segments) == sum(map(len, sampled))


@pytest.mark.asyncio
async def test_token_in_requires_a_proven_eos_boundary(tokenizer):
    tokenizer.eos_token = "<|im_end|>"
    first_text = "ab first.\n</think>\n\n```bash\ninspect\n```"
    messages = [{"role": "user", "content": "task"}]
    prompts = []

    async def decode(prompt, limit):
        prompts.append(list(prompt))
        # A stop reason alone is insufficient: no EOS ID completed this output.
        tokens = [token for char in first_text for token in tokenizer.encode(char, add_special_tokens=False)]
        return Completion(tokens, [-0.5] * len(tokens), first_text)

    agent = TerminalAgent(tokenizer=tokenizer, decode=decode, execute=None, config=TerminalConfig())
    await agent._complete(messages, limit=1000, kind="action")
    assert agent._completed_prefix is None
    next_messages = messages + [{"role": "assistant", "content": first_text}, {"role": "user", "content": "obs"}]
    assert agent._prompt(next_messages) == agent._encode(next_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["action", "invalid_action", "summary", "unclosed_summary"])
@pytest.mark.parametrize("topk", [False, True])
async def test_length_limited_turns_continue_with_masked_eos_and_exact_tokens(tokenizer, kind, topk):
    import numpy as np

    compact = "summary" in kind
    first = "ab inspect.\n</think>\n\n```bash\ninspect\n```"
    if kind == "invalid_action":
        first = "ab unfinished reasoning"
    summary = "ab summary reasoning\n</think>\n\nSUMMARY_SENTINEL"
    if kind == "unclosed_summary":
        summary = "ab unfinished SUMMARY_SENTINEL"
    replies = [(first, "stop" if compact else "length")]
    if compact:
        replies.append((summary, "length"))
    replies.append(("ab done.\n</think>\n\n```bash\nsubmit\n```", "stop"))
    pending = iter(replies)
    prompts, sampled, commands = [], [], []

    async def decode(prompt, limit):
        text, reason = next(pending)
        # Preserve a noncanonical BPE split even when no sampled EOS closes the turn.
        tokens = [token for char in text for token in tokenizer.encode(char, add_special_tokens=False)]
        if reason == "stop":
            tokens.append(tokenizer.eos_token_id)
        assert len(tokens) <= limit
        prompts.append(list(prompt))
        sampled.append(tokens)
        ids = np.asarray(tokens, dtype=np.int32)[:, None] if topk else None
        logps = np.full((len(tokens), 1), -0.5, dtype=np.float32) if topk else None
        return Completion(tokens, [-0.5] * len(tokens), text, reason, ids, logps)

    async def execute(command, timeout):
        commands.append(command)
        return {
            "returncode": 0,
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "observation " * 500,
        }

    config = TerminalConfig(compaction="summary" if compact else "none", continue_on_length=True)
    agent = TerminalAgent(tokenizer=tokenizer, decode=decode, execute=execute, config=config)
    initial_size = len(agent._encode([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": "task"}]))
    agent.options = replace(config, compact_threshold=initial_size + 500)
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and not trace.truncated
    assert trace.truncated_turns == 1 and trace.critic_memory_exact
    assert trace.n_compactions == int(compact)
    assert commands == (["submit"] if kind == "invalid_action" else ["inspect", "submit"])
    assert trace.n_format_errors == int(kind == "invalid_action")
    assert trace.generated_tokens == sum(map(len, sampled)) == sum(sum(s.loss_mask) for s in trace.segments)
    for prompt, tokens in zip(prompts, sampled, strict=True):
        assert any(s.tokens[: len(prompt) + len(tokens)] == prompt + tokens for s in trace.segments)
    # The inserted end marker closes the truncated turn without a value/policy target.
    call_index = 1 if compact else 0
    segment = trace.segments[0]
    closing_position = len(prompts[call_index]) + len(sampled[call_index])
    response_position = closing_position - segment.response_start
    assert segment.tokens[closing_position] == tokenizer.eos_token_id
    assert segment.loss_mask[response_position] == trace.logprobs[0][response_position] == 0
    if compact:
        assert "SUMMARY_SENTINEL" in tokenizer.decode(prompts[-1])
    else:
        assert prompts[1][: closing_position + 1] == prompts[0] + sampled[0] + [tokenizer.eos_token_id]
    for i, (actor, critic) in enumerate(zip(trace.segments, trace.critic_segments, strict=True)):
        assert actor.tokens[-actor.response_length :] == critic.tokens[-critic.response_length :]
        assert actor.loss_mask == critic.loss_mask
        if topk:
            active = np.asarray(actor.loss_mask, dtype=bool)
            ids, logps = trace.topk_token_ids[i], trace.topk_log_probs[i]
            assert ids.shape == logps.shape == (actor.response_length, 1)
            assert (ids[~active] == -1).all() and np.isneginf(logps[~active]).all()
            np.testing.assert_array_equal(ids[active, 0], np.asarray(actor.tokens[-actor.response_length :])[active])


@pytest.mark.asyncio
async def test_continuing_truncated_turn_still_obeys_episode_budget(tokenizer):
    text = "Done inspecting.\n</think>\n\n```bash\ninspect\n```"
    tokens = tokenizer.encode(text, add_special_tokens=False)
    requests, commands = [], []

    async def decode(prompt, limit):
        requests.append(limit)
        return Completion(tokens, [-0.5] * len(tokens), text, "length")

    async def execute(command, timeout):
        commands.append(command)
        return {"returncode": 0, "output": "ok"}

    agent = TerminalAgent(
        tokenizer=tokenizer,
        decode=decode,
        execute=execute,
        config=TerminalConfig(compaction="none", continue_on_length=True, max_total_tokens=len(tokens)),
    )
    trace = await agent.run_async("task")
    assert trace.exit_status == "ContextOrTokenLimit" and trace.truncated
    assert trace.truncated_turns == 1 and trace.generated_tokens == len(tokens)
    assert requests == [len(tokens)] and commands == ["inspect"]
