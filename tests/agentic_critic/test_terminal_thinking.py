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
