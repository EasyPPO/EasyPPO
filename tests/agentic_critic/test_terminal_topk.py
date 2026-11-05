"""Real terminal/Miles top-k collection through decoding and context compaction."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("minisweagent")
pytest.importorskip("harbor")

from agentic_critic import harbor_terminal, miles_terminal
from agentic_critic.terminal_env import Completion, TerminalAgent, TerminalConfig, TerminalTrace
from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils import http_utils
from miles.utils.types import Sample


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def _reply(text):
    tokens = list(map(ord, text + "</assistant>"))
    return Completion(tokens=tokens, logprobs=[-0.5] * len(tokens), text=text)


def _metadata(reply: Completion, *, topk: bool = True) -> dict:
    output = {
        "text": reply.text,
        "meta_info": {
            "output_token_logprobs": [
                (lp, token, None) for lp, token in zip(reply.logprobs, reply.tokens, strict=True)
            ],
            "finish_reason": {"type": reply.finish_reason},
        },
    }
    if topk:
        output["meta_info"]["output_top_logprobs"] = [
            [(lp, token, None), (-2.0, token + 1_000_000, None)]
            for lp, token in zip(reply.logprobs, reply.tokens, strict=True)
        ]
    return output


def _args(tmp_path: Path, *, candidate_count: int = 3):
    parser = argparse.ArgumentParser()
    miles_terminal.generate.add_arguments(parser)
    args = parser.parse_args([])
    for key, value in dict(
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
        sglang_router_policy="round_robin",
        rollout_max_response_len=4096,
        rollout_max_context_len=6000,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        rollout_top_logprobs_num=candidate_count,
        rollout_sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
        terminal_tasks_dir=str(tmp_path),
        terminal_max_context_tokens=6000,
        terminal_max_new_tokens=1000,
        terminal_compact_threshold=1200,
        terminal_summary_tokens=200,
    ).items():
        setattr(args, key, value)
    return args


def _task(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").touch()


def _assert_candidates(sample: Sample, k: int = 3):
    validate_rollout_topk_logprobs_sample(sample, k)
    active = np.asarray(sample.loss_mask, dtype=bool)
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    response = np.asarray(sample.tokens[-sample.response_length :])
    assert ids.dtype == np.int32 and logps.dtype == np.float32
    np.testing.assert_array_equal(ids[active, 0], response[active])
    np.testing.assert_allclose(logps[active, 0], np.asarray(sample.rollout_log_probs)[active])
    assert (ids[~active] == -1).all() and np.isneginf(logps[~active]).all()
    assert (ids[:, -1] == -1).all() and np.isneginf(logps[:, -1]).all()


@pytest.mark.asyncio
async def test_topk_survives_real_wrapper_summary_and_masked_glue(monkeypatch, tmp_path):
    _task(tmp_path)
    replies = [
        _reply("```bash\ninspect\n```"),
        _reply("Progress: inspected. Next submit."),
        _reply("```bash\nsubmit\n```"),
    ]
    pending, requests = iter(replies), []

    async def post(url, payload, **kwargs):
        requests.append(payload)
        return _metadata(next(pending))

    async def trial(**kwargs):
        async def execute(command, timeout):
            return {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "x" * 1800,
                "returncode": 0,
            }

        agent = TerminalAgent(
            tokenizer=kwargs["tokenizer"], decode=kwargs["decode"], execute=execute, config=kwargs["config"]
        )
        return await agent.run_async("task"), {"reward": 1.0}, "/trial"

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 512}
    result = await miles_terminal.generate(
        SimpleNamespace(
            args=_args(tmp_path),
            sample=Sample(index=1, metadata={"task_path": "task"}),
            evaluation=False,
            state=SimpleNamespace(tokenizer=_Tokenizer()),
            sampling_params=params,
        )
    )
    assert len(result.samples) == 2 and len(requests) == 3
    assert [request["sampling_params"]["max_new_tokens"] for request in requests] == [512, 200, 512]
    assert all(request["top_logprobs_num"] == 3 and request["return_logprob"] for request in requests)
    assert all(sample.metadata["n_compactions"] == 1 for sample in result.samples)
    assert sum(sum(sample.loss_mask) for sample in result.samples) == sum(len(reply.tokens) for reply in replies)
    for sample in result.samples:
        _assert_candidates(sample)
    assert 0 in result.samples[0].loss_mask
    assert params == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 512}


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation,candidate_count", [(True, 3), (False, 0)])
async def test_evaluation_and_feature_off_do_not_request_or_require_candidates(
    monkeypatch, tmp_path, evaluation, candidate_count
):
    _task(tmp_path)
    requests = []

    async def post(url, payload, **kwargs):
        requests.append(payload)
        output = _metadata(_reply("```bash\nsubmit\n```"), topk=False)
        # Irrelevant candidate output must not be consumed by feature-off/eval.
        output["meta_info"]["output_top_logprobs"] = "unused malformed side channel"
        return output

    async def trial(**kwargs):
        async def execute(command, timeout):
            return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "returncode": 0}

        agent = TerminalAgent(
            tokenizer=kwargs["tokenizer"], decode=kwargs["decode"], execute=execute, config=kwargs["config"]
        )
        return await agent.run_async("task"), {"reward": 1.0}, "/trial"

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    params = {
        "temperature": 1.0,
        "top_p": 0.95 if evaluation else 1.0,
        "top_k": 20 if evaluation else -1,
        "max_new_tokens": 512,
    }
    result = await miles_terminal.generate(
        SimpleNamespace(
            args=_args(tmp_path, candidate_count=candidate_count),
            sample=Sample(index=1, metadata={"task_path": "task"}),
            evaluation=evaluation,
            state=SimpleNamespace(tokenizer=_Tokenizer()),
            sampling_params=params,
        )
    )
    assert len(result.samples) == 1
    assert "top_logprobs_num" not in requests[0] and "sampling_logprobs_mode" not in requests[0]
    assert requests[0]["sampling_params"] == params
    assert result.samples[0].rollout_topk_token_ids is None and result.samples[0].rollout_topk_log_probs is None


@pytest.mark.parametrize(
    "sampling",
    [
        {"top_p": 0.95},
        {"top_k": 20},
        {"temperature": 0.7},
        {"min_p": 0.1},
        {"frequency_penalty": 0.2},
        {"repetition_penalty": 1.1},
    ],
)
def test_training_candidate_sampling_rejects_filters_and_logit_transforms(sampling):
    with pytest.raises(ValueError, match="unfiltered"):
        miles_terminal._validate_topk_sampling(sampling)


def test_training_rejects_support_mode_and_nonunit_temperature(tmp_path):
    args = _args(tmp_path)
    args.rollout_sampling_logprobs_mode = "support"
    with pytest.raises(ValueError, match="selected"):
        miles_terminal._validate_runtime(args)
    args.rollout_sampling_logprobs_mode = "selected"
    args.rollout_temperature = 0.7
    with pytest.raises(ValueError, match="temperature 1"):
        miles_terminal._validate_runtime(args)
    miles_terminal._validate_runtime(args, evaluation=True)


@pytest.mark.parametrize(
    "rows,match",
    [
        (None, "output_top_logprobs"),
        ([], "output_top_logprobs"),
        ([[]], "Every trained token"),
        ([[(-0.5, 7), (-1.5, 7)]], "Duplicate"),
        ([[(-0.5, 7.0)]], "invalid.*token ID"),
        ([[(-0.5, -2)]], "invalid.*token ID"),
        ([[(float("nan"), 7)]], "invalid.*log probability"),
        ([[(float("inf"), 7)]], "invalid.*log probability"),
        ([[(0.1, 7)]], "invalid.*log probability"),
        ([[(-0.7, 7)]], "same sampler"),
        ([[(-0.5, 7), (-2.0, 8), (-3.0, 9)]], "more top-k"),
        ([[(-0.5, True)]], "invalid.*token ID"),
    ],
)
def test_malformed_candidate_side_channels_fail(rows, match):
    output = {
        "text": "x",
        "meta_info": {
            "output_token_logprobs": [(-0.5, 7, None)],
            "output_top_logprobs": rows,
            "finish_reason": {"type": "stop"},
        },
    }
    with pytest.raises(ValueError, match=match):
        miles_terminal._completion_from_output(output, candidate_count=2)


def test_trace_refuses_missing_changed_or_invalid_candidate_arrays():
    valid = Completion(
        [7],
        [-0.5],
        "x",
        topk_token_ids=np.asarray([[7, -1]], dtype=np.int32),
        topk_log_probs=np.asarray([[-0.5, -np.inf]], dtype=np.float32),
    )
    trace = TerminalTrace()
    trace.record([1], valid, kind="action")
    for invalid in (
        replace(valid, topk_token_ids=None, topk_log_probs=None),
        replace(valid, topk_log_probs=None),
        replace(valid, topk_token_ids=np.asarray([[7, 8, -1]]), topk_log_probs=np.asarray([[-0.5, -2.0, -np.inf]])),
        replace(valid, topk_log_probs=np.asarray([[-0.5, 0.0]])),
        replace(valid, topk_token_ids=np.asarray([[7, 8]]), topk_log_probs=np.asarray([[-0.5, -0.5]])),
        replace(valid, topk_token_ids=np.asarray([[7.0, -1.0]])),
        replace(valid, topk_token_ids=np.asarray([[7, 8]]), topk_log_probs=np.asarray([[-0.5, -1e100]])),
    ):
        with pytest.raises(ValueError):
            trace.record([1, 7, 2], invalid, kind="summary")
    assert trace.segments[0].tokens == [1, 7]
    assert trace.generated_tokens == 1


def test_sampled_token_outside_topk_is_not_added_to_candidates():
    output = {
        "text": "x",
        "meta_info": {
            "output_token_logprobs": [(-3.0, 7, None)],
            "output_top_logprobs": [[(-0.5, 8), (-2.0, 9)]],
            "finish_reason": {"type": "stop"},
        },
    }
    reply = miles_terminal._completion_from_output(output, candidate_count=3)
    trace = TerminalTrace()
    trace.record([1], reply, kind="action")
    np.testing.assert_array_equal(trace.sample_fields()[0]["rollout_topk_token_ids"], [[8, 9, -1]])
    assert 7 not in trace.sample_fields()[0]["rollout_topk_token_ids"]


@pytest.mark.asyncio
async def test_keep_last_turn_compaction_does_not_duplicate_generated_candidates():
    replies = [_reply("```bash\nfirst\n```"), _reply("```bash\nsecond\n```"), _reply("```bash\nsubmit\n```")]
    pending = iter(replies)

    async def decode(prompt, limit):
        return miles_terminal._completion_from_output(_metadata(next(pending)), candidate_count=3)

    async def execute(command, timeout):
        return {
            "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "x" * 650,
            "returncode": 0,
        }

    config = TerminalConfig(
        compaction="keep_last_turns",
        keep_last_turns=1,
        compact_threshold=2200,
        max_context_tokens=6000,
        max_new_tokens=500,
        summary_tokens=200,
    )
    agent = TerminalAgent(tokenizer=_Tokenizer(), decode=decode, execute=execute, config=config)
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and trace.n_compactions == 1
    assert len(trace.sample_fields()) == 2
    assert sum(sum(fields["loss_mask"]) for fields in trace.sample_fields()) == sum(
        len(reply.tokens) for reply in replies
    )
    for fields in trace.sample_fields():
        sample = Sample(
            **{
                name: fields[name]
                for name in (
                    "tokens",
                    "response_length",
                    "loss_mask",
                    "rollout_log_probs",
                    "rollout_topk_token_ids",
                    "rollout_topk_log_probs",
                )
            }
        )
        _assert_candidates(sample)


@pytest.fixture
def _bpe_tokenizer():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {s: i for i, s in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    vocab["ab"] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, [("a", "b")]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"]
    )
    tokenizer.chat_template = (
        Path(__file__).resolve().parents[2].joinpath("agentic_critic/qwen3_5_interleaved.jinja").read_text()
    )
    return tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
async def test_noncanonical_sampled_ids_and_eos_preserve_candidate_alignment(_bpe_tokenizer, compact):
    tokenizer = _bpe_tokenizer
    texts = [
        "ab inspect.\n</think>\n\n```bash\ninspect\n```",
        "ab remembered.\n</think>\n\nSUMMARY" if compact else "ab check.\n</think>\n\n```bash\ncheck\n```",
        "ab done.\n</think>\n\n```bash\nsubmit\n```",
    ]
    pending, prompts, sampled = iter(texts), [], []

    async def decode(prompt, limit):
        text = next(pending)
        tokens = [token for char in text for token in tokenizer.encode(char, add_special_tokens=False)]
        assert tokens != tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        assert len(tokens) <= limit
        prompts.append(list(prompt))
        sampled.append(tokens)
        return miles_terminal._completion_from_output(
            _metadata(Completion(tokens, [-0.5] * len(tokens), text)), candidate_count=3
        )

    async def execute(command, timeout):
        return {
            "output": (
                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "z " * 4000 if compact else "ok"
            ),
            "returncode": 0,
        }

    config = TerminalConfig(compaction="summary" if compact else "none")
    agent = TerminalAgent(tokenizer=tokenizer, decode=decode, execute=execute, config=config)
    initial = len(agent._encode([{"role": "system", "content": "task"}, {"role": "user", "content": "task"}]))
    agent.options = replace(config, compact_threshold=initial + 1500)
    trace = await agent.run_async("task")
    assert trace.exit_status == "Submitted" and trace.n_compactions == int(compact)
    prefix = prompts[0] + sampled[0]
    assert prompts[1][: len(prefix)] == prefix
    for prompt, completion in zip(prompts, sampled, strict=True):
        assert any(
            fields["tokens"][: len(prompt) + len(completion)] == prompt + completion
            for fields in trace.sample_fields()
        )
    for fields in trace.sample_fields():
        sample = Sample(
            **{
                name: fields[name]
                for name in (
                    "tokens",
                    "response_length",
                    "loss_mask",
                    "rollout_log_probs",
                    "rollout_topk_token_ids",
                    "rollout_topk_log_probs",
                )
            }
        )
        _assert_candidates(sample)
    assert sum(sum(fields["loss_mask"]) for fields in trace.sample_fields()) == sum(map(len, sampled))


@pytest.mark.parametrize("enabled", [False, True])
def test_harbor_persists_candidates_losslessly_in_npz_only(tmp_path, enabled):
    reply = _reply("x")
    if enabled:
        reply = miles_terminal._completion_from_output(_metadata(reply), candidate_count=3)
    trace = TerminalTrace()
    trace.record([1], reply, kind="summary")
    harbor_terminal._write_trajectory(trace, [{"role": "assistant", "content": "x"}], tmp_path)
    raw = (tmp_path / "easyppo-trajectory.json").read_text()
    payload = json.loads(raw)["trace"]
    assert "topk_token_ids" not in payload and "topk_log_probs" not in payload
    if enabled:
        assert "Infinity" not in raw
        reference = payload["topk_candidates"]
        assert reference["path"] == "easyppo-topk.npz"
        with np.load(tmp_path / reference["path"], allow_pickle=False) as archive:
            segment = reference["segments"][0]
            np.testing.assert_array_equal(archive[segment["token_ids"]], trace.topk_token_ids[0])
            np.testing.assert_array_equal(archive[segment["log_probs"]], trace.topk_log_probs[0])
            assert segment["shape"] == list(trace.topk_token_ids[0].shape)
    else:
        expected = asdict(trace)
        expected.pop("topk_token_ids")
        expected.pop("topk_log_probs")
        assert payload == expected and not (tmp_path / "easyppo-topk.npz").exists()
