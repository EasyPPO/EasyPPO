"""CPU end-to-end wrapper: real Miles payload/Sample, scripted decode and sandbox."""

import argparse
from types import SimpleNamespace

import pytest

pytest.importorskip("minisweagent")
pytest.importorskip("harbor")
pytest.importorskip("miles")

from agentic_critic import harbor_terminal, miles_terminal
from agentic_critic.terminal_env import TerminalAgent

from miles.utils import http_utils
from miles.utils.types import Sample
from tests.agentic_critic.test_terminal_data import make_task
from tests.agentic_critic.test_terminal_env import Tokenizer, action, completion


def args_for(tmp_path):
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
        rollout_top_logprobs_num=0,
        terminal_tasks_dir=str(tmp_path),
        terminal_max_context_tokens=6000,
        terminal_max_new_tokens=1000,
        terminal_compact_threshold=1200,
        terminal_summary_tokens=200,
    ).items():
        setattr(args, key, value)
    return args


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [True, False])
async def test_wrapper_returns_trainable_compacted_episode(monkeypatch, tmp_path, evaluation):
    make_task(tmp_path)
    replies = iter([action("inspect"), completion("Progress: done. Next submit."), action("submit")])
    requests = []

    async def post(url, payload, **kwargs):
        requests.append(payload)
        reply = next(replies)
        return {
            "text": reply.text,
            "meta_info": {
                "output_token_logprobs": [
                    (lp, token, None) for token, lp in zip(reply.tokens, reply.logprobs, strict=True)
                ],
                "finish_reason": {"type": reply.finish_reason},
            },
        }

    async def trial(**kwargs):
        async def execute(command, timeout):
            return {
                "output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else "x" * 1800,
                "returncode": 0,
            }

        agent = TerminalAgent(
            tokenizer=kwargs["tokenizer"], decode=kwargs["decode"], execute=execute, config=kwargs["config"]
        )
        return await agent.run_async("task"), {"reward": 0.75}, "/trial"

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    args = args_for(tmp_path)
    base = Sample(index=9, prompt="task", metadata={"task_path": "task", "terminal": {"max_steps": 5}})
    params = {"temperature": 1.0, "max_new_tokens": 512}
    result = await miles_terminal.generate(
        SimpleNamespace(
            args=args,
            sample=base,
            evaluation=evaluation,
            state=SimpleNamespace(tokenizer=Tokenizer()),
            sampling_params=params,
        )
    )
    assert len(result.samples) == 2
    assert all(s.rollout_id == 9 and s.reward == 0.75 and s.status == Sample.Status.COMPLETED for s in result.samples)
    assert all(s.metadata["n_compactions"] == 1 for s in result.samples)
    for sample in result.samples:
        sample.validate()
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
    assert [r["sampling_params"]["max_new_tokens"] for r in requests] == [512, 200, 512]
    assert all(r["return_logprob"] for r in requests)
    assert args.terminal_max_steps == 100 and params["max_new_tokens"] == 512
    assert base.tokens == []  # input sample is not mutated


def test_launch_constraint_and_request_override(tmp_path):
    args = args_for(tmp_path)
    assert miles_terminal._config(args, {"terminal": {"max_steps": 4}}).max_steps == 4
    with pytest.raises(ValueError, match="rollout-max-context-len"):
        miles_terminal._config(args, {"terminal": {"max_context_tokens": 9000}})
    args.use_sampling_support_replay = True
    with pytest.raises(ValueError, match="sampling_support"):
        miles_terminal._validate_runtime(args)


def test_thinking_cli_and_per_task_override(tmp_path):
    parser = argparse.ArgumentParser()
    miles_terminal.generate.add_arguments(parser)
    assert parser.parse_args(["--no-terminal-enable-thinking"]).terminal_enable_thinking is False
    assert parser.parse_args(["--terminal-enable-thinking"]).terminal_enable_thinking is True
    args = args_for(tmp_path)
    assert miles_terminal._config(args, {"terminal": {"enable_thinking": False}}).enable_thinking is False
    assert args.terminal_enable_thinking is True
    with pytest.raises(ValueError, match="enable_thinking"):
        miles_terminal._config(args, {"terminal": {"enable_thinking": "false"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [{"other": 1.0}, {"reward": float("nan")}])
async def test_wrapper_rejects_missing_or_invalid_reward(monkeypatch, tmp_path, reward):
    from agentic_critic.terminal_env import TerminalTrace

    make_task(tmp_path)

    async def trial(**kwargs):
        return TerminalTrace(), reward, "/trial"

    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    with pytest.raises(ValueError, match="reward"):
        await miles_terminal.generate(
            SimpleNamespace(
                args=args_for(tmp_path),
                sample=Sample(index=1, metadata={"task_path": "task"}),
                state=SimpleNamespace(tokenizer=Tokenizer()),
                evaluation=True,
                sampling_params={},
            )
        )
