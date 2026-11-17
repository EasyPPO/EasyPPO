"""Exercise the pinned Harbor API without requiring a container daemon."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
pytest.importorskip("minisweagent")

from agentic_critic.harbor_terminal import MiniSweTerminalAgent, run_trial
from agentic_critic.terminal_env import TerminalConfig, TerminalTrace
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from tests.agentic_critic.test_terminal_data import make_task
from tests.agentic_critic.test_terminal_env import Tokenizer, action


@pytest.mark.asyncio
async def test_native_harbor_trial_constructs_custom_agent(tmp_path):
    task = make_task(tmp_path / "tasks")
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task),
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(import_path="agentic_critic.harbor_terminal:MiniSweTerminalAgent"),
        )
    )
    try:
        assert isinstance(trial.agent, MiniSweTerminalAgent)
        assert trial.task.instruction == "Create the requested artifact."
        assert trial.config.environment.delete
    finally:
        trial._close_logger_handler()


@pytest.mark.asyncio
async def test_adapter_uses_real_mini_parser_and_writes_trace(tmp_path):
    commands = []

    async def exec_command(**kwargs):
        commands.append(kwargs)
        return SimpleNamespace(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", stderr="", return_code=0)

    async def decode(prompt, limit):
        return action("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    agent = MiniSweTerminalAgent(logs_dir=tmp_path)
    agent.bind(tokenizer=Tokenizer(), decode=decode, config=TerminalConfig())
    context = AgentContext()
    await agent.run("Do the task", SimpleNamespace(exec=exec_command), context)
    assert agent.trace.exit_status == "Submitted" and context.n_output_tokens > 0
    assert commands == [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "timeout_sec": 120}]
    assert json.loads((tmp_path / "easyppo-trajectory.json").read_text())["trace"]["exit_status"] == "Submitted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,has_reward,raises",
    [
        (None, True, False),
        (None, False, True),
        ("VerifierTimeoutError", True, True),
        ("EnvironmentStartTimeoutError", False, True),
        ("AgentTimeoutError", True, False),
    ],
)
async def test_trial_failure_and_reward_contract(monkeypatch, tmp_path, error, has_reward, raises):
    from agentic_critic import harbor_terminal

    trace = TerminalTrace()
    result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=error, exception_message="failure") if error else None,
        verifier_result=SimpleNamespace(rewards={"reward": 0.5}) if has_reward else None,
    )

    async def run():
        return result

    async def create(config):
        assert config.environment.delete and not config.verifier.disable
        return SimpleNamespace(
            config=config,
            agent=SimpleNamespace(bind=lambda **kw: None, trace=trace),
            run=run,
            paths=SimpleNamespace(trial_dir=tmp_path),
        )

    monkeypatch.setattr(harbor_terminal.Trial, "create", create)
    kwargs = dict(
        task_path=tmp_path,
        trials_dir=tmp_path,
        environment="docker",
        tokenizer=Tokenizer(),
        decode=None,
        config=TerminalConfig(),
    )
    if raises:
        with pytest.raises(RuntimeError):
            await run_trial(**kwargs)
    else:
        actual, rewards, _ = await run_trial(**kwargs)
        assert rewards == {"reward": 0.5}
        assert actual.truncated == (error == "AgentTimeoutError")


@pytest.mark.asyncio
async def test_adapter_persists_cancelled_trace_and_propagates_cancellation(tmp_path):
    import asyncio

    async def decode(prompt, limit):
        raise asyncio.CancelledError

    agent = MiniSweTerminalAgent(logs_dir=tmp_path)
    agent.bind(tokenizer=Tokenizer(), decode=decode, config=TerminalConfig())
    with pytest.raises(asyncio.CancelledError):
        await agent.run("task", SimpleNamespace(), AgentContext())
    recorded = json.loads((tmp_path / "easyppo-trajectory.json").read_text())
    assert recorded["trace"]["exit_status"] == "Cancelled" and recorded["trace"]["truncated"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, False])
async def test_harbor_command_timeout_recovers_but_infrastructure_failure_propagates(tmp_path, timeout):
    commands = []

    async def execute(**kwargs):
        commands.append(kwargs["command"])
        if len(commands) == 1:
            if timeout:
                try:
                    raise TimeoutError()
                except TimeoutError as exc:
                    raise RuntimeError("Command timed out after 120 seconds") from exc
            raise RuntimeError("Container connection lost")
        return SimpleNamespace(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", stderr="", return_code=0)

    replies = iter([action("long-command"), action("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])

    async def decode(prompt, limit):
        return next(replies)

    agent = MiniSweTerminalAgent(logs_dir=tmp_path)
    agent.bind(tokenizer=Tokenizer(), decode=decode, config=TerminalConfig())
    if timeout:
        await agent.run("task", SimpleNamespace(exec=execute), AgentContext())
        assert agent.trace.exit_status == "Submitted" and len(commands) == 2
        messages = json.loads((tmp_path / "easyppo-trajectory.json").read_text())["messages"]
        assert any("command timed out" in m["content"].lower() for m in messages)
    else:
        with pytest.raises(RuntimeError, match="Container connection lost"):
            await agent.run("task", SimpleNamespace(exec=execute), AgentContext())
