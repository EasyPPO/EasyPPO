"""CPU contracts for Harbor's retry queue and Modal transport; no cloud calls."""

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

pytest.importorskip("harbor")
pytest.importorskip("modal")

from harbor.trial.trial import Trial

from agentic_critic.harbor_modal import TerminalModalEnvironment, _stage_command
from agentic_critic.harbor_terminal import run_trial
from agentic_critic.terminal_env import TerminalConfig, TerminalTrace
from scripts.make_miles_frontiersmith_data import _install_verifier


@pytest.mark.parametrize("hosts", [["other:1.2.3.4"], {"other": "1.2.3.4"}])
def test_remote_judge_removes_only_local_gateway_and_reprepare_is_reversible(tmp_path, hosts):
    (tmp_path / "environment").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "task.toml").write_text("")
    compose = tmp_path / "environment/docker-compose.yaml"
    compose.write_text(yaml.safe_dump({"services": {"main": {"extra_hosts": hosts}}}))
    for url, local in [("http://host.docker.internal:8081", True), ("https://judge.modal.host", False)] * 2:
        _install_verifier(tmp_path, judge_url=url, judge_timeout=10)
        actual = yaml.safe_load(compose.read_text())["services"]["main"]["extra_hosts"]
        assert ("host.docker.internal" in str(actual)) == local
        assert "other" in str(actual)


async def test_long_unicode_command_is_uploaded_and_runs_with_exact_content(tmp_path):
    uploads = []

    async def upload(source, target):
        # Use the same absolute path as the sandbox, on this CPU-only test host.
        Path(target).write_bytes(source.read_bytes())
        uploads.append((target, source.read_bytes()))

    command = "#" + "界" * 30000 + "\n" + r"printf '\377ok'" + "\nprintf 'err' >&2\nexit 7"
    staged = await _stage_command(command, upload)
    assert len(staged.encode()) < 65536
    assert uploads[0][1] == command.encode()
    result = subprocess.run(["bash", "-c", staged], capture_output=True)
    assert result.returncode == 7 and result.stdout == b"\xffok" and result.stderr == b"err"
    assert not Path(uploads[0][0]).exists()


async def test_modal_byte_output_preserves_return_code_and_exec_options():
    process = SimpleNamespace(
        stdout=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=b"out\xff"))),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=b"err\xfe"))),
        wait=SimpleNamespace(aio=AsyncMock(return_value=23)),
    )
    execute = AsyncMock(return_value=process)
    env = object.__new__(TerminalModalEnvironment)
    env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=execute))
    env._merge_env = lambda env: env
    result = await env._sdk_exec("command", cwd="/app", timeout_sec=9, shell="sh", login=True)
    assert (result.stdout, result.stderr, result.return_code) == ("out\ufffd", "err\ufffd", 23)
    execute.assert_awaited_once_with("sh", "-lc", "command", workdir="/app", secrets=[], timeout=9, text=False)


async def test_modal_transport_keeps_oom_detection_and_cancellation():
    env = object.__new__(TerminalModalEnvironment)
    env._merge_env = lambda env: env
    env._raise_if_unexpected_exit_137 = AsyncMock()
    execute = AsyncMock(side_effect=ConnectionError("lost"))
    env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=execute))
    with pytest.raises(ConnectionError):
        await env._sdk_exec("true")
    env._raise_if_unexpected_exit_137.assert_awaited_once()
    execute.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await env._sdk_exec("true")
    assert env._raise_if_unexpected_exit_137.await_count == 1


@pytest.mark.parametrize(
    "error,failures,expected",
    [
        ("EnvironmentStartTimeoutError", 1, 2),
        ("RewardFileNotFoundError", 1, 2),
        ("RewardFileNotFoundError", 9, 3),
        ("ValueError", 9, 1),
        ("AgentTimeoutError", 1, 1),
    ],
)
async def test_native_harbor_retries_are_bounded_and_bind_each_attempt(
    monkeypatch, tmp_path, error, failures, expected
):
    attempts, bindings = [], []

    async def create(config):
        attempts.append(config)
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)

        async def run():
            failed = len(attempts) <= failures
            return SimpleNamespace(
                exception_info=SimpleNamespace(exception_type=error, exception_message="failure") if failed else None,
                verifier_result=(
                    SimpleNamespace(rewards={"reward": 0.7}) if not failed or error == "AgentTimeoutError" else None
                ),
            )

        return SimpleNamespace(
            config=config,
            run=run,
            paths=SimpleNamespace(trial_dir=trial_dir),
            agent=SimpleNamespace(bind=lambda **kw: bindings.append(kw), trace=TerminalTrace()),
        )

    monkeypatch.setattr(Trial, "create", create)
    # Keep the framework's actual retry loop while removing only backoff time.
    monkeypatch.setattr("harbor.trial.queue.asyncio.sleep", AsyncMock())
    kwargs = dict(
        task_path=tmp_path,
        trials_dir=tmp_path,
        environment="modal",
        tokenizer=None,
        decode=None,
        config=TerminalConfig(),
        memory_mb=16384,
        max_retries=2,
    )
    if failures > 2:
        with pytest.raises(RuntimeError, match=error):
            await run_trial(**kwargs)
    else:
        trace, rewards, _ = await run_trial(**kwargs)
        assert rewards == {"reward": 0.7}
        assert trace.truncated == (error == "AgentTimeoutError")
    assert len(attempts) == len(bindings) == expected
    assert attempts[0].environment.override_memory_mb == 16384
    assert attempts[0].environment.import_path == "agentic_critic.harbor_modal:TerminalModalEnvironment"


async def test_harbor_trial_cancellation_is_not_retried(monkeypatch, tmp_path):
    create = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(Trial, "create", create)
    with pytest.raises(asyncio.CancelledError):
        await run_trial(
            task_path=tmp_path,
            trials_dir=tmp_path,
            environment="docker",
            tokenizer=None,
            decode=None,
            config=TerminalConfig(),
        )
    assert create.await_count == 1
