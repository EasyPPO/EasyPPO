"""Task command size, binary output and memory isolation preserve trial behavior."""

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor")
pytest.importorskip("modal")
pytest.importorskip("minisweagent")

from agentic_critic.harbor_modal import _STDIN_COMMAND, TerminalModalEnvironment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, ResourceMode, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from tests.agentic_critic.test_terminal_data import make_task


async def environment_with_output(tmp_path, stdout=b"", stderr=b"", return_code=0, memory_mode=ResourceMode.AUTO):
    process = SimpleNamespace(
        stdin=SimpleNamespace(write=Mock(), write_eof=Mock(), drain=SimpleNamespace(aio=AsyncMock())),
        stdout=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=stdout))),
        stderr=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=stderr))),
        wait=SimpleNamespace(aio=AsyncMock(return_value=return_code)),
    )
    task = make_task(tmp_path / "tasks")
    # Match the training tasks' Compose backend; the direct-image backend has
    # an optional Dockerfile parser dependency that this path does not use.
    (task / "environment/docker-compose.yaml").write_text("services:\n  main:\n    build: .\n")
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task),
            trials_dir=tmp_path / "trials",
            agent=AgentConfig(import_path="agentic_critic.harbor_terminal:MiniSweTerminalAgent"),
            environment=EnvironmentConfig(
                import_path="agentic_critic.harbor_modal:TerminalModalEnvironment",
                override_memory_mb=4096,
                memory_enforcement_policy=memory_mode,
                kwargs={"modal_vm_runtime": True},
            ),
        )
    )
    trial._close_logger_handler()
    environment = trial.agent_environment
    assert isinstance(environment, TerminalModalEnvironment)
    environment._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=AsyncMock(return_value=process)))
    environment._raise_if_unexpected_exit_137 = AsyncMock()
    return environment, process


@pytest.mark.asyncio
async def test_binary_output_preserves_valid_text_and_exit_status(tmp_path):
    environment, _ = await environment_with_output(tmp_path, "你好\n".encode() + b"\xfftail\xe4", b"error\x80\n", 7)
    result = await environment._sdk_exec("./solution", cwd="/app", timeout_sec=120, shell="sh", login=True)
    assert result.stdout == "你好\n\ufffdtail\ufffd"
    assert result.stderr == "error\ufffd\n"
    assert result.return_code == 7
    environment._sandbox.exec.aio.assert_awaited_once_with(
        "sh", "-lc", "./solution", workdir="/app", secrets=[], timeout=120, text=False
    )
    environment._raise_if_unexpected_exit_137.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_unicode_command_uses_stdin_and_preserves_payload_and_exec_settings(tmp_path):
    environment, process = await environment_with_output(tmp_path, b"result\xff", b"diagnostic", 7)
    command = "#" + "界" * 30000 + "\nprintf result\n\n"
    result = await environment._sdk_exec(command, cwd="/app", timeout_sec=120, shell="sh", login=True)
    args, kwargs = environment._sandbox.exec.aio.call_args
    assert args[:2] == ("sh", "-lc")
    assert sum(len(arg.encode()) for arg in args) < 65536
    assert kwargs == {"workdir": "/app", "secrets": [], "timeout": 120, "text": False}
    assert b"".join(call.args[0] for call in process.stdin.write.call_args_list) == command.encode()
    process.stdin.write_eof.assert_called_once_with()
    process.stdin.drain.aio.assert_awaited()
    assert (result.stdout, result.stderr, result.return_code) == ("result\ufffd", "diagnostic", 7)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("stdin failed"), asyncio.CancelledError()])
async def test_stdin_failure_and_cancellation_propagate_without_retry(tmp_path, error):
    environment, process = await environment_with_output(tmp_path)
    process.stdin.drain.aio.side_effect = error
    with pytest.raises(type(error)):
        await environment._sdk_exec("#" + "x" * 70000)
    environment._sandbox.exec.aio.assert_awaited_once()
    if isinstance(error, asyncio.CancelledError):
        environment._raise_if_unexpected_exit_137.assert_not_awaited()
    else:
        environment._raise_if_unexpected_exit_137.assert_awaited_once_with(cause=error)


@pytest.mark.parametrize("shell", ["sh", "bash"])
@pytest.mark.parametrize(
    "body",
    [
        'cat <<\'END\'\nquotes \' " $HOME $(false)\nEND\nprintf \'<%s>\\n\' "$PWD" "$0" "$MARKER"\nexit 7\n\n',
        "printf before; cat; printf after\n",
        "printf continued\\\n\n",
    ],
)
def test_stdin_transport_matches_direct_shell_semantics(tmp_path, shell, body):
    command = "#" + "界" * 22000 + "\n" + body
    options = dict(cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "MARKER": "preserved"}, capture_output=True)
    direct = subprocess.run([shell, "-c", command], input=b"", **options)
    streamed = subprocess.run([shell, "-c", _STDIN_COMMAND], input=command.encode(), **options)
    assert (streamed.returncode, streamed.stdout, streamed.stderr) == (
        direct.returncode,
        direct.stdout,
        direct.stderr,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["exec", "stdout", "stderr", "wait"])
async def test_transport_failures_propagate_without_retry(tmp_path, stage):
    environment, process = await environment_with_output(tmp_path)
    operation = (
        environment._sandbox.exec.aio
        if stage == "exec"
        else (process.wait.aio if stage == "wait" else getattr(process, stage).read.aio)
    )
    error = RuntimeError("transport failure")
    operation.side_effect = error
    with pytest.raises(RuntimeError, match="transport failure"):
        await environment._sdk_exec("command")
    environment._sandbox.exec.aio.assert_awaited_once()
    environment._raise_if_unexpected_exit_137.assert_awaited_once_with(cause=error)


@pytest.mark.asyncio
async def test_cancellation_propagates(tmp_path):
    environment, process = await environment_with_output(tmp_path)
    process.stdout.read.aio.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await environment._sdk_exec("command")
    environment._raise_if_unexpected_exit_137.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,expected",
    [(ResourceMode.AUTO, 6144), (ResourceMode.REQUEST, 6144), (ResourceMode.LIMIT, (6144, 6144))],
)
async def test_dind_reserves_host_memory_without_widening_task_limit(tmp_path, mode, expected):
    environment, _ = await environment_with_output(tmp_path, memory_mode=mode)
    assert environment._memory_config() == expected
    assert environment._effective_memory_mb == 4096
    if mode != ResourceMode.REQUEST:
        assert environment._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT) == 4096
