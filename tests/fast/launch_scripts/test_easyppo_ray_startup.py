"""Record Ray startup without starting services, SSH sessions or GPU work."""

import shlex
from unittest.mock import Mock

import pytest
import typer
from examples.easyppo import async_ppo as recipe
from tests.fast.launch_scripts.py_harness import (
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT
from typer.testing import CliRunner

from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend
from miles.utils.workers.types import ClusterBackend


@pytest.fixture(
    params=[f"{schedule}_{dataset}" for schedule in ("sync", "async") for dataset in ("frontiersmith", "tmax")]
)
def launcher(request):
    return import_launch_script(REPO_ROOT / f"examples/easyppo/run_qwen3_5_27b_{request.param}.py")


@pytest.fixture
def commands(monkeypatch):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.10")
    return install_command_recorder(monkeypatch).commands


def test_head_then_ssh_workers_then_job_submission(launcher, commands):
    args = launcher.ScriptArgs(ray_hostfile="/shared/hosts with spaces")
    launcher.execute(args)
    head = next(i for i, command in enumerate(commands) if "ray start --head" in command)
    workers = next(i for i, command in enumerate(commands) if 'ssh root@"$worker_ip"' in command)
    job = next(i for i, command in enumerate(commands) if "ray job submit" in command)
    assert head < workers < job
    assert shlex.quote(args.ray_hostfile) in commands[workers]
    assert "--address=10.0.0.10:6379 --num-gpus 8" in commands[workers]
    assert 'if [ "$worker_ip" = 10.0.0.10 ]; then continue;' in commands[workers]


def test_hostfile_head_alias_is_skipped(launcher, commands, monkeypatch):
    monkeypatch.setenv("MLP_WORKER_0_HOST", "head-node")
    launcher.execute(launcher.ScriptArgs())
    worker_command = next(command for command in commands if 'ssh root@"$worker_ip"' in command)
    assert 'if [ "$worker_ip" = head-node ]; then continue;' in worker_command
    assert "--address=10.0.0.10:6379" in worker_command


def test_external_cluster_skips_head_and_worker_startup(launcher, commands, monkeypatch):
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    launcher.execute(launcher.ScriptArgs())
    assert not any("ray start" in command or "ray stop" in command or "ssh root@" in command for command in commands)
    assert any("ray job submit" in command for command in commands)


def test_cli_can_disable_worker_joining(launcher, commands):
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(app, ["--action", "train", "--no-join-ray-workers"])
    assert result.exit_code == 0, result.output
    assert any("ray start --head" in command for command in commands)
    assert not any("ssh root@" in command for command in commands)


def test_cli_hostfile_reaches_worker_joining(launcher, commands):
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(app, ["--action", "train", "--ray-hostfile", "/shared/custom-hosts"])
    assert result.exit_code == 0, result.output
    assert any("/shared/custom-hosts" in command and "ssh root@" in command for command in commands)


def test_missing_head_address_fails_before_any_commands(launcher, commands, monkeypatch):
    monkeypatch.delenv("MASTER_ADDR")
    with pytest.raises(ValueError, match="MASTER_ADDR"):
        launcher.execute(launcher.ScriptArgs())
    assert commands == []


def test_prepare_does_not_require_ray_address(launcher, commands, monkeypatch, tmp_path):
    monkeypatch.delenv("MASTER_ADDR")
    launcher.prepare(launcher.ScriptArgs(model_dir=str(tmp_path / "models"), data_dir=str(tmp_path / "data")))
    assert commands
    assert not any("ray start" in command or "ssh root@" in command for command in commands)


def test_non_ray_backend_does_not_receive_ssh_callback(launcher, monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    args = launcher.ScriptArgs(cluster_backend=ClusterBackend.KUBERNETES)
    backend = Mock(spec=BaseCommandBackend)
    assert recipe._ray_worker_startup(args, backend) is None
    assert backend.mock_calls == []
