"""Docker ownership and cleanup without starting containers."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_launchers import docker_runtime


def test_existing_daemon_is_not_owned(monkeypatch, tmp_path):
    monkeypatch.setattr(docker_runtime.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    start = Mock(side_effect=AssertionError("must reuse the daemon"))
    monkeypatch.setattr(docker_runtime.subprocess, "Popen", start)
    with docker_runtime.docker_daemon(tmp_path / "docker.log"):
        pass
    start.assert_not_called()


@pytest.mark.parametrize("startup_failure", [False, True])
def test_owned_daemon_is_stopped_on_failure(monkeypatch, tmp_path, startup_failure):
    daemon = Mock()
    daemon.poll.return_value = 1 if startup_failure else None
    daemon.wait.side_effect = [subprocess.TimeoutExpired("dockerd", 30), 0]
    monkeypatch.setattr(docker_runtime.subprocess, "Popen", Mock(return_value=daemon))
    monkeypatch.setattr(
        docker_runtime.subprocess,
        "run",
        Mock(
            side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)]
        ),
    )
    with pytest.raises(RuntimeError, match="startup" if startup_failure else "training failed"):
        with docker_runtime.docker_daemon(tmp_path / "docker.log"):
            raise RuntimeError("training failed")
    daemon.terminate.assert_called_once()
    daemon.kill.assert_called_once()
    assert daemon.wait.call_count == 2
