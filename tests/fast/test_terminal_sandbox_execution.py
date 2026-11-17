"""CPU contracts for Harbor trial failures and Docker configuration; no cloud calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

pytest.importorskip("harbor")

from harbor.trial.trial import Trial

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


@pytest.mark.parametrize("error", [None, "EnvironmentStartTimeoutError", "RewardFileNotFoundError"])
async def test_harbor_trial_runs_once_and_forwards_docker_configuration(monkeypatch, tmp_path, error):
    attempts, bindings = [], []

    async def create(config):
        attempts.append(config)
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)

        async def run():
            return SimpleNamespace(
                exception_info=SimpleNamespace(exception_type=error, exception_message="failure") if error else None,
                verifier_result=SimpleNamespace(rewards={"reward": 0.7}) if not error else None,
            )

        return SimpleNamespace(
            config=config,
            run=run,
            paths=SimpleNamespace(trial_dir=trial_dir),
            agent=SimpleNamespace(bind=lambda **kw: bindings.append(kw), trace=TerminalTrace()),
        )

    monkeypatch.setattr(Trial, "create", create)
    kwargs = dict(
        task_path=tmp_path,
        trials_dir=tmp_path,
        environment="docker",
        tokenizer=None,
        decode=None,
        config=TerminalConfig(),
        memory_mb=16384,
        docker_compose=(str(tmp_path / "host.yaml"),),
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            await run_trial(**kwargs)
    else:
        trace, rewards, _ = await run_trial(**kwargs)
        assert rewards == {"reward": 0.7}
        assert not trace.truncated
    assert len(attempts) == len(bindings) == 1
    assert attempts[0].environment.override_memory_mb == 16384
    assert attempts[0].environment.import_path is None
    assert attempts[0].environment.extra_docker_compose == [tmp_path / "host.yaml"]


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
