import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from modal_launchers import frontiersmith as launcher
from modal_launchers.config import Config, check_manifest, parse_args


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_recipe_matches_claude(mode):
    config = Config(mode=mode, run_id="test")
    args = config.command()
    for flag, value in {
        "--num-rollout": "100",
        "--save-interval": "5",
        "--max-keep": "2",
        "--num-critic-only-steps": "30",
        "--critic-steps-per-rollout": "4",
        "--rollout-batch-size": "32",
        "--samples-per-prompt": "16",
        "--eval-max-tasks": "0",
        "--eval-samples-per-prompt": "5",
        "--max-context-tokens": "131072",
        "--terminal-environment": "docker",
        "--terminal-docker-compose": f"{config.source}/modal_launchers/docker-compose-host.yaml",
        "--sglang-server-concurrency": "8",
        "--critic-context-mode": mode,
    }.items():
        assert args[args.index(flag) + 1] == value


def test_preview_does_not_build_or_launch(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("preview allocated resources")

    monkeypatch.setattr(launcher, "_train", forbidden)
    launcher.main([])
    assert json.loads(capsys.readouterr().out)["launch"] is False


def test_one_round_smoke_plan_reaches_actor_update_without_startup_eval():
    args, config = parse_args(
        [
            "--run-id",
            "smoke",
            "--num-rollout",
            "1",
            "--num-critic-only-steps",
            "0",
            "--rollout-batch-size",
            "4",
            "--samples-per-prompt",
            "2",
            "--critic-steps-per-rollout",
            "1",
            "--eval-max-tasks",
            "2",
            "--eval-samples-per-prompt",
            "1",
            "--skip-eval-before-train",
            "--terminal-memory-mb",
            "32768",
            "--concurrency",
            "1",
        ]
    )
    assert not args.launch
    command = config.command()
    for flag, value in {
        "--num-rollout": "1",
        "--num-critic-only-steps": "0",
        "--rollout-batch-size": "4",
        "--samples-per-prompt": "2",
        "--critic-steps-per-rollout": "1",
        "--eval-max-tasks": "2",
        "--eval-samples-per-prompt": "1",
        "--terminal-memory-mb": "32768",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--skip-eval-before-train" in command


def test_manifest_rejects_changed_mode_or_topology(tmp_path):
    config = Config(mode="full_detach", run_id="test", mount=str(tmp_path))
    check_manifest(config, wandb_project="test", wandb_entity=None)
    check_manifest(config, wandb_project="test", wandb_entity=None)
    for changed in (replace(config, mode="actor-summary"), replace(config, gpus=4)):
        with pytest.raises(ValueError, match="configuration differs"):
            check_manifest(changed, wandb_project="test", wandb_entity=None)


def test_manifest_rejects_unowned_output(tmp_path):
    config = Config(mode="full_detach", run_id="test", mount=str(tmp_path))
    from pathlib import Path

    Path(config.output).mkdir(parents=True)
    Path(config.output, "checkpoints").mkdir()
    with pytest.raises(ValueError, match="no Modal manifest"):
        check_manifest(config, wandb_project="test", wandb_entity=None)


def test_legacy_run_accepts_default_controls_and_pins_them(tmp_path):
    from pathlib import Path

    config = Config(mode="full_detach", run_id="test", mount=str(tmp_path))
    check_manifest(config, wandb_project="test", wandb_entity=None)
    path = Path(config.output) / "modal_run.json"
    current = json.loads(path.read_text())
    unversioned = dict(current)
    unversioned.pop("agent_protocol")
    path.write_text(json.dumps(unversioned))
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(config, wandb_project="test", wandb_entity=None)
    legacy = dict(current)
    for name in (
        "terminal_memory_mb",
        "eval_max_tasks",
        "eval_samples_per_prompt",
        "skip_eval_before_train",
        "num_rollout",
        "num_critic_only_steps",
        "rollout_batch_size",
        "samples_per_prompt",
        "critic_steps_per_rollout",
    ):
        legacy.pop(name)
    path.write_text(json.dumps(legacy))
    check_manifest(config, wandb_project="test", wandb_entity=None)
    assert json.loads(path.read_text()) == current
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(replace(config, eval_max_tasks=2), wandb_project="test", wandb_entity=None)


@pytest.mark.parametrize("change", [{"run_id": "../escape"}, {"gpus": 32}, {"max_restarts": -1}])
def test_invalid_plan_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(mode="full_detach", run_id="test"), **change)


def test_cancel_gpu_before_terminating_judge(monkeypatch):
    events = []

    class Judge:
        def wait_until_ready(self, **kwargs):
            pass

        def create_connect_token(self, **kwargs):
            return SimpleNamespace(url="https://judge", token="x")

        def terminate(self):
            events.append("judge_terminated")

    class Call:
        def get(self, **kwargs):
            raise RuntimeError("training failed")

        def cancel(self, **kwargs):
            events.append("gpu_cancelled")

    modal = SimpleNamespace(
        Sandbox=SimpleNamespace(create=lambda *a, **kw: Judge()), Probe=SimpleNamespace(with_tcp=lambda port: None)
    )
    monkeypatch.setattr(launcher, "_check_judge", lambda *args: None)
    with pytest.raises(RuntimeError, match="training failed"):
        launcher._attempt(
            modal,
            None,
            Config(mode="full_detach", run_id="test"),
            None,
            SimpleNamespace(remote=lambda url: False),
            SimpleNamespace(spawn=lambda *a: Call()),
        )
    assert events == ["gpu_cancelled", "judge_terminated"]


@pytest.mark.parametrize("error, expected_attempts", [(launcher.JudgeLifetimeTimeout, 3), (RuntimeError, 1)])
def test_retries_are_bounded_and_only_for_timeouts(monkeypatch, error, expected_attempts):
    import sys
    from contextlib import nullcontext
    from types import ModuleType

    exceptions = ModuleType("modal.exception")
    exceptions.FunctionTimeoutError = type("FunctionTimeoutError", (TimeoutError,), {})
    monkeypatch.setitem(sys.modules, "modal.exception", exceptions)
    app = SimpleNamespace(run=nullcontext)
    modal = SimpleNamespace(enable_output=nullcontext)
    monkeypatch.setattr(launcher, "_build_training", lambda *args: (app, None, None, None))
    monkeypatch.setattr(launcher, "_judge_image", lambda *args: None)
    attempts = []

    def attempt(*args):
        attempts.append(1)
        raise error("injected failure")

    monkeypatch.setattr(launcher, "_attempt", attempt)
    with pytest.raises(error):
        launcher._train(modal, Config(mode="full_detach", run_id="test", max_restarts=2))
    assert len(attempts) == expected_attempts
