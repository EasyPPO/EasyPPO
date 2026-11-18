import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

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
        "--eval-samples-per-prompt": "3",
        "--max-context-tokens": "131072",
        "--terminal-environment": "shared-judge",
        "--sglang-server-concurrency": "128",
        "--critic-context-mode": mode,
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--terminal-docker-compose" not in args and "--terminal-memory-mb" not in args


def test_docker_launch_preserves_resources_and_compose():
    _, config = parse_args(["--terminal-environment", "docker"])
    command = config.command()
    for flag, value in {
        "--terminal-environment": "docker",
        "--terminal-memory-mb": "16384",
        "--terminal-docker-compose": f"{config.source}/modal_launchers/docker-compose-host.yaml",
        "--sglang-server-concurrency": "8",
    }.items():
        assert command[command.index(flag) + 1] == value
    with pytest.raises(ValueError, match="requires --terminal-environment docker"):
        parse_args(["--terminal-memory-mb", "16384"])


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
            "--terminal-environment",
            "docker",
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


def test_legacy_run_accepts_original_controls_and_pins_them(tmp_path):
    from pathlib import Path

    config = Config(
        mode="full_detach",
        run_id="test",
        mount=str(tmp_path),
        eval_samples_per_prompt=5,
        terminal_environment="docker",
    )
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
        "terminal_environment",
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
    direct = Config(mode="full_detach", run_id="test", mount=str(tmp_path), eval_samples_per_prompt=5, concurrency=8)
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(direct, wandb_project="test", wandb_entity=None)
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(replace(config, eval_samples_per_prompt=3), wandb_project="test", wandb_entity=None)
    check_manifest(config, wandb_project="test", wandb_entity=None)
    assert json.loads(path.read_text()) == current
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(replace(config, eval_max_tasks=2), wandb_project="test", wandb_entity=None)


@pytest.mark.parametrize("change", [{"run_id": "../escape"}, {"gpus": 32}, {"max_restarts": -1}])
def test_invalid_plan_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(mode="full_detach", run_id="test"), **change)


@pytest.mark.parametrize(
    "failure,expected_error,expected_spawns",
    [
        ("preflight", "preflight timed out", 0),
        ("training", "training failed", 1),
        ("training_timeout", "trial timed out", 1),
        ("function_timeout", "function timed out", 1),
        ("remote_timeout_at_deadline", "Training operation timed out: trial timed out", 1),
        ("cleanup", "cleanup timed out", 1),
        ("judge_lifetime", "Restarting before the judge expires", 3),
    ],
)
def test_restart_policy(monkeypatch, failure, expected_error, expected_spawns):
    class FunctionTimeoutError(Exception):
        pass

    events = []
    now = 0

    def wait(*, timeout):
        nonlocal now
        if failure == "judge_lifetime":
            now += timeout
            raise TimeoutError("poll timed out")
        if failure == "remote_timeout_at_deadline":
            now += timeout
            return remote_training("fake-judge", "fake-token")
        if failure == "function_timeout":
            now += timeout
            raise FunctionTimeoutError("function timed out")
        if failure == "training_timeout":
            raise TimeoutError("trial timed out")
        raise RuntimeError("training failed")

    def terminate():
        events.append("terminate")
        if failure == "cleanup":
            raise FunctionTimeoutError("cleanup timed out")

    judge = Mock()
    judge.terminate.side_effect = terminate
    call = Mock()
    call.get.side_effect = wait
    call.cancel.side_effect = lambda **kwargs: events.append("cancel")
    preflight = Mock()
    if failure == "preflight":
        preflight.remote.side_effect = FunctionTimeoutError("preflight timed out")
    else:
        preflight.remote.return_value = False
    training = Mock()
    training.spawn.return_value = call
    modal = SimpleNamespace(
        Sandbox=SimpleNamespace(create=Mock(return_value=judge)),
        Probe=SimpleNamespace(with_tcp=Mock()),
        exception=SimpleNamespace(FunctionTimeoutError=FunctionTimeoutError),
        enable_output=nullcontext,
    )
    app = SimpleNamespace(run=nullcontext)
    if failure == "remote_timeout_at_deadline":
        from modal_launchers import runtime

        # The decorator returns the local function. Invoke its error-handling
        # boundary with a failing Volume reload; no Modal calls are made.
        sdk = Mock()
        sdk.App.return_value.function.return_value = lambda fn: fn
        sdk.Volume.from_name.return_value.reload.side_effect = TimeoutError("trial timed out")
        monkeypatch.setattr(launcher, "_training_image", lambda *args: None)
        monkeypatch.setattr(launcher, "_secrets", lambda *args: [])
        monkeypatch.setattr(runtime, "train", Mock(side_effect=AssertionError("unexpected train call")))
        _, _, _, remote_training = launcher._build_training(sdk, Config(mode="full_detach", run_id="test"))
    monkeypatch.setattr(launcher, "_build_training", lambda *args: (app, None, preflight, training))
    monkeypatch.setattr(launcher, "_judge_image", lambda *args: None)
    monkeypatch.setattr(launcher, "_check_judge", lambda *args: None)
    monkeypatch.setattr(launcher, "time", SimpleNamespace(monotonic=lambda: now))

    with pytest.raises((TimeoutError, RuntimeError, FunctionTimeoutError), match=expected_error) as caught:
        launcher._train(modal, Config(mode="full_detach", run_id="test", max_restarts=2))
    if failure == "cleanup":
        assert str(caught.value.__context__) == "training failed"
    assert training.spawn.call_count == expected_spawns
    assert events == (["cancel", "terminate"] * expected_spawns if expected_spawns else ["terminate"])
