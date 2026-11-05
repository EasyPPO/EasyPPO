"""The synchronous recipes share tuning while disabling asynchronous SC state."""

import base64
import json
import os
import shlex
import subprocess
import sys

import pytest
import typer
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT
from typer.testing import CliRunner


@pytest.fixture(params=["frontiersmith", "tmax"])
def launcher(request):
    return import_launch_script(REPO_ROOT / f"examples/easyppo/run_qwen3_5_27b_sync_{request.param}.py")


class _Backend:
    def __init__(self):
        self.commands = []
        self.conversions = []
        self.training = []

    def exec_command_cpu(self, command):
        self.commands.append(shlex.split(command))

    def convert_checkpoint(self, **kwargs):
        self.conversions.append(kwargs)

    def execute_train(self, **kwargs):
        self.training.append(kwargs)


def _backend(monkeypatch, launcher):
    backend = _Backend()
    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: backend)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return backend


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_sync_argv_preserves_tuning_without_sc_or_old_actor(launcher, monkeypatch, tmp_path, mode):
    backend = _backend(monkeypatch, launcher)
    args = launcher.ScriptArgs(
        run_id="sync-parity",
        data_dir=str(tmp_path / "data with spaces"),
        output_dir=str(tmp_path / "runs with spaces"),
        critic_context_mode=mode,
    )
    launcher.execute(args)
    [submitted] = backend.training
    argv = shlex.split(submitted["train_args"])

    def value(flag):
        assert argv.count(flag) == 1
        return argv[argv.index(flag) + 1]

    assert submitted["train_script"] == "train.py"
    assert submitted["megatron_model_type"] == "qwen3.5-27B"
    assert submitted["extra_env_vars"] == {"PYTHONPATH": args.megatron_path}
    expected = {
        "--advantage-estimator": "ppo",
        "--critic-context-mode": mode,
        "--rollout-batch-size": "32",
        "--over-sampling-batch-size": "35",
        "--n-samples-per-prompt": "16",
        "--global-batch-size": "512",
        "--critic-num-steps-per-rollout": "4",
        "--critic-return-std-floor": "0.075",
        "--num-critic-only-steps": "30",
        "--num-rollout": "120",
        "--lr": "1e-06",
        "--critic-lr": "5e-06",
        "--adam-beta1": "0.9",
        "--adam-beta2": "0.999",
        "--weight-decay": "0.01",
        "--eps-clip": "0.2",
        "--eps-clip-high": "0.28",
        "--value-clip": "0.2",
        "--value-loss-coef": "0.5",
        "--gamma": "1",
        "--lambd": "1",
        "--kl-coef": "0",
        "--terminal-max-total-tokens": "120000",
        "--terminal-max-context-tokens": "32768",
        "--rollout-max-context-len": "32768",
        "--max-tokens-per-gpu": "262144",
        "--actor-num-nodes": "2",
        "--actor-num-gpus-per-node": "8",
        "--critic-num-nodes": "2",
        "--critic-num-gpus-per-node": "8",
        "--rollout-num-gpus": "48",
        "--rollout-num-gpus-per-engine": "4",
        "--tensor-model-parallel-size": "4",
    }
    assert {flag: value(flag) for flag in expected} == expected
    assert {"--filter-truncated-in-actor-loss", "--offload-train", "--calculate-per-token-loss"} <= set(argv)
    assert not {
        "--keep-old-actor",
        "--use-tis",
        "--custom-tis-function-path",
        "--use-rollout-logprobs",
        "--sglang-enable-fp32-lm-head",
        "--rollout-top-logprobs-num",
        "--score-centering-is",
        "--score-centering-tis-clip",
        "--recompute-loss-function",
        "--fully-async",
    } & set(argv)
    assert "--loss-type" not in argv  # The existing Miles default is policy_loss.
    evaluation = json.loads(base64.b64decode(value("--eval-config").removeprefix("base64:")))["datasets"][0]
    assert evaluation["path"] == args.validation_data
    assert evaluation["name"] == ("fcs_quick" if args.dataset == "frontiersmith" else "tb21_quick")
    assert evaluation["max_samples"] == 16 and evaluation["sample_seed"] == 42
    assert value("--prompt-data") == args.train_data

    # All remaining flags must be identical to the existing async clipped-PPO
    # recipe, including order. The schedule and the old-actor copy are the changes.
    asynchronous = import_launch_script(REPO_ROOT / "examples/easyppo/run_qwen3_5_27b_async_ppo.py")
    monkeypatch.setattr(asynchronous.ScriptArgs, "create_backend", lambda self: backend)
    asynchronous.execute(asynchronous.ScriptArgs(**vars(args)))
    async_argv = shlex.split(backend.training[-1]["train_args"])
    assert "--keep-old-actor" in async_argv
    assert argv == [flag for flag in async_argv if flag != "--keep-old-actor"]
    assert backend.training[-1]["train_script"] == "train_async.py"


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_prepare_uses_the_fixed_training_and_evaluation_corpus(launcher, monkeypatch, tmp_path, mode):
    backend = _backend(monkeypatch, launcher)
    args = launcher.ScriptArgs(
        critic_context_mode=mode,
        model_dir=str(tmp_path / "models with spaces"),
        data_dir=str(tmp_path / "data with spaces"),
    )
    launcher.prepare(args)
    assert backend.commands[1][:3] == ["hf", "download", args.hf_repo]
    assert backend.commands[1][backend.commands[1].index("--revision") + 1] == args.hf_revision
    preparation = backend.commands[2]
    if args.dataset == "frontiersmith":
        assert preparation[1].endswith("make_miles_frontiersmith_data.py")
        assert "--frontier-cs-source-dir" in preparation
        assert args.validation_data.endswith("fcs.eval.jsonl")
    else:
        assert preparation[1].endswith("make_miles_terminal_data.py")
        assert preparation[preparation.index("--dataset") + 1] == "tmax"
        assert "--tokenizer" in preparation
        assert args.validation_data.endswith("tb21.eval.jsonl")
    assert backend.conversions[0]["megatron_model_type"] == "qwen3.5-27B"
    assert backend.training == []


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("action", ["prepare", "train", "all"])
def test_main_dispatches_the_real_prepare_and_sync_execute(launcher, monkeypatch, mode, action):
    backend = _backend(monkeypatch, launcher)
    launcher.main(action=action, critic_context_mode=mode, run_id="sync-main")
    assert bool(backend.commands) == (action in ("prepare", "all"))
    assert bool(backend.conversions) == (action in ("prepare", "all"))
    assert bool(backend.training) == (action in ("train", "all"))
    if backend.training:
        assert backend.training[0]["train_script"] == "train.py"


@pytest.mark.parametrize(
    "override",
    [
        {"score_centering": True},
        {"score_centering_candidates": 128},
        {"score_centering_is": "tis"},
        {"score_centering_is": "mis"},
        {"score_centering_recompute_loss": True},
        {"score_centering_tis_clip": 1.5},
        {"score_centering_mis_low": 0.7},
        {"score_centering_mis_high": 3.0},
        {"score_centering_chunk_size": 64},
        {"score_centering_tis_clip": float("nan")},
        {"score_centering_mis_high": float("inf")},
    ],
)
def test_sync_rejects_score_centering_before_backend_commands(launcher, monkeypatch, override):
    backend = _backend(monkeypatch, launcher)
    with pytest.raises(ValueError, match="[Ss]core.centering|score-centering"):
        launcher.main(**override)
    assert not backend.commands and not backend.conversions and not backend.training


def test_each_wrapper_rejects_a_different_dataset(launcher, monkeypatch):
    backend = _backend(monkeypatch, launcher)
    other_dataset = "tmax" if launcher.ScriptArgs().dataset == "frontiersmith" else "frontiersmith"
    with pytest.raises(ValueError, match="requires --dataset"):
        launcher.ScriptArgs(dataset=other_dataset)
    for entrypoint in (launcher.prepare, launcher.execute):
        args = launcher.ScriptArgs()
        args.dataset = other_dataset
        with pytest.raises(ValueError, match="requires --dataset"):
            entrypoint(args)
    assert not backend.commands and not backend.conversions and not backend.training


def test_cli_explicit_sc_enable_is_rejected(launcher, monkeypatch):
    backend = _backend(monkeypatch, launcher)
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(app, ["--score-centering", "--action", "train"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "do not support --score-centering" in str(result.exception)
    assert not backend.commands and not backend.conversions and not backend.training


def test_cli_explicit_no_sc_uses_sync_driver(launcher, monkeypatch):
    backend = _backend(monkeypatch, launcher)
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(
        app, ["--no-score-centering", "--action", "train", "--critic-context-mode", "actor-summary"]
    )
    assert result.exit_code == 0, result.output
    assert backend.training[0]["train_script"] == "train.py"
    assert "--keep-old-actor" not in shlex.split(backend.training[0]["train_args"])


def test_cli_dataset_literal_prevents_cross_recipe_dispatch(launcher, monkeypatch):
    backend = _backend(monkeypatch, launcher)
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    other_dataset = "tmax" if launcher.ScriptArgs().dataset == "frontiersmith" else "frontiersmith"
    result = CliRunner().invoke(app, ["--dataset", other_dataset, "--action", "train"])
    assert result.exit_code == 2
    assert not backend.commands and not backend.conversions and not backend.training


def test_file_invocation_help_needs_neither_pythonpath_nor_repo_working_directory(launcher, tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["NO_COLOR"] = "1"
    environment["COLUMNS"] = "240"  # Keep Rich from ellipsizing long option names.
    result = subprocess.run(
        [sys.executable, launcher.__file__, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--action" in result.stdout and "--critic-context-mode" in result.stdout
    assert launcher.ScriptArgs().dataset in result.stdout
