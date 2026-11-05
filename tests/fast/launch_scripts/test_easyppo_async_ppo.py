"""Verify the generated async recipe and reject invalid resource budgets offline."""

import base64
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from examples.easyppo import async_ppo as _recipe
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT
from typer.testing import CliRunner


@pytest.fixture(params=["frontiersmith", "tmax"])
def launcher(request):
    return import_launch_script(REPO_ROOT / f"examples/easyppo/run_qwen3_5_27b_async_{request.param}.py")


def _recording_backend(monkeypatch, launcher):
    class Backend:
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

    backend = Backend()
    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: backend)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return backend


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("centering", [False, True])
def test_one_step_async_recipe(launcher, monkeypatch, tmp_path, mode, centering):
    captured = {}

    class Backend:
        def execute_train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: Backend())
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = launcher.ScriptArgs(
        data_dir=str(tmp_path / "data with spaces"),
        output_dir=str(tmp_path / "runs"),
        critic_context_mode=mode,
        score_centering=centering,
    )
    launcher.execute(args)
    argv = shlex.split(captured["train_args"])

    def value(flag):
        assert argv.count(flag) == 1
        return argv[argv.index(flag) + 1]

    assert captured["train_script"] == "train_async.py"
    assert value("--critic-context-mode") == mode
    assert value("--lr") == "1e-06" and value("--critic-lr") == "5e-06"
    assert value("--adam-beta2") == "0.999" and value("--weight-decay") == "0.01"
    batch, oversample = "32", "35"
    assert value("--rollout-batch-size") == batch and value("--over-sampling-batch-size") == oversample
    assert value("--n-samples-per-prompt") == "16" and value("--global-batch-size") == "512"
    assert value("--terminal-max-total-tokens") == "120000"
    assert value("--terminal-max-context-tokens") == value("--rollout-max-context-len") == "32768"
    assert value("--max-tokens-per-gpu") == "262144"
    assert value("--tensor-model-parallel-size") == "4"
    assert int(value("--actor-num-nodes")) * int(value("--actor-num-gpus-per-node")) == 16
    assert value("--rollout-num-gpus") == "48" and value("--rollout-num-gpus-per-engine") == "4"
    assert args.num_nodes * args.num_gpus_per_node == 64
    assert (
        int(value("--actor-num-nodes")) * int(value("--actor-num-gpus-per-node")) + int(value("--rollout-num-gpus"))
        == 64
    )
    assert value("--update-weights-interval") == "1"
    assert "--offload-train" in argv
    assert value("--prompt-data") == args.train_data
    eval_config = json.loads(base64.b64decode(value("--eval-config").removeprefix("base64:")))
    eval_dataset = eval_config["datasets"][0]
    assert eval_dataset["path"] == args.validation_data
    assert eval_dataset["name"] == ("fcs" if args.dataset == "frontiersmith" else "tb21")
    assert eval_dataset["max_samples"] is None and eval_dataset["sample_seed"] == 42
    assert value("--eval-temperature") == "1.0" and value("--eval-top-p") == "1.0"
    assert value("--eval-top-k") == "-1" and value("--n-samples-per-eval-prompt") == "5"
    assert {"--no-eval-on-epoch", "--no-save-on-epoch"} <= set(argv)
    assert Path(value("--terminal-trials-dir")) == Path(args.output_dir) / args.run_id / "trials"
    assert value("--critic-num-steps-per-rollout") == "4"
    assert value("--critic-return-std-floor") == "0.075"
    assert "--filter-truncated-in-actor-loss" in argv
    assert value("--num-critic-only-steps") == "30"
    assert value("--kl-coef") == "0"
    if centering:
        assert value("--loss-type") == "score_centering"
        assert value("--rollout-top-logprobs-num") == "128"
        assert value("--score-centering-is") == "tis"
        assert value("--score-centering-tis-clip") == "2.0"
        assert "--recompute-loss-function" in argv
        assert "--use-rollout-logprobs" in argv and "--sglang-enable-fp32-lm-head" in argv
        assert "--keep-old-actor" not in argv and "--eps-clip" not in argv
        assert "--use-tis" not in argv and "--skip-actor-forward-only" not in argv
    else:
        assert "--loss-type" not in argv and "--rollout-top-logprobs-num" not in argv
        assert "--use-rollout-logprobs" not in argv and "--sglang-enable-fp32-lm-head" not in argv
        assert "--keep-old-actor" in argv and value("--eps-clip") == "0.2"
        assert value("--eps-clip-high") == "0.28"
    assert not {
        "--colocate",
        "--fully-async",
        "--normalize-advantages",
        "--use-kl-loss",
        "--use-dynamic-global-batch-size",
        "--use-rollout-routing-replay",
    } & set(argv)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_nodes": 1},
        {"rollout_num_gpus": 8},
        {"rollout_gpus_per_engine": 3},
        {"rollout_batch_size": 17},
        {"over_sampling_batch_size": 15},
        {"samples_per_prompt": 1},
        {"critic_steps_per_rollout": 3},
        {"critic_return_std_floor": 0},
        {"eval_max_tasks": -1},
        {"eval_samples_per_prompt": 0},
        {"eval_samples_per_prompt": -1},
        {"num_critic_only_steps": 121},
        {"num_critic_only_steps": -1},
        {"critic_context_mode": "full_history"},
        {"max_tokens_per_gpu": 100},
        {"eval_interval": 0},
        {"actor_lr": 0},
        {"top_p": 0},
        {"top_k": 0},
        {"max_steps": 0},
        {"compact_threshold": 32768},
        {"weight_decay": -1},
        {"critic_weights_only": True},
        {"score_centering_candidates": 0},
        {"score_centering_is": "custom"},
        {"score_centering_tis_clip": float("inf")},
        {"score_centering_tis_clip": 0},
        {"score_centering_mis_low": 6},
        {"score_centering_mis_high": float("nan")},
        {"score_centering_chunk_size": 0},
        {"temperature": 0.7},
    ],
)
def test_invalid_recipe_fails_before_commands(launcher, overrides):
    with pytest.raises(ValueError):
        _recipe._validate(launcher.ScriptArgs(**overrides))


@pytest.mark.parametrize("nodes,train_nodes,rollout", [(8, 2, 48), (4, 1, 24)])
def test_train_rollout_resource_options(launcher, nodes, train_nodes, rollout):
    args = launcher.ScriptArgs(num_nodes=nodes, trainer_num_nodes=train_nodes)
    _recipe._validate(args)
    assert args.rollout_num_gpus == rollout
    assert args.trainer_num_nodes * args.num_gpus_per_node + rollout == nodes * args.num_gpus_per_node


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_official_score_centering_weights_reach_the_existing_loss(launcher, mode):
    args = launcher.ScriptArgs(
        score_centering_is=mode,
        score_centering_tis_clip=1.5,
        score_centering_mis_low=0.7,
        score_centering_mis_high=3.0,
    )
    _recipe._validate(args)
    argv = _recipe._algorithm_args(args)
    assert argv[argv.index("--score-centering-is") + 1] == mode
    if mode == "tis":
        assert argv[argv.index("--score-centering-tis-clip") + 1] == "1.5"
    elif mode == "mis":
        assert argv[argv.index("--score-centering-mis-low") + 1] == "0.7"
        assert argv[argv.index("--score-centering-mis-high") + 1] == "3.0"
    assert "--use-tis" not in argv


def test_prepare_selects_the_data_recipe(launcher, monkeypatch, tmp_path):
    calls = []

    class Backend:
        def exec_command_cpu(self, cmd):
            calls.append(shlex.split(cmd))

        def convert_checkpoint(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: Backend())
    launcher.prepare(
        launcher.ScriptArgs(
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
        )
    )
    assert calls[1][:2] == ["hf", "download"]
    assert "--revision" in calls[1]
    if launcher.ScriptArgs().dataset == "frontiersmith":
        assert calls[2][1].endswith("make_miles_frontiersmith_data.py")
    else:
        assert calls[2][1].endswith("make_miles_terminal_data.py")
        assert "tmax" in calls[2] and "--tokenizer" in calls[2]
    assert calls[3]["megatron_model_type"] == "qwen3.5-27B"


def test_critic_warmstart_explicitly_resets_training_state(launcher):
    args = launcher.ScriptArgs(critic_load="/shared/warm-critic", critic_weights_only=True)
    _recipe._validate(args)
    argv = _recipe._checkpoint_args(args)
    assert argv[argv.index("--critic-load") + 1] == args.critic_load
    assert {"--finetune", "--no-load-optim", "--no-load-rng"} <= set(argv)


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("action", ["prepare", "train", "all"])
def test_main_dispatches_real_prepare_and_async_execute(launcher, monkeypatch, mode, action):
    backend = _recording_backend(monkeypatch, launcher)
    launcher.main(action=action, critic_context_mode=mode, run_id="async-main")
    assert bool(backend.commands) == (action in ("prepare", "all"))
    assert bool(backend.conversions) == (action in ("prepare", "all"))
    assert bool(backend.training) == (action in ("train", "all"))
    if backend.training:
        assert backend.training[0]["train_script"] == "train_async.py"


def test_wrapper_and_cli_refuse_the_other_dataset(launcher, monkeypatch):
    backend = _recording_backend(monkeypatch, launcher)
    other = "tmax" if launcher.ScriptArgs().dataset == "frontiersmith" else "frontiersmith"
    with pytest.raises(ValueError, match="requires --dataset"):
        launcher.ScriptArgs(dataset=other)
    for entrypoint in (launcher.prepare, launcher.execute):
        args = launcher.ScriptArgs()
        args.dataset = other
        with pytest.raises(ValueError, match="requires --dataset"):
            entrypoint(args)
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(app, ["--dataset", other, "--action", "train"])
    assert result.exit_code == 2
    assert not backend.commands and not backend.conversions and not backend.training


@pytest.mark.parametrize("maximum,samples", [(0, 5), (16, 1), (7, 3)])
def test_full_and_quick_eval_controls_reach_the_async_argv(launcher, monkeypatch, maximum, samples):
    backend = _recording_backend(monkeypatch, launcher)
    launcher.execute(launcher.ScriptArgs(eval_max_tasks=maximum, eval_samples_per_prompt=samples))
    argv = shlex.split(backend.training[0]["train_args"])
    assert argv[argv.index("--n-samples-per-eval-prompt") + 1] == str(samples)
    config = json.loads(base64.b64decode(argv[argv.index("--eval-config") + 1].removeprefix("base64:")))
    assert config["datasets"][0]["max_samples"] == (maximum or None)
    assert config["datasets"][0]["name"].endswith("_quick") == bool(maximum)


def test_file_invocation_help_without_pythonpath_or_repo_cwd(launcher, tmp_path):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(NO_COLOR="1", COLUMNS="240")
    result = subprocess.run(
        [sys.executable, launcher.__file__, "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--eval-samples-per-prompt" in result.stdout and "--critic-context-mode" in result.stdout
    assert launcher.ScriptArgs().dataset in result.stdout


def test_cli_eval_sample_override_is_consumed(launcher, monkeypatch):
    backend = _recording_backend(monkeypatch, launcher)
    app = typer.Typer(add_completion=False)
    app.command()(launcher.main)
    result = CliRunner().invoke(app, ["--action", "train", "--eval-samples-per-prompt", "2"])
    assert result.exit_code == 0, result.output
    argv = shlex.split(backend.training[0]["train_args"])
    assert argv[argv.index("--n-samples-per-eval-prompt") + 1] == "2"


def test_episode_batch_divisibility_allows_prompt_groups_to_span_dp_ranks(launcher):
    args = launcher.ScriptArgs(
        num_nodes=8,
        trainer_num_nodes=4,
        rollout_batch_size=12,
        samples_per_prompt=16,
        critic_steps_per_rollout=4,
    )
    _recipe._validate(args)
    assert args.trainer_num_nodes * args.num_gpus_per_node // args.tensor_model_parallel_size == 8
    assert args.rollout_batch_size * args.samples_per_prompt == 192
    assert 192 // args.critic_steps_per_rollout == 48


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"rollout_batch_size": 3, "samples_per_prompt": 3, "critic_steps_per_rollout": 1}, "episode batch"),
        ({"rollout_batch_size": 4, "samples_per_prompt": 3, "critic_steps_per_rollout": 4}, "equal episode counts"),
    ],
)
def test_nondivisible_episode_batches_still_fail(launcher, overrides, message):
    with pytest.raises(ValueError, match=message):
        _recipe._validate(launcher.ScriptArgs(**overrides))
