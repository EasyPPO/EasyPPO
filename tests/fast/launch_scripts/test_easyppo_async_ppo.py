"""Verify the generated async recipe and reject invalid resource budgets offline."""

import base64
import json
import shlex
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.fixture
def launcher():
    return import_launch_script(REPO_ROOT / "examples/easyppo/run_qwen3_5_27b_async_ppo.py")


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("dataset", ["frontiersmith", "tmax"])
@pytest.mark.parametrize("centering", [False, True])
def test_one_step_async_recipe(launcher, monkeypatch, tmp_path, mode, dataset, centering):
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
        dataset=dataset,
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
    assert value("--update-weights-interval") == "1"
    assert "--offload-train" in argv
    assert value("--prompt-data") == args.train_data
    eval_config = json.loads(base64.b64decode(value("--eval-config").removeprefix("base64:")))
    eval_dataset = eval_config["datasets"][0]
    assert eval_dataset["path"] == args.validation_data
    assert eval_dataset["name"] == ("fcs_quick" if dataset == "frontiersmith" else "tb21_quick")
    assert eval_dataset["max_samples"] == 16 and eval_dataset["sample_seed"] == 42
    assert value("--eval-temperature") == "1.0" and value("--eval-top-p") == "1.0"
    assert value("--eval-top-k") == "-1" and value("--n-samples-per-eval-prompt") == "1"
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
        launcher._validate(launcher.ScriptArgs(**overrides))


@pytest.mark.parametrize("nodes,train_nodes,rollout", [(8, 2, 48), (4, 1, 24)])
def test_train_rollout_resource_options(launcher, nodes, train_nodes, rollout):
    args = launcher.ScriptArgs(num_nodes=nodes, trainer_num_nodes=train_nodes)
    launcher._validate(args)
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
    launcher._validate(args)
    argv = launcher._algorithm_args(args)
    assert argv[argv.index("--score-centering-is") + 1] == mode
    if mode == "tis":
        assert argv[argv.index("--score-centering-tis-clip") + 1] == "1.5"
    elif mode == "mis":
        assert argv[argv.index("--score-centering-mis-low") + 1] == "0.7"
        assert argv[argv.index("--score-centering-mis-high") + 1] == "3.0"
    assert "--use-tis" not in argv


@pytest.mark.parametrize(
    "dataset,script",
    [
        ("frontiersmith", "make_miles_frontiersmith_data.py"),
        ("tmax", "make_miles_terminal_data.py"),
    ],
)
def test_prepare_selects_the_data_recipe(launcher, monkeypatch, tmp_path, dataset, script):
    calls = []

    class Backend:
        def exec_command_cpu(self, cmd):
            calls.append(shlex.split(cmd))

        def convert_checkpoint(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(launcher.ScriptArgs, "create_backend", lambda self: Backend())
    launcher.prepare(
        launcher.ScriptArgs(
            dataset=dataset,
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
        )
    )
    assert calls[1][:2] == ["hf", "download"]
    assert "--revision" in calls[1]
    assert calls[2][1].endswith(script)
    if dataset == "tmax":
        assert "tmax" in calls[2] and "--tokenizer" in calls[2]
    assert calls[3]["megatron_model_type"] == "qwen3.5-27B"


def test_critic_warmstart_explicitly_resets_training_state(launcher):
    args = launcher.ScriptArgs(critic_load="/shared/warm-critic", critic_weights_only=True)
    launcher._validate(args)
    argv = launcher._checkpoint_args(args)
    assert argv[argv.index("--critic-load") + 1] == args.critic_load
    assert {"--finetune", "--no-load-optim", "--no-load-rng"} <= set(argv)
