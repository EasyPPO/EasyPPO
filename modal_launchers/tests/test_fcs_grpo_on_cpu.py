"""FCS launch contracts; all remote allocation is mocked or forbidden."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir

from modal_launchers import fcs_grpo
from modal_launchers.config import load_modal_config
from modal_launchers.multinode import commit_all_nodes, prepare_output

ROOT = Path(__file__).resolve().parents[2]


def test_default_only_prints_plan(monkeypatch, capsys):
    launch = Mock(side_effect=AssertionError("No remote resources allowed"))
    monkeypatch.setattr(fcs_grpo, "launch", launch)
    fcs_grpo.main(["--run-name", "cpu-plan"])
    launch.assert_not_called()
    output = capsys.readouterr().out
    assert '"total_gpus": 16' in output and '"launch": false' in output


@pytest.mark.parametrize("recipe", fcs_grpo.RECIPES)
@pytest.mark.parametrize("nodes", [1, 2, 4])
def test_real_recipe_composes_without_aime_overrides(recipe, nodes):
    settings = load_modal_config()
    command = fcs_grpo.training_command(
        settings,
        recipe=recipe,
        model_path="/models/Qwen (FCS)",
        run_name="fcs-test",
        judge_url="https://judge.example",
        nodes=nodes,
    )
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=recipe, overrides=command[7:])
    assert cfg.trainer.nnodes == nodes and cfg.trainer.n_gpus_per_node == 8
    assert cfg.trainer.save_freq == 5 and cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.trainer.total_training_steps == 240
    assert cfg.algorithm.adv_estimator == "grpo" and not cfg.critic.enable
    assert cfg.actor_rollout_ref.model.path == "/models/Qwen (FCS)"
    assert cfg.actor_rollout_ref.model.override_config.attn_implementation == "flash_attention_2"
    assert not cfg.actor_rollout_ref.actor.fsdp_config.param_offload
    assert not cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload
    assert cfg.actor_rollout_ref.ref.fsdp_config.param_offload
    assert cfg.reward.custom_reward_function.reward_kwargs.judge_url == "https://judge.example"
    assert "frontiercs" in cfg.data.train_files[0] and "frontiercs" in cfg.data.val_files[0]
    assert cfg.data.max_response_length == 32768 and cfg.actor_rollout_ref.rollout.n == 32
    assert cfg.ray_kwargs.ray_init.address == "auto"


def test_two_node_checkpoint_commits_worker_before_head():
    values = {}
    events = []

    def check(keys):
        events.append("worker")
        assert keys == ["committed/1/step-5"]
        return True

    store = SimpleNamespace(set=values.__setitem__, check=check)
    commit_all_nodes(store, Mock(), lambda: events.append("head"), "step-5", nodes=2)
    assert values == {"commit": "step-5"}
    assert events == ["worker", "head"]


def test_resume_rejects_changed_recipe_or_world_size(tmp_path):
    metadata = {"recipe": fcs_grpo.RECIPES[0], "model": fcs_grpo.MODEL_ID}
    prepare_output(tmp_path, "run", "FCS", None, Mock(), nodes=2, run_metadata=metadata)
    manifest = json.loads((tmp_path / "modal_cluster_run.json").read_text())
    assert manifest["world_size"] == 16 and manifest["training"] == metadata
    for nodes, recipe in [(4, fcs_grpo.RECIPES[0]), (2, fcs_grpo.RECIPES[-1])]:
        with pytest.raises(ValueError, match="does not match"):
            prepare_output(
                tmp_path,
                "run",
                "FCS",
                None,
                Mock(),
                nodes=nodes,
                run_metadata={**metadata, "recipe": recipe},
            )


@pytest.mark.parametrize("judge_ok", [True, False])
def test_judge_preflight_gates_gpu_and_cleanup_runs(monkeypatch, judge_ok):
    modal = pytest.importorskip("modal")
    events = []
    app = SimpleNamespace(run=lambda: nullcontext())
    prepare = SimpleNamespace(remote=lambda: "/volume/model")
    train = SimpleNamespace(remote=Mock(side_effect=lambda *a: events.append("gpu")))
    image = object()
    judge = SimpleNamespace(
        wait_until_ready=Mock(),
        create_connect_token=Mock(return_value=SimpleNamespace(url="https://judge.example", token="test-token")),
        terminate=Mock(side_effect=lambda: events.append("stop-judge")),
    )
    monkeypatch.setattr(fcs_grpo, "build_app", lambda *a: (app, prepare, train, image))
    create = Mock(return_value=judge)
    monkeypatch.setattr(modal.Sandbox, "create", create)

    def check(url, token):
        assert url == "https://judge.example" and token == "test-token"
        events.append("check-judge")
        if not judge_ok:
            raise RuntimeError("bad judge")
        return {"score": 0.5}

    monkeypatch.setattr(fcs_grpo, "check_judge", check)
    if judge_ok:
        fcs_grpo.launch(fcs_grpo.parse_args([]), load_modal_config())
        assert events == ["check-judge", "gpu", "stop-judge"]
    else:
        with pytest.raises(RuntimeError, match="bad judge"):
            fcs_grpo.launch(fcs_grpo.parse_args([]), load_modal_config())
        train.remote.assert_not_called()
        assert events == ["check-judge", "stop-judge"]
    assert create.call_args.kwargs["runtime"] == "vm"
    assert "gpu" not in create.call_args.kwargs


def test_wandb_credentials_checked_before_allocation(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    build = Mock(side_effect=AssertionError("must fail before defining app"))
    monkeypatch.setattr(fcs_grpo, "build_app", build)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        fcs_grpo.launch(fcs_grpo.parse_args(["--wandb"]), load_modal_config())
    build.assert_not_called()
