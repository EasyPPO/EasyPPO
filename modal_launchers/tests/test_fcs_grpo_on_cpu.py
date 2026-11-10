"""FCS launch contracts; all remote allocation is mocked or forbidden."""

import json
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
    monkeypatch.setattr(fcs_grpo.modal.Sandbox, "create", launch)
    monkeypatch.setattr(fcs_grpo.train, "remote", launch)
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
    assert list(cfg.trainer.logger) == ["console", "file", "wandb"]
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


@pytest.mark.parametrize("outcome", ["success", "bad-judge", "timeout", "training-error", "cancelled"])
def test_remote_coordinator_preflight_and_cleanup(monkeypatch, outcome):
    modal = pytest.importorskip("modal")
    events = []
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(fcs_grpo, "check_model", Mock())
    error = {
        "timeout": modal.exception.FunctionTimeoutError("timeout"),
        "training-error": RuntimeError("training-error"),
        "cancelled": KeyboardInterrupt("cancelled"),
    }.get(outcome)

    def result():
        events.append("wait-gpu")
        if error is not None:
            raise error

    call = SimpleNamespace(get=result, cancel=Mock(side_effect=lambda **kw: events.append("cancel-gpu")))
    cluster = SimpleNamespace(spawn=Mock(return_value=call))
    judge = SimpleNamespace(
        wait_until_ready=Mock(),
        create_connect_token=Mock(return_value=SimpleNamespace(url="https://judge.example", token="test-token")),
        terminate=Mock(side_effect=lambda: events.append("stop-judge")),
    )
    create = Mock(return_value=judge)
    monkeypatch.setattr(modal.Sandbox, "create", create)

    def check(url, token):
        assert url == "https://judge.example" and token == "test-token"
        events.append("check-judge")
        if outcome == "bad-judge":
            raise RuntimeError("bad-judge")
        return {"score": 0.5}

    monkeypatch.setattr(fcs_grpo, "check_judge", check)
    kwargs = dict(run_name="test-run", model_path="/volume/model", recipe=fcs_grpo.RECIPES[0])
    image = object()
    if outcome == "success":
        fcs_grpo.run_training(cluster, image, fcs_grpo.parse_args([]), **kwargs)
        assert events == ["check-judge", "wait-gpu", "stop-judge"]
        call.cancel.assert_not_called()
    else:
        with pytest.raises(type(error) if error else RuntimeError, match=outcome):
            fcs_grpo.run_training(cluster, image, fcs_grpo.parse_args([]), **kwargs)
        if outcome == "bad-judge":
            cluster.spawn.assert_not_called()
            assert events == ["check-judge", "stop-judge"]
        else:
            call.cancel.assert_called_once_with(terminate_containers=True)
            assert events == ["check-judge", "wait-gpu", "cancel-gpu", "stop-judge"]
    assert create.call_args.kwargs["runtime"] == "vm"
    assert "gpu" not in create.call_args.kwargs
    if outcome != "bad-judge":
        cluster.spawn.assert_called_once_with(
            "test-run", "/volume/model", fcs_grpo.RECIPES[0], "https://judge.example", "test-token"
        )


def test_missing_wandb_fails_before_judge_or_gpu(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    create = Mock(side_effect=AssertionError("No allocations allowed"))
    monkeypatch.setattr(fcs_grpo.modal.Sandbox, "create", create)
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        fcs_grpo.run_training(
            None,
            None,
            fcs_grpo.parse_args([]),
            run_name="test",
            model_path="/missing",
            recipe=fcs_grpo.RECIPES[0],
        )
    create.assert_not_called()
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        fcs_grpo.resume(run_name="test")


@pytest.mark.parametrize("outcome", ["success", "timeout", "error", "exhausted"])
def test_resume_only_retries_timeouts_with_same_identity(monkeypatch, outcome):
    modal = pytest.importorskip("modal")
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    timeout = modal.exception.FunctionTimeoutError("timeout")
    effects = {
        "success": [None],
        "timeout": [timeout, None],
        "error": [RuntimeError("training failed")],
        "exhausted": [timeout, timeout],
    }
    remote = Mock(side_effect=effects[outcome])
    monkeypatch.setattr(fcs_grpo.train, "remote", remote)
    kwargs = dict(run_name="same-run", model_path="/cached/sft", recipe=fcs_grpo.RECIPES[0])
    if outcome in ("success", "timeout"):
        fcs_grpo.resume(**kwargs, max_timeout_restarts=1)
    else:
        with pytest.raises(modal.exception.FunctionTimeoutError if outcome == "exhausted" else RuntimeError):
            fcs_grpo.resume(**kwargs, max_timeout_restarts=1)
    assert remote.call_count == len(effects[outcome])
    assert all(call.kwargs == kwargs for call in remote.call_args_list)


def test_default_model_uses_fcs_cache_and_custom_volume_paths():
    from dataclasses import replace

    config = replace(load_modal_config(), volume_mount="/custom", hf_cache_dir="hf-cache")
    path = fcs_grpo.default_model_path(config)
    assert path.startswith("/custom/hf-cache/hub/models--HuanzhiMao--")
    assert path.endswith("/snapshots/" + fcs_grpo.MODEL_REVISION)
    assert path != config.model_path


@pytest.mark.parametrize("sharded", [False, True])
def test_cached_model_preflight_detects_broken_weights(tmp_path, sharded):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}")
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "shard1.safetensors", "b": "shard2.safetensors"}})
        )
        (tmp_path / "shard1.safetensors").write_bytes(b"test")
        path = tmp_path / "shard2.safetensors"
    else:
        path = tmp_path / "model.safetensors"
    path.symlink_to(tmp_path / "missing-blob")
    with pytest.raises(FileNotFoundError, match="Missing model weights"):
        fcs_grpo.check_model(str(tmp_path))
    (tmp_path / "missing-blob").write_bytes(b"test")
    fcs_grpo.check_model(str(tmp_path))


def test_remote_helper_import_does_not_need_local_build_context(tmp_path, monkeypatch):
    import importlib.util

    source = tmp_path / "fcs_remote.py"
    source.write_text((ROOT / "modal_launchers/fcs_grpo.py").read_text())
    monkeypatch.setattr(fcs_grpo.modal, "is_local", lambda: False)
    create = Mock(side_effect=AssertionError("Remote import must not rebuild images"))
    monkeypatch.setattr(fcs_grpo.modal.Image, "from_dockerfile", create)
    spec = importlib.util.spec_from_file_location("fcs_remote", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.run_training)
    create.assert_not_called()
