"""Configuration propagation checks; no Modal allocations or training jobs."""

import importlib.util
import socket
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from modal_launchers.config import DEFAULT_CONFIG, REMOTE_CONFIG_ENV, ModalConfig, load_modal_config
from modal_launchers.multinode import node_environment, training_command
from scripts.check_easyppo_config import compose_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def custom_config(tmp_path, monkeypatch):
    monkeypatch.delenv(REMOTE_CONFIG_ENV, raising=False)
    values = yaml.safe_load(DEFAULT_CONFIG.read_text())
    values.update(
        image="example.com/public/easyppo:test",
        source_root=str(tmp_path / "source checkout"),
        volume_name="custom-volume",
        volume_mount=str(tmp_path / "volume mount"),
        model_dir="models/base (test)",
        output_dir="runs,(test)",
        hf_cache_dir="hf-cache",
        vllm_cache_dir="vllm-cache",
    )
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(values))
    monkeypatch.setenv("EASYPPO_MODAL_CONFIG", str(path))
    return ModalConfig(**values)


def test_external_config_is_frozen_for_remote_workers(custom_config, monkeypatch):
    assert load_modal_config() == custom_config
    env = custom_config.container_env()
    assert env["HF_HOME"] == custom_config.volume_path("hf-cache")
    assert env["VLLM_CACHE_ROOT"] == custom_config.volume_path("vllm-cache")
    monkeypatch.setenv(REMOTE_CONFIG_ENV, env[REMOTE_CONFIG_ENV])
    # The local override file need not exist inside a container.
    monkeypatch.setenv("EASYPPO_MODAL_CONFIG", "/missing/local-config.yaml")
    assert load_modal_config() == custom_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_root", "relative"),
        ("volume_mount", "/"),
        ("model_dir", "/outside-volume"),
        ("output_dir", "../outside-volume"),
        ("hf_cache_dir", ""),
        ("vllm_cache_dir", "."),
    ],
)
def test_invalid_paths_fail_before_launch(custom_config, field, value):
    values = asdict(custom_config)
    values[field] = value
    with pytest.raises(ValueError, match=field):
        ModalConfig(**values)


@pytest.mark.parametrize("nested", [False, True])
def test_source_upload_cannot_overlap_volume(custom_config, nested):
    values = asdict(custom_config)
    values["source_root"] = custom_config.volume_mount + ("/source" if nested else "")
    with pytest.raises(ValueError, match="overlap"):
        ModalConfig(**values)


def import_launcher(name, monkeypatch):
    modal = pytest.importorskip("modal")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # A leftover partial registry credential must not affect a public pull.
    monkeypatch.setenv("REGISTRY_USERNAME", "unused")
    monkeypatch.delenv("REGISTRY_PASSWORD", raising=False)
    registry = Mock(wraps=modal.Image.from_registry)
    volumes = Mock(wraps=modal.Volume.from_name)
    secrets = Mock(side_effect=AssertionError("No registry credentials should be read"))
    monkeypatch.setattr(modal.Image, "from_registry", registry)
    monkeypatch.setattr(modal.Volume, "from_name", volumes)
    monkeypatch.setattr(modal.Secret, "from_local_environ", secrets)
    spec = importlib.util.spec_from_file_location(name, ROOT / "modal_launchers" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry.assert_called_once_with(module.CONFIG.image)
    volumes.assert_called_once_with(module.CONFIG.volume_name)
    secrets.assert_not_called()
    return module


def test_eight_gpu_training_uses_custom_paths(custom_config, monkeypatch):
    import subprocess

    module = import_launcher("b200", monkeypatch)
    assert module.app.name == "easyppo-aime"
    model = Path(custom_config.model_path)
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    process = Mock()
    monkeypatch.setattr(subprocess, "run", process)
    module.volume = SimpleNamespace(commit=Mock())
    with pytest.warns(UserWarning, match="executing locally"):
        module.train.local(run_name="test-run")
    args, kwargs = process.call_args
    config = compose_config(args[0][2:])
    assert config.actor_rollout_ref.model.path == str(model)
    assert config.trainer.default_local_dir == custom_config.output_path("test-run") + "/checkpoints"
    assert config.trainer.rollout_data_dir == custom_config.output_path("test-run") + "/rollouts"
    assert config.trainer.validation_data_dir == custom_config.output_path("test-run") + "/validation"
    assert config.trainer.nnodes == 1
    assert kwargs["cwd"] == custom_config.source_root
    assert kwargs["env"]["PYTHONPATH"] == custom_config.source_root
    module.volume.commit.assert_called_once()


def test_thirty_two_gpu_training_forwards_config(custom_config, monkeypatch):
    import modal.experimental

    from modal_launchers import multinode

    module = import_launcher("b200_32gpu", monkeypatch)
    assert module.app.name == "easyppo-aime-32gpu"
    run_node = Mock()
    monkeypatch.setattr(multinode, "run_node", run_node)
    cluster = object()
    monkeypatch.setattr(modal.experimental, "get_cluster_info", lambda: cluster)
    module.volume = SimpleNamespace(commit=Mock())
    with pytest.warns(UserWarning, match="executing locally"):
        module.train.local(run_name="test-run")
    run_node.assert_called_once_with(
        cluster,
        run_name="test-run",
        model_path=custom_config.model_path,
        commit=module.volume.commit,
        config=custom_config,
    )
    command = training_command(custom_config.model_path, "test-run", "project", custom_config)
    config = compose_config(command[2:])
    assert config.actor_rollout_ref.model.path == custom_config.model_path
    assert config.trainer.default_local_dir == custom_config.output_path("test-run") + "/checkpoints"
    assert config.trainer.rollout_data_dir == custom_config.output_path("test-run") + "/rollouts"
    assert config.trainer.validation_data_dir == custom_config.output_path("test-run") + "/validation"
    assert config.trainer.nnodes == 4
    assert config.trainer.total_training_steps == 150


def test_ray_workers_use_custom_source_and_caches(custom_config, monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "eth1": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.2")],
        },
    )
    env = node_environment("10.0.0.2", "10.0.0.1", "run", custom_config)
    assert env["PYTHONPATH"] == custom_config.source_root
    assert env["HF_HOME"] == custom_config.volume_path("hf-cache")
    assert env["VLLM_CACHE_ROOT"] == custom_config.volume_path("vllm-cache")


@pytest.mark.parametrize("name", ["b200", "b200_32gpu"])
def test_timeout_restart_keeps_configured_model_path(custom_config, monkeypatch, name):
    modal = pytest.importorskip("modal")
    module = import_launcher(name, monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_RUN_ID", "test-run")
    monkeypatch.setenv("WANDB_RESUME", "must")
    remote = Mock(side_effect=[modal.exception.FunctionTimeoutError("timeout"), None])
    module.train = SimpleNamespace(remote=remote)
    module.resume(run_name="test-run", max_timeout_restarts=1)
    assert remote.call_count == 2
    for call in remote.call_args_list:
        assert call.kwargs == {"run_name": "test-run", "model_path": custom_config.model_path}
