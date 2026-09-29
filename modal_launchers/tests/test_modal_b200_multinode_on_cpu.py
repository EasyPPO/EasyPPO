# Copyright 2026 EasyPPO contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Local checks only: no Modal allocation, Ray server, or training subprocess."""

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_launchers.multinode import (
    check_health,
    commit_all_nodes,
    node_environment,
    prepare_output,
    ray_command,
    training_command,
    validate_run_name,
    wait_for,
    worker_loop,
)
from scripts.check_easyppo_config import compose_config


class MemoryStore:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode()

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)


def test_real_hydra_config_keeps_recipe_and_uses_32_gpus_150_steps():
    command = training_command("/models/qwen", "fresh-run", "AIME project")
    config = compose_config(command[2:])
    assert config.trainer.nnodes == 4
    assert config.trainer.n_gpus_per_node == 8
    assert config.trainer.total_training_steps == 150
    assert config.trainer.project_name == "AIME project"
    assert config.data.train_batch_size == 32
    assert config.actor_rollout_ref.rollout.n == 16
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
    assert config.critic.ppo_mini_batch_size == 8
    assert config.critic.fsdp.fsdp_size == 8
    assert config.trainer.save_freq == 10
    assert config.trainer.test_freq == 5
    assert config.ray_kwargs.ray_init.address == "auto"
    assert config.trainer.resume_mode == "auto"


def test_head_and_workers_join_one_ray_cluster():
    for rank in range(4):
        command = ray_command(rank, f"10.0.0.{rank + 1}", "10.0.0.1")
        assert "--num-gpus=8" in command
        assert "--block" in command
        assert ("--head" in command) == (rank == 0)
        if rank:
            assert "--address=10.0.0.1:6379" in command


def test_wandb_id_and_network_are_node_specific(monkeypatch):
    import psutil

    monkeypatch.setenv("WANDB_RUN_ID", "old-eight-gpu-run")
    monkeypatch.setenv("WANDB_RESUME", "must")
    # Regression: Modal can supply an IPv6-only NCCL family while the launcher
    # selects a private IPv4 interface. That combination finds no interface.
    monkeypatch.setenv("NCCL_SOCKET_FAMILY", "AF_INET6")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
        "eth1": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.2")],
    })
    env = node_environment("10.0.0.2", "10.0.0.1", "new-32gpu")
    assert env["WANDB_RUN_ID"] == "new-32gpu"
    assert env["WANDB_RESUME"] == "allow"
    assert env["VLLM_HOST_IP"] == "10.0.0.2"
    assert env["RAY_ADDRESS"] == "10.0.0.1:6379"
    assert env["NCCL_SOCKET_IFNAME"] == "=eth1"
    assert env["NCCL_SOCKET_FAMILY"] == "AF_INET"
    assert env["GLOO_SOCKET_IFNAME"] == "eth1"


def test_new_output_and_resume_require_same_run_identity(tmp_path):
    output = tmp_path / "run"
    commit = Mock()
    prepare_output(output, "new", "AIME", "team", commit)
    commit.assert_called_once()
    prepare_output(output, "new", "AIME", "team", commit)
    commit.assert_called_once()
    with pytest.raises(ValueError, match="does not match"):
        prepare_output(output, "new", "different-project", "team", commit)


def test_existing_8gpu_directory_is_not_reused(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    commit = Mock()
    with pytest.raises(ValueError, match="Refusing to reuse"):
        prepare_output(tmp_path, "old", "AIME", "team", commit)
    commit.assert_not_called()


@pytest.mark.parametrize("name", ["../old-run", "a/b", "a b", "", "x" * 129])
def test_invalid_run_names(name):
    with pytest.raises(ValueError):
        validate_run_name(name)


def test_worker_failure_reaches_head():
    store = MemoryStore()
    store.set("error/2", "Ray startup failed")
    with pytest.raises(RuntimeError, match="Node 2 failed"):
        check_health(store, Mock(poll=lambda: None))


def test_setup_timeout_is_not_modal_function_timeout():
    with pytest.raises(RuntimeError, match="Timed out waiting for nodes"):
        wait_for(MemoryStore(), lambda: False, Mock(poll=lambda: None), "nodes", timeout=0)


def test_head_commit_waits_for_worker_commits(monkeypatch):
    store = MemoryStore()
    events = []

    def workers_commit(_):
        assert store.get("commit") == b"step-10"
        for rank in range(1, 4):
            events.append(rank)
            store.set(f"committed/{rank}/step-10", "1")

    monkeypatch.setattr("modal_launchers.multinode.time.sleep", workers_commit)
    commit_all_nodes(store, Mock(poll=lambda: None), lambda: events.append(0), "step-10")
    assert events == [1, 2, 3, 0]


def test_worker_commits_before_acknowledging_and_stops_on_completion(monkeypatch):
    store = MemoryStore()
    store.set("commit", "step-10")
    commit = Mock()
    monkeypatch.setattr("modal_launchers.multinode.time.sleep", lambda _: None)
    worker = worker_loop(store, Mock(poll=lambda: None), commit)
    assert next(worker) == "step-10"
    commit.assert_called_once()
    store.set("finished", "1")
    with pytest.raises(StopIteration):
        next(worker)
