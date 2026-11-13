"""CPU-only checks for topology, platform allocation and checkpoint barriers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import pytest

from modal_launchers import frontiersmith, multinode
from modal_launchers.config import Config, check_manifest, parse_args


@pytest.mark.parametrize("nodes", [2, 4])
@pytest.mark.parametrize("gpu", ["H200", "B200"])
def test_requested_topology_reaches_miles_and_modal(nodes, gpu, monkeypatch):
    _, config = parse_args(["--nodes", str(nodes), "--gpu", gpu])
    command = config.command()
    assert command[command.index("--num-nodes") + 1] == str(nodes)
    assert command[command.index("--num-gpus-per-node") + 1] == "8"
    functions, clusters = [], []

    class App:
        def __init__(self, name):
            pass

        def function(self, **options):
            def decorate(fn):
                functions.append((fn.__name__, options))
                return fn

            return decorate

    def clustered(**options):
        clusters.append(options)
        return lambda fn: fn

    modal = SimpleNamespace(App=App, Volume=SimpleNamespace(from_name=lambda *a, **kw: object()), clustered=clustered)
    monkeypatch.setattr(frontiersmith, "_training_image", lambda *a: object())
    monkeypatch.setattr(frontiersmith, "_secrets", lambda *a: [])
    frontiersmith._build_training(modal, config)
    settings = dict(functions)
    assert settings["training"]["gpu"] == f"{gpu}:8"
    assert settings["training"]["max_containers"] == nodes
    assert settings["training"]["retries"] == 0
    assert settings["preflight"]["max_containers"] == 1
    assert "gpu" not in settings["preflight"]
    assert clusters == [{"size": nodes, "rdma": True}]


@pytest.mark.parametrize("nodes,gpus", [(2, 4), (4, 4), (3, 8)])
def test_invalid_cluster_rejected_before_launch(nodes, gpus):
    with pytest.raises(ValueError):
        Config(mode="full_detach", run_id="test", nodes=nodes, gpus=gpus)


def test_conversion_stays_on_one_node(capsys):
    frontiersmith.main(["--nodes", "4", "--gpu", "B200", "--action", "prepare"])
    import json

    plan = json.loads(capsys.readouterr().out)
    assert plan["nodes"] == 1 and plan["gpu_count"] == 8
    command = plan["command"]
    assert command[command.index("--num-nodes") + 1] == "1"


@pytest.mark.parametrize("nodes", [2, 4])
def test_training_preview_reports_total_gpus(nodes, capsys):
    frontiersmith.main(["--nodes", str(nodes)])
    import json

    plan = json.loads(capsys.readouterr().out)
    assert plan["nodes"] == nodes and plan["gpu_count"] == nodes * 8
    assert plan["launch"] is False


def test_resume_rejects_topology_and_gpu_changes(tmp_path):
    config = Config(mode="full_detach", run_id="test", mount=str(tmp_path), nodes=2)
    check_manifest(config, wandb_project="test", wandb_entity=None)
    for changed in (replace(config, nodes=4), replace(config, gpu="B200")):
        with pytest.raises(ValueError, match="configuration differs"):
            check_manifest(changed, wandb_project="test", wandb_entity=None)


class Store:
    def __init__(self):
        self.data = {}
        self.lock = Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = str(value).encode()

    def get(self, key):
        with self.lock:
            return self.data[key]

    def check(self, keys):
        with self.lock:
            return all(key in self.data for key in keys)


@pytest.mark.parametrize("nodes", [2, 4])
def test_every_node_commits_before_reload_and_return(nodes):
    store, events = Store(), []

    def volume(rank):
        return SimpleNamespace(
            commit=lambda: events.append((rank, "commit")), reload=lambda: events.append((rank, "reload"))
        )

    with ThreadPoolExecutor(max_workers=nodes - 1) as pool:
        workers = [
            pool.submit(multinode._worker_loop, store, None, volume(rank), rank=rank, nodes=nodes)
            for rank in range(1, nodes)
        ]
        try:
            multinode._commit_all(store, volume(0), nodes, "first")
            assert len(events) == 2 * nodes
            assert all(kind == "commit" for _, kind in events[:nodes])
            assert all(kind == "reload" for _, kind in events[nodes:])
            multinode._commit_all(store, volume(0), nodes, "second")
            assert len(events) == 4 * nodes
        finally:
            store.set("finished", "1")
        for future in workers:
            future.result(timeout=5)


def test_worker_failure_prevents_checkpoint_publication():
    store = Store()
    store.set("error/1", "injected commit failure")
    volume = SimpleNamespace(commit=lambda: None, reload=lambda: pytest.fail("must not reload"))
    with pytest.raises(RuntimeError, match="injected commit failure"):
        multinode._commit_all(store, volume, 2, "failed")


def test_ray_head_has_jobs_endpoint_and_worker_joins_it():
    config = Config(mode="full_detach", run_id="test", nodes=2)
    ips = ["10.0.0.1", "10.0.0.2"]
    head = multinode._ray_command(config, rank=0, addresses=ips)
    worker = multinode._ray_command(config, rank=1, addresses=ips)
    assert "--head" in head and "--dashboard-port=8265" in head
    assert "--address=10.0.0.1:6379" in worker and "--head" not in worker
    assert "--node-ip-address=10.0.0.2" in worker
    assert "--num-gpus=8" in head and "--num-gpus=8" in worker


def test_old_single_node_manifest_can_still_resume(tmp_path):
    import json
    from pathlib import Path

    config = Config(mode="full_detach", run_id="legacy", mount=str(tmp_path))
    check_manifest(config, wandb_project="test", wandb_entity=None)
    path = Path(config.output, "modal_run.json")
    old = json.loads(path.read_text())
    del old["nodes"]
    path.write_text(json.dumps(old))
    check_manifest(config, wandb_project="test", wandb_entity=None)
    with pytest.raises(ValueError, match="configuration differs"):
        check_manifest(replace(config, nodes=2), wandb_project="test", wandb_entity=None)


def test_node_environment_binds_private_interface_and_preserves_rdma(monkeypatch):
    import psutil
    import socket

    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_0")
    monkeypatch.setattr(
        psutil,
        "net_if_addrs",
        lambda: {
            "eth0": [SimpleNamespace(family=socket.AF_INET, address="192.168.0.1")],
            "eth1": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.2")],
        },
    )
    config = Config(mode="full_detach", run_id="test", nodes=2)
    env = multinode._node_environment(config, rank=1, addresses=["10.0.0.1", "10.0.0.2"], judge_token="test")
    assert env["NCCL_SOCKET_IFNAME"] == "=eth1" and env["GLOO_SOCKET_IFNAME"] == "eth1"
    assert env["NCCL_IB_HCA"] == "mlx5_0"
    assert env["RAY_ADDRESS"] == "10.0.0.1:6379" and env["MASTER_ADDR"] == "10.0.0.1"
    assert env["EASYPPO_MODAL_NODES"] == "2" and env["MILES_SCRIPT_EXTERNAL_RAY"] == "1"
