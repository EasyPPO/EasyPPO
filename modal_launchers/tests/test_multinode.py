"""CPU-only checks for topology, platform allocation and checkpoint barriers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace

import pytest

from modal_launchers import frontiersmith, multinode
from modal_launchers.config import Config, check_manifest, parse_args


@pytest.mark.parametrize("nodes", [2, 4])
@pytest.mark.parametrize("gpu", ["H200", "B200"])
@pytest.mark.parametrize("environment", ["docker", "shared-judge"])
def test_requested_topology_reaches_miles_and_modal(nodes, gpu, environment, monkeypatch):
    _, config = parse_args(["--nodes", str(nodes), "--gpu", gpu, "--terminal-environment", environment])
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
    _, _, _, training = frontiersmith._build_training(modal, config)
    daemon_starts = []
    from modal_launchers import docker_runtime

    def fake_daemon(path):
        daemon_starts.append(path)
        return nullcontext()

    modal.Cluster = SimpleNamespace(from_context=lambda: object())
    monkeypatch.setattr(docker_runtime, "docker_daemon", fake_daemon)
    monkeypatch.setattr(multinode, "run_node", lambda *args: None)
    training("http://judge", "token")
    settings = dict(functions)
    assert settings["training"]["gpu"] == f"{gpu}:8"
    assert settings["training"]["max_containers"] == nodes
    assert settings["training"]["retries"] == 0
    assert bool(daemon_starts) == (environment == "docker")
    assert settings["training"].get("experimental_options", {}) == (
        {"enable_docker": True} if environment == "docker" else {}
    )
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


def test_head_reload_failure_releases_workers_before_driver_cleanup(monkeypatch):
    monkeypatch.setattr(multinode, "STARTUP_TIMEOUT", 2)
    store = Store()
    def fail_reload():
        raise RuntimeError("injected head reload failure")
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(
            multinode._worker_loop, store, None, SimpleNamespace(commit=lambda: None, reload=lambda: None),
            rank=1, nodes=2,
        )
        with pytest.raises(RuntimeError, match="injected head reload failure"):
            multinode._commit_all(store, SimpleNamespace(commit=lambda: None, reload=fail_reload), 2, "failed")
        with pytest.raises(RuntimeError, match="Node 0 failed.*injected head reload failure"):
            worker.result(timeout=3)


def test_ray_head_has_jobs_endpoint_and_worker_joins_it():
    config = Config(mode="full_detach", run_id="test", nodes=2)
    ips = ["10.0.0.1", "10.0.0.2"]
    head = multinode._ray_command(config, rank=0, addresses=ips)
    worker = multinode._ray_command(config, rank=1, addresses=ips)
    assert "--head" in head and "--dashboard-port=8265" in head
    assert "--address=10.0.0.1:6379" in worker and "--head" not in worker
    assert "--node-ip-address=10.0.0.2" in worker
    assert "--num-gpus=8" in head and "--num-gpus=8" in worker


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("custom_train", [False, True])
def test_head_checks_topology_only_before_training(monkeypatch, tmp_path, exit_code, custom_train):
    import subprocess
    import sys
    from unittest.mock import Mock

    from modal_launchers import runtime

    addresses = ["10.0.0.1", "10.0.0.2"]
    ray = Mock()
    ray.nodes.side_effect = [[{"Alive": True, "NodeManagerAddress": ip, "Resources": {"GPU": 8}} for ip in addresses]]
    monkeypatch.setitem(sys.modules, "ray", ray)
    store = Store()
    store.set("finished/1", "1")
    commit = Mock()
    monkeypatch.setattr(multinode, "_commit_all", commit)

    def train(config, judge_url, judge_token, *, run_command):
        run_command([sys.executable, "-c", f"raise SystemExit({exit_code})"], check=True)

    monkeypatch.setattr(runtime, "train", train)
    config = SimpleNamespace(gpus=8, nodes=2, source=tmp_path)
    options = {"train_fn": train} if custom_train else {}
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            multinode._run_head(store, None, None, config, addresses, "", "", **options)
        assert error.value.returncode == exit_code
        commit.assert_not_called()
    else:
        multinode._run_head(store, None, None, config, addresses, "", "", **options)
        commit.assert_called_once()
    ray.nodes.assert_called_once()
    ray.shutdown.assert_called_once()


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


def test_terminal_volume_access_is_scoped_to_checkpoints_and_pins_coordinator(monkeypatch):
    import socket
    import sys
    import psutil
    from examples.easyppo import terminus_ppo
    from modal_launchers.terminal_config import TerminalConfig

    config = TerminalConfig(run_id="lease-test")
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: {
        "eth0": [SimpleNamespace(family=socket.AF_INET, address="10.0.0.1")]
    })
    env = multinode._node_environment(config, rank=0, addresses=["10.0.0.1", "10.0.0.2"], judge_token="test")
    assert env["MILES_VOLUME_IO_ROOT"] == config.output_mount
    assert not env["MILES_VOLUME_IO_LOCK"].startswith(config.output_mount)
    monkeypatch.setenv("MILES_VOLUME_IO_ROOT", env["MILES_VOLUME_IO_ROOT"])
    monkeypatch.setenv("MILES_VOLUME_IO_LOCK", env["MILES_VOLUME_IO_LOCK"])
    monkeypatch.setattr(sys, "argv", ["recipe", *config.command()[3:]])
    recorded = []
    monkeypatch.setattr(terminus_ppo.async_ppo, "_execute", lambda args, **kw: recorded.append(kw))
    terminus_ppo.main()
    assert "--pin-rollout-manager-to-head" in recorded[0]["extra_args"]


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_terminus_allocates_and_runs_same_recipe_on_64_cpu_nodes(monkeypatch, mode):
    import sys
    from examples.easyppo import terminus_ppo
    from modal_launchers import terminus

    functions, clusters, launches = [], [], []

    class App:
        def __init__(self, name):
            pass

        def function(self, **resources):
            functions.append(resources)
            return lambda fn: SimpleNamespace(remote=fn)

        def run(self):
            return nullcontext()

    fake = SimpleNamespace(
        App=App, enable_output=nullcontext,
        Volume=SimpleNamespace(from_name=lambda name, create_if_missing: name),
        Secret=SimpleNamespace(from_name=lambda name, required_keys: name),
        Cluster=SimpleNamespace(from_context=lambda: object()),
        clustered=lambda **kwargs: (clusters.append(kwargs) or (lambda fn: fn)),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.setattr(terminus, "_training_image", lambda modal, config: object())
    monkeypatch.setattr(terminus, "_secrets", lambda modal: [])
    monkeypatch.setattr(terminus, "run_node", lambda *a, **kw: launches.append((a, kw)))
    monkeypatch.setattr(sys, "argv", [
        "run", "--async-save", "--launch-modal", "--modal-volume", "data", "--modal-sandbox-secret", "harbor",
        "--model-dir", "/shared/easyppo/models", "--output-dir", "/shared/easyppo/runs",
        "--tasks-dir", "/shared/easyppo/tasks", "--prompt-data", "/shared/easyppo/train.jsonl",
        "--eval-data", "/shared/easyppo/eval.jsonl", "--run-id", "resume-me",
    ])
    terminus_ppo.main(mode)
    assert clusters == [{"size": 2, "rdma": True}]
    assert len(functions) == len(launches) == 1
    assert (functions[0]["cpu"], functions[0]["gpu"], functions[0]["secrets"]) == (64, "B200:8", ["harbor"])
    (_, config, volume, _, _), options = launches[0]
    assert volume == "data" and config.output == "/shared/easyppo/runs/resume-me"
    assert options["cpus"] == 64
    for rank in (0, 1):
        start = multinode._ray_command(config, rank=rank, addresses=["10.0.0.1", "10.0.0.2"], cpus=options["cpus"])
        assert "--num-cpus=64" in start
    command = options["train_command"]
    assert command[:3] == ["python3", "-m", "examples.easyppo.terminus_" + mode.replace("-", "_")]
    assert "--launch-modal" not in command and command.count("--run-id") == 1
    # The head invokes the normal recipe, including checkpoint and async PPO settings.
    monkeypatch.setenv("EASYPPO_MODAL_VOLUME", volume)
    monkeypatch.setattr(sys, "argv", ["run", *command[3:]])
    recorded = []
    monkeypatch.setattr(terminus_ppo.async_ppo, "_execute", lambda args, **kw: recorded.append((args, kw)))
    terminus_ppo.main(mode)
    args, kw = recorded[0]
    assert args.critic_context_mode == mode and args.run_id == "resume-me"
    assert {"--async-save", "--use-persistent-ckpt-worker"} <= set(kw["extra_args"])
    assert kw["extra_args"][kw["extra_args"].index("--checkpoint-commit-hook") + 1] == "modal_launchers.runtime.commit_volume"


def test_terminal_workers_pin_and_reserve_each_node_before_dispatch(monkeypatch):
    from agentic_critic import terminal_workers

    requests = []
    nodes = [
        {"NodeID": node_id, "Alive": True, "Resources": {"GPU": 8, "CPU": 64}}
        for node_id in ["a" * 56, "b" * 56]
    ]
    monkeypatch.setattr(terminal_workers.ray, "nodes", lambda: nodes)
    monkeypatch.setattr(terminal_workers, "distributed_post_actors", lambda: ("shared-poster",))

    def options(*, scheduling_strategy, runtime_env):
        assert not scheduling_strategy.soft
        assert set(runtime_env["env_vars"].values()) == {"1"}
        def remote(args, *, evaluation, post_actors, archive_writer=None):
            assert post_actors == ("shared-poster",)
            assert archive_writer is None
            requests.append((scheduling_strategy.node_id, evaluation))
            return object()
        return SimpleNamespace(remote=remote)

    # Inspect Ray's class resource declaration as well as each placement request.
    assert terminal_workers.TerminalWorker._default_options["num_cpus"] == 1
    assert terminal_workers.TerminalWorker._default_options["num_gpus"] == 0
    monkeypatch.setattr(terminal_workers.TerminalWorker, "options", options)
    args = SimpleNamespace(terminal_train_workers_per_node=32, terminal_eval_workers_per_node=4, num_gpus_per_node=8)
    for evaluation, count in [(False, 32), (True, 4)]:
        pool = terminal_workers.TerminalWorkers(args, count=count, evaluation=evaluation)
        assert len(pool.workers) == 2 * count
        assert all(requests.count((node["NodeID"], evaluation)) == count for node in nodes)
    nodes[1]["Resources"]["CPU"] = 36
    with pytest.raises(ValueError, match="b" * 56):
        terminal_workers.TerminalWorkers(args, count=32, evaluation=False)
    assert len(requests) == 72


@pytest.mark.parametrize("action,expected", [
    ("prepare", ["download", "convert"]),
    ("validate", ["validate", "preflight"]),
    ("preflight", ["preflight"]),
    ("train", ["preflight", "spawn"]),
    ("all", ["download", "convert", "validate", "preflight", "spawn"]),
])
def test_terminal_submission_detaches_without_waiting_or_cancelling(monkeypatch, action, expected):
    from modal_launchers import terminal
    from modal_launchers.terminal_config import TerminalConfig

    events = []
    workload = SimpleNamespace(app_id="ap-test")

    def run(*, detach):
        assert detach is True
        return nullcontext()

    def phase(name):
        return SimpleNamespace(remote=lambda: events.append(name))

    def spawn():
        events.append("spawn")
        # No get/cancel methods: the submitting client must release ownership.
        return SimpleNamespace(object_id="fc-test")

    workload.run = run
    monkeypatch.setattr(terminal.modal, "enable_output", nullcontext)
    monkeypatch.setattr(terminal, "_build", lambda config: (
        workload, phase("download"), phase("convert"), phase("validate"),
        phase("preflight"), SimpleNamespace(spawn=spawn),
    ))
    result = terminal._launch(TerminalConfig(run_id="test"), action)
    assert events == expected
    if "spawn" in expected:
        assert result == {"app_id": "ap-test", "call_id": "fc-test", "run_id": "test"}
    else:
        assert result is None


def test_failed_terminal_preflight_never_submits_training(monkeypatch):
    from modal_launchers import terminal
    from modal_launchers.terminal_config import TerminalConfig

    def fail():
        raise RuntimeError("invalid prepared assets")

    workload = SimpleNamespace(run=lambda **kwargs: nullcontext())
    training = SimpleNamespace(spawn=lambda: pytest.fail("Training must not start after failed preflight"))
    monkeypatch.setattr(terminal.modal, "enable_output", nullcontext)
    monkeypatch.setattr(terminal, "_build", lambda config: (
        workload, None, None, None, SimpleNamespace(remote=fail), training,
    ))
    with pytest.raises(RuntimeError, match="invalid prepared assets"):
        terminal._launch(TerminalConfig(run_id="test"), "train")


def test_terminal_storage_uses_v2_and_default_gpu_preemption(monkeypatch):
    from modal_launchers import terminal
    from modal_launchers.terminal_config import TerminalConfig

    volumes, functions = [], []

    class App:
        def __init__(self, name):
            pass

        def function(self, **options):
            def decorate(fn):
                functions.append((fn.__name__, options))
                return fn
            return decorate

    def volume(name, **options):
        volumes.append((name, options))
        return name

    image = SimpleNamespace()
    image.pip_install = lambda *args: image
    image.env = lambda *args: image
    monkeypatch.setattr(terminal.modal, "App", App)
    monkeypatch.setattr(terminal.modal.Volume, "from_name", volume)
    monkeypatch.setattr(terminal.modal, "clustered", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(terminal, "_training_image", lambda *args: image)
    monkeypatch.setattr(terminal, "_secrets", lambda *args: [])
    config = TerminalConfig(run_id="fresh")
    terminal._build(config)
    assert volumes == [
        (config.assets_volume, {"create_if_missing": True, "version": 2}),
        (config.volume, {"create_if_missing": True, "version": 2}),
    ]
    assert config.assets_volume != "easyppo-miles"
    for name, options in functions:
        assert not options.get("nonpreemptible", False)
        if name == "training":
            assert options["gpu"] == "B200:8" and options["max_containers"] == 2
            assert options["retries"] == 0
    command = config.command()
    assert "--async-save" in command
    assert command[command.index("--save-interval") + 1] == "1"
    assert command[command.index("--max-keep") + 1] == "3"


@pytest.mark.parametrize("rank", [0, 1])
def test_terminal_gpu_exit_owns_cleanup_even_when_training_fails(monkeypatch, tmp_path, rank):
    from modal_launchers import terminal
    from modal_launchers.terminal_config import TerminalConfig

    events = []
    config = TerminalConfig(run_id="test", mount=str(tmp_path))
    volume = SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit"))
    cluster = SimpleNamespace(container_rank=lambda: rank)
    monkeypatch.setattr(terminal, "_cleanup_sandboxes", lambda config: events.append("cleanup"))

    def fail(*args, **kwargs):
        raise RuntimeError("training failure")

    with pytest.raises(RuntimeError, match="training failure"):
        terminal._run_training(config, cluster, volume, object(), run_node=fail, train_fn=None)
    assert events == (["reload", "cleanup", "commit"] if rank == 0 else ["reload", "commit"])


@pytest.mark.parametrize("failure", [None, "schema", "prompt"])
def test_terminal_preflight_validates_all_tasks_and_preserves_failures(monkeypatch, tmp_path, failure):
    import json
    from harbor.models.task import task as harbor_task
    from modal_launchers.terminal_runtime import check_data

    seen = []
    lock = Lock()
    for filename, prefix, count in [("train.jsonl", "train", 802), ("tb21.eval.jsonl", "eval", 89)]:
        rows = [{"prompt": f"{prefix}-{i}", "metadata": {"task_path": f"{prefix}-{i}"}} for i in range(count)]
        (tmp_path / filename).write_text("\n".join(json.dumps(row) for row in rows))

    def task(path):
        with lock:
            seen.append(path.name)
        if path.name == "train-1" and failure == "schema":
            raise ValueError("invalid task schema")
        return SimpleNamespace(instruction="mismatch" if path.name == "train-1" and failure == "prompt" else path.name)

    monkeypatch.setattr(harbor_task, "Task", task)
    config = SimpleNamespace(data=str(tmp_path), prompt_data="train.jsonl")
    if failure is None:
        check_data(config)
    else:
        message = "invalid task schema" if failure == "schema" else "Prepared prompt differs"
        with pytest.raises(ValueError, match=message):
            check_data(config)
    assert len(seen) == len(set(seen)) == 891
