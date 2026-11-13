"""Modal cluster bootstrap, adapted from the FCS branch's multinode launcher.

Miles owns the Ray job and training workers. This module joins the platform
allocation and coordinates commit/reload across its separate Volume mounts.
"""

import os
import socket
import subprocess
import time
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

RAY_PORT = 6379
DASHBOARD_PORT = 8265
CONTROL_PORT = 29599
STARTUP_TIMEOUT = 900


def _store(head_ip, *, master=False):
    # CPU coordination only; no process group or CUDA initialization.
    from torch.distributed import TCPStore

    return TCPStore(
        head_ip, CONTROL_PORT, None, master, timeout=timedelta(seconds=STARTUP_TIMEOUT), wait_for_workers=False
    )


def _check_health(store, process, nodes):
    if process is not None and process.poll() is not None:
        raise RuntimeError(f"Local Ray process exited with code {process.returncode}")
    for rank in range(nodes):
        if store.check([f"error/{rank}"]):
            raise RuntimeError(f"Node {rank} failed: {store.get(f'error/{rank}').decode()}")


def _wait(store, predicate, process, nodes, description):
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not predicate():
        _check_health(store, process, nodes)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for {description}")
        time.sleep(0.2)


def _commit_all(store, volume, nodes, generation):
    # All commits finish before any reload. Workers acknowledge reloading before
    # the driver may resume training or publish the next completion marker.
    store.set("commit", generation)
    volume.commit()
    keys = [f"committed/{rank}/{generation}" for rank in range(1, nodes)]
    _wait(store, lambda: store.check(keys), None, nodes, "checkpoint commits")
    store.set("reload", generation)
    volume.reload()
    keys = [f"reloaded/{rank}/{generation}" for rank in range(1, nodes)]
    _wait(store, lambda: store.check(keys), None, nodes, "checkpoint reloads")


def commit_cluster_volume(volume, *, nodes, head_ip):
    """Storage hook called synchronously by the head's training driver."""
    _commit_all(_store(head_ip), volume, nodes, uuid4().hex)


def _worker_loop(store, process, volume, *, rank, nodes):
    generation = ""
    while not store.check(["finished"]):
        _check_health(store, process, nodes)
        if store.check(["commit"]):
            requested = store.get("commit").decode()
            if requested != generation:
                volume.commit()
                store.set(f"committed/{rank}/{requested}", "1")
                _wait(
                    store,
                    lambda target=requested: store.check(["reload"]) and store.get("reload").decode() == target,
                    process,
                    nodes,
                    "reload barrier",
                )
                volume.reload()
                store.set(f"reloaded/{rank}/{requested}", "1")
                generation = requested
        time.sleep(0.2)
    store.set(f"finished/{rank}", "1")


def _node_environment(config, *, rank, addresses, judge_token):
    import psutil

    head_ip, node_ip = addresses[0], addresses[rank]
    env = dict(os.environ)
    env.update(
        MASTER_ADDR=head_ip,
        RAY_ADDRESS=f"{head_ip}:{RAY_PORT}",
        MILES_SCRIPT_EXTERNAL_RAY="1",
        NCCL_SOCKET_FAMILY="AF_INET",
        FRONTIERCS_JUDGE_TOKEN=judge_token,
        EASYPPO_MODAL_NODES=str(config.nodes),
        EASYPPO_MODAL_VOLUME=config.volume,
        WANDB_RUN_ID=config.run_id,
        WANDB_RESUME="allow",
    )
    for interface, items in psutil.net_if_addrs().items():
        if any(item.family == socket.AF_INET and item.address == node_ip for item in items):
            env["NCCL_SOCKET_IFNAME"] = f"={interface}"
            env["GLOO_SOCKET_IFNAME"] = interface
            return env
    raise RuntimeError(f"No interface for Modal cluster IPv4 address {node_ip}")


def _ray_command(config, *, rank, addresses):
    command = [
        "ray",
        "start",
        f"--node-ip-address={addresses[rank]}",
        f"--num-gpus={config.gpus}",
        "--num-cpus=32",
        "--disable-usage-stats",
        "--block",
    ]
    if rank == 0:
        command += [
            "--head",
            f"--port={RAY_PORT}",
            "--include-dashboard=true",
            "--dashboard-host=0.0.0.0",
            f"--dashboard-port={DASHBOARD_PORT}",
        ]
    else:
        command += [f"--address={addresses[0]}:{RAY_PORT}"]
    return command


def _recover_run(config):
    # Reconcile the existing joint checkpoint once, before any worker loads it.
    from miles.backends.training_utils.checkpoint.joint_checkpoint import recover_checkpoint

    root = f"{config.output}/checkpoints"
    recover_checkpoint(
        SimpleNamespace(
            save=root,
            load=root,
            critic_save=root + "_critic",
            critic_load=root + "_critic",
            finetune=False,
            max_keep=2,
            use_critic=True,
            megatron_config=None,
            rollout_global_dataset=True,
            checkpoint_commit_hook=None,
        )
    )


def _cluster_ready(ray, addresses, gpus):
    alive = [node for node in ray.nodes() if node["Alive"]]
    return (
        len(alive) == len(addresses)
        and {node["NodeManagerAddress"] for node in alive} == set(addresses)
        and all(node["Resources"].get("GPU", 0) == gpus for node in alive)
    )


def _port_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def _run_head(store, process, volume, config, addresses, judge_url, judge_token):
    import ray
    from modal_launchers.runtime import train

    ray.init(address=f"{addresses[0]}:{RAY_PORT}", log_to_driver=False)
    try:
        _wait(store, lambda: _cluster_ready(ray, addresses, config.gpus), process, config.nodes, "all Ray nodes")

        def run_command(command, *, check):
            # The launcher talks to Ray Jobs over HTTP; the raylets and Ray
            # workers retain their TCP GCS address in the inherited environment.
            env = dict(os.environ, RAY_ADDRESS=f"http://{addresses[0]}:{DASHBOARD_PORT}")
            trainer = subprocess.Popen(command, cwd=config.source, env=env)
            try:
                while trainer.poll() is None:
                    _check_health(store, process, config.nodes)
                    if not _cluster_ready(ray, addresses, config.gpus):
                        raise RuntimeError("A Ray node disappeared during training")
                    time.sleep(1)
                if check and trainer.returncode:
                    raise subprocess.CalledProcessError(trainer.returncode, command)
            finally:
                _stop_process(trainer)

        train(config, judge_url, judge_token, run_command=run_command)
        _commit_all(store, volume, config.nodes, uuid4().hex)
        store.set("finished", "1")
        keys = [f"finished/{rank}" for rank in range(1, config.nodes)]
        _wait(store, lambda: store.check(keys), process, config.nodes, "worker completion")
    finally:
        ray.shutdown()


def run_node(cluster, config, volume, judge_url, judge_token):
    addresses = list(cluster.container_ips(family="ipv4"))
    rank = cluster.container_rank()
    if len(addresses) != config.nodes or len(set(addresses)) != config.nodes or not 0 <= rank < config.nodes:
        raise ValueError("Modal allocation does not match the requested node count")
    os.environ.update(_node_environment(config, rank=rank, addresses=addresses, judge_token=judge_token))
    store = _store(addresses[0], master=rank == 0)
    process = None
    try:
        volume.reload()
        if rank == 0:
            _recover_run(config)
            volume.commit()
            store.set("recovery_done", "1")
        else:
            _wait(store, lambda: store.check(["recovery_done"]), None, config.nodes, "checkpoint recovery")
            volume.reload()
        store.set(f"interface/{rank}", os.environ["NCCL_SOCKET_IFNAME"])
        store.set(f"ready/{rank}", "1")
        keys = [f"ready/{i}" for i in range(config.nodes)]
        _wait(store, lambda: store.check(keys), None, config.nodes, "node storage readiness")
        # Miles broadcasts the head's NCCL interface name in the Ray job env.
        # Check that every node exposes that same interface before submission.
        interfaces = {store.get(f"interface/{i}") for i in range(config.nodes)}
        if len(interfaces) != 1:
            raise RuntimeError("Modal nodes have different NCCL interface names")
        if rank:
            _wait(store, lambda: store.check(["head_started"]), None, config.nodes, "Ray head")
        process = subprocess.Popen(_ray_command(config, rank=rank, addresses=addresses))
        if rank == 0:
            _wait(store, lambda: _port_ready(addresses[0], DASHBOARD_PORT), process, config.nodes, "Ray Jobs API")
            store.set("head_started", "1")
            _run_head(store, process, volume, config, addresses, judge_url, judge_token)
        else:
            _worker_loop(store, process, volume, rank=rank, nodes=config.nodes)
    except BaseException as error:
        store.set(f"error/{rank}", f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            subprocess.run(["ray", "stop", "--force"], check=False, timeout=60)
        finally:
            _stop_process(process)
            volume.commit()
