"""Ray orchestration for the 4 x 8 B200 launcher (no GPU work at import time)."""

import json
import os
import re
import socket
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from modal_launchers.config import load_modal_config

NODES = 4
GPUS_PER_NODE = 8
RAY_PORT = 6379
CONTROL_PORT = 29599
STARTUP_TIMEOUT = 900


def validate_run_name(run_name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_name):
        raise ValueError("run_name must contain 1-128 letters, digits, underscores or hyphens")


def training_command(model_path, run_name, project, config=None):
    config = config or load_modal_config()
    output = config.output_path(run_name)
    return [
        "bash",
        "scripts/run_aime_ours.sh",
        f"actor_rollout_ref.model.path={json.dumps(model_path)}",
        "actor_rollout_ref.model.enable_activation_offload=false",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.ref.fsdp_config.param_offload=false",
        "critic.model.enable_activation_offload=false",
        "critic.fsdp.param_offload=false",
        "critic.fsdp.optimizer_offload=false",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.50",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        # Keep eight-way sharding within each node, with four replicas.
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=8",
        "actor_rollout_ref.ref.fsdp_config.fsdp_size=8",
        "critic.fsdp.fsdp_size=8",
        "trainer.nnodes=4",
        "trainer.n_gpus_per_node=8",
        "trainer.total_training_steps=150",
        "trainer.logger=[console,wandb]",
        # JSON quoting also handles spaces and punctuation in Hydra string values.
        f"trainer.project_name={json.dumps(project)}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={json.dumps(output + '/checkpoints')}",
        f"trainer.rollout_data_dir={json.dumps(output + '/rollouts')}",
        f"trainer.validation_data_dir={json.dumps(output + '/validation')}",
        "trainer.resume_mode=auto",
        "+ray_kwargs.ray_init.address=auto",
    ]


def prepare_output(output, run_name, project, entity, commit):
    """Refuse to reuse an 8-GPU checkpoint or an unrelated experiment directory."""
    output = Path(output)
    manifest = output / "modal_32gpu_run.json"
    expected = {
        "world_size": NODES * GPUS_PER_NODE,
        "run_name": run_name,
        "wandb_run_id": run_name,
        "wandb_project": project,
        "wandb_entity": entity,
    }
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f"Run configuration does not match {manifest}; use a new run name")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to reuse existing output without a 32-GPU manifest: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2) + "\n")
        temporary.replace(manifest)
        commit()


def node_environment(node_ip, head_ip, run_name, config=None):
    import psutil

    config = config or load_modal_config()
    env = dict(os.environ, **config.container_env())
    env.update(
        RAY_ADDRESS=f"{head_ip}:{RAY_PORT}",
        VLLM_HOST_IP=node_ip,
        WANDB_RUN_ID=run_name,
        WANDB_RESUME="allow",
        PYTHON_BIN="python3",
        # Modal may inject AF_INET6. The selected interface below carries the
        # cluster IPv4 address, so the family must be changed along with it.
        NCCL_SOCKET_FAMILY="AF_INET",
        HYDRA_FULL_ERROR="1",
    )
    # Pick the interface for Modal's cluster IPv4 address, rather than a public
    # interface or loopback. NCCL selects its RDMA transport independently.
    for interface, addresses in psutil.net_if_addrs().items():
        if any(address.family == socket.AF_INET and address.address == node_ip for address in addresses):
            env["NCCL_SOCKET_IFNAME"] = f"={interface}"
            env["GLOO_SOCKET_IFNAME"] = interface
            break
    else:
        raise RuntimeError(f"No local interface found for cluster IPv4 address {node_ip}")
    return env


class NcclProbe:
    """One Ray actor per GPU, using the same raylet environment as the trainer."""

    def check(self, rank, world_size, head_ip, port):
        import torch
        import torch.distributed as dist

        print(
            f"NCCL probe rank={rank} interface={os.environ.get('NCCL_SOCKET_IFNAME')} "
            f"family={os.environ.get('NCCL_SOCKET_FAMILY')} "
            f"node_ip={os.environ.get('VLLM_HOST_IP')}",
            flush=True,
        )
        torch.cuda.set_device(0)
        try:
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://{head_ip}:{port}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=120),
            )
            tensor = torch.tensor([rank + 1], device="cuda", dtype=torch.float32)
            dist.all_reduce(tensor)
            expected = world_size * (world_size + 1) / 2
            if tensor.item() != expected:
                raise RuntimeError(f"NCCL all-reduce returned {tensor.item()}, expected {expected}")
            return rank
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def check_nccl_cluster(ray, addresses):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    with socket.socket() as listener:
        listener.bind((addresses[0], 0))
        port = listener.getsockname()[1]
    nodes = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
    actors = []
    try:
        actor_class = ray.remote(num_gpus=1, num_cpus=1)(NcclProbe)
        for address in addresses:
            for _ in range(GPUS_PER_NODE):
                actors.append(
                    actor_class.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[address], soft=False),
                    ).remote()
                )
        ray.get(
            [actor.check.remote(rank, len(actors), addresses[0], port) for rank, actor in enumerate(actors)],
            timeout=240,
        )
        print(f"NCCL preflight passed: all-reduce across {len(actors)} GPUs", flush=True)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)


def ray_command(rank, node_ip, head_ip):
    command = [
        "ray",
        "start",
        f"--node-ip-address={node_ip}",
        "--num-gpus=8",
        "--num-cpus=32",
        "--disable-usage-stats",
        "--block",
    ]
    if rank == 0:
        command += ["--head", f"--port={RAY_PORT}", "--include-dashboard=false"]
    else:
        command += [f"--address={head_ip}:{RAY_PORT}"]
    return command


def check_health(store, ray_process):
    if ray_process.poll() is not None:
        raise RuntimeError(f"Local Ray process exited with code {ray_process.returncode}")
    for rank in range(NODES):
        key = f"error/{rank}"
        if store.check([key]):
            raise RuntimeError(f"Node {rank} failed: {store.get(key).decode()}")


def wait_for(store, predicate, ray_process, description, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not predicate():
        check_health(store, ray_process)
        if time.monotonic() >= deadline:
            # Deliberately not FunctionTimeoutError: setup failures must not
            # trigger the local entrypoint's 24-hour continuation policy.
            raise RuntimeError(f"Timed out waiting for {description}")
        time.sleep(1)


def commit_all_nodes(store, ray_process, commit, generation):
    # The trainer writes its iteration marker after all ranks have saved. Ask
    # every node to commit its own Volume view before committing the head view.
    store.set("commit", generation)
    keys = [f"committed/{rank}/{generation}" for rank in range(1, NODES)]
    wait_for(store, lambda: store.check(keys), ray_process, "worker checkpoint commits")
    commit()


def worker_loop(store, ray_process, commit):
    generation = ""
    while not store.check(["finished"]):
        check_health(store, ray_process)
        if store.check(["commit"]):
            requested = store.get("commit").decode()
            if requested != generation:
                commit()
                generation = requested
                yield generation
        time.sleep(1)


def run_node(cluster, *, run_name, model_path, commit, config=None):
    import ray
    from torch.distributed import TCPStore

    validate_run_name(run_name)
    config = config or load_modal_config()
    addresses = list(cluster.container_ipv4_ips)
    if len(addresses) != NODES or any(not address for address in addresses):
        raise RuntimeError("This launcher requires four Modal cluster IPv4 addresses")
    rank = cluster.rank
    head_ip, node_ip = addresses[0], addresses[rank]
    env = node_environment(node_ip, head_ip, run_name, config)
    # Ray actors inherit their local raylet's environment. Update the parent too
    # so Ray introspection uses the same address as the training subprocess.
    os.environ.update(env)
    store = TCPStore(
        head_ip,
        CONTROL_PORT,
        NODES,
        rank == 0,
        timeout=timedelta(seconds=STARTUP_TIMEOUT),
        wait_for_workers=True,
    )
    ray_process = trainer = None
    connected = False
    try:
        if not env.get("WANDB_API_KEY"):
            raise ValueError("Export WANDB_API_KEY before starting training")
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(f"Missing model config: {model_path}/config.json")
        project = env.get("WANDB_PROJECT", "EasyPPO")
        output = Path(config.output_path(run_name))
        if rank == 0:
            prepare_output(output, run_name, project, env.get("WANDB_ENTITY"), commit)
            store.set("output_ready", "1")
        else:
            # Also surface head preflight errors instead of waiting for 24 hours.
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while not store.check(["output_ready"]):
                if store.check(["error/0"]):
                    raise RuntimeError(store.get("error/0").decode())
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for head preflight")
                time.sleep(1)

        # Start the head before allowing workers to contact its GCS service.
        if rank != 0:
            store.wait(["head_started"])
        ray_process = subprocess.Popen(ray_command(rank, node_ip, head_ip), env=env)
        if rank == 0:

            def gcs_ready():
                try:
                    with socket.create_connection((head_ip, RAY_PORT), timeout=1):
                        return True
                except OSError:
                    return False

            wait_for(store, gcs_ready, ray_process, "Ray head")
            store.set("head_started", "1")
        else:
            for generation in worker_loop(store, ray_process, commit):
                store.set(f"committed/{rank}/{generation}", "1")
            store.set(f"finished/{rank}", "1")
            return

        ray.init(address=f"{head_ip}:{RAY_PORT}", log_to_driver=False)
        connected = True

        def cluster_ready():
            nodes = [node for node in ray.nodes() if node["Alive"]]
            return (
                len(nodes) == NODES
                and {node["NodeManagerAddress"] for node in nodes} == set(addresses)
                and all(node["Resources"].get("GPU", 0) == GPUS_PER_NODE for node in nodes)
            )

        wait_for(store, cluster_ready, ray_process, "four Ray nodes with eight GPUs each")
        print(f"Ray cluster ready: {NODES} nodes / {NODES * GPUS_PER_NODE} B200s; run={run_name}", flush=True)
        check_nccl_cluster(ray, addresses)
        trainer = subprocess.Popen(
            training_command(model_path, run_name, project, config),
            cwd=config.source_root,
            env=env,
        )
        tracker = output / "checkpoints/latest_checkpointed_iteration.txt"
        last_step = tracker.read_text().strip() if tracker.exists() else ""
        while trainer.poll() is None:
            check_health(store, ray_process)
            if not cluster_ready():
                raise RuntimeError("A Ray node disappeared during training")
            if tracker.exists():
                step = tracker.read_text().strip()
                if step.isdigit() and step != last_step:
                    commit_all_nodes(store, ray_process, commit, f"step-{step}")
                    last_step = step
            time.sleep(2)
        if trainer.returncode:
            raise subprocess.CalledProcessError(trainer.returncode, trainer.args)
        commit_all_nodes(store, ray_process, commit, "final")
        store.set("finished", "1")
        wait_for(
            store,
            lambda: store.check([f"finished/{rank}" for rank in range(1, NODES)]),
            ray_process,
            "worker completion",
        )
    except BaseException as error:
        try:
            store.set(f"error/{rank}", f"{type(error).__name__}: {error}")
        except Exception:
            pass
        raise
    finally:
        try:
            if trainer is not None and trainer.poll() is None:
                trainer.terminate()
                try:
                    trainer.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    trainer.kill()
                    trainer.wait(timeout=15)
            if connected:
                ray.shutdown()
            if ray_process is not None:
                subprocess.run(["ray", "stop", "--force"], env=env, timeout=60, check=False)
                try:
                    ray_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    ray_process.kill()
                    ray_process.wait(timeout=15)
        finally:
            commit()
