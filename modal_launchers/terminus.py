"""Allocate Modal nodes for the two Terminus PPO launchers."""

import os
from dataclasses import dataclass
from pathlib import Path

from modal_launchers.config import Config
from modal_launchers.frontiersmith import _secrets, _training_image
from modal_launchers.multinode import run_node

CPUS_PER_NODE = 64


@dataclass(frozen=True)
class _Allocation:
    volume: str
    mount: str
    nodes: int
    gpus: int
    run_id: str
    output: str
    image: str = Config.image
    source: str = Config.source
    terminal_environment: str = "modal"


def launch(opts, args, training_argv):
    import modal

    if args.run_id in {".", ".."} or not args.run_id or Path(args.run_id).name != args.run_id:
        raise ValueError("Use a run ID containing a single directory name")
    mount = Path(opts.modal_mount)
    paths = (args.tasks_dir, args.prompt_data, args.eval_data, args.model_dir, args.output_dir)
    if not mount.is_absolute() or not all(
        Path(path).is_relative_to(mount) and ".." not in Path(path).parts for path in paths
    ):
        raise ValueError("Place the shared model, tasks, JSONL and output paths under --modal-mount")
    if args.num_gpus_per_node != 8:
        raise ValueError("The Modal clustered launcher requires eight GPUs on each node")

    config = _Allocation(
        opts.modal_volume,
        str(mount),
        args.num_nodes,
        args.num_gpus_per_node,
        args.run_id,
        str(Path(args.output_dir) / args.run_id),
    )
    module = {"full_detach": "terminus_full_detach", "actor-summary": "terminus_actor_summary"}[
        args.critic_context_mode
    ]
    command = ["python3", "-m", f"examples.easyppo.{module}", *training_argv]
    if opts.run_id is None:
        command.extend(["--run-id", args.run_id])
    keys = ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    if opts.modal_sandbox_secret:
        sandbox_secret = modal.Secret.from_name(opts.modal_sandbox_secret, required_keys=keys)
    elif all(os.environ.get(key) for key in keys):
        sandbox_secret = modal.Secret.from_local_environ(keys)
    else:
        raise ValueError(
            "Provide --modal-sandbox-secret or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET for Harbor"
        )
    app = modal.App("easyppo-terminus-ppo")
    volume = modal.Volume.from_name(config.volume, create_if_missing=False)
    gpu = "B200" if opts.gpu_arch == "blackwell" else "H200"

    @app.function(
        image=_training_image(modal, config),
        cpu=CPUS_PER_NODE,
        gpu=f"{gpu}:{config.gpus}",
        memory=1024 * 1024,
        volumes={config.mount: volume},
        secrets=[*_secrets(modal), sandbox_secret],
        max_containers=config.nodes,
        retries=0,
        timeout=86400,
        serialized=True,
    )
    @modal.clustered(size=config.nodes, rdma=True)
    def training():
        run_node(
            modal.Cluster.from_context(), config, volume, "", "", cpus=CPUS_PER_NODE, train_command=command
        )

    with modal.enable_output(), app.run():
        training.remote()
