"""AIME training on one Modal cluster: four nodes with eight B200s each.

New run (starts training):
    modal run --detach modal_launchers/b200_32gpu.py::train --run-name aime-b200-32gpu-NEW

Credentials: WANDB_API_KEY and optional WANDB_ENTITY/WANDB_PROJECT.
Image, Volume, and paths are configured in modal_launchers/config.yaml.
Set EASYPPO_MODAL_CONFIG to use another YAML file. The image is public.
The run name also identifies the W&B run, independently of any WANDB_RUN_ID
or WANDB_RESUME left in the launching shell. Reuse the name only to resume
this 32-GPU run. Use ::resume in tmux for timeout-only automatic continuation.
"""

import os
from pathlib import Path

import modal
import modal.experimental

from modal_launchers.config import load_modal_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_modal_config()
app = modal.App("easyppo-aime-32gpu")

image = (
    modal.Image.from_registry(CONFIG.image)
    .env(CONFIG.container_env())
    .add_local_dir(
        ROOT,
        remote_path=CONFIG.source_root,
        copy=False,
        ignore=modal.FilePatternMatcher.from_file(ROOT / ".dockerignore"),
    )
)
volume = modal.Volume.from_name(CONFIG.volume_name)
wandb_keys = ["WANDB_API_KEY"] + [key for key in ("WANDB_ENTITY", "WANDB_PROJECT") if os.environ.get(key)]


@app.function(
    image=image,
    gpu="B200:8",
    cpu=32,
    memory=262144,
    timeout=86400,
    retries=0,
    single_use_containers=True,
    max_containers=4,
    volumes={CONFIG.volume_mount: volume},
    secrets=[modal.Secret.from_local_environ(wandb_keys)] if os.environ.get("WANDB_API_KEY") else [],
)
@modal.experimental.clustered(size=4, rdma=True)
def train(run_name: str, model_path: str = CONFIG.model_path):
    from modal_launchers.multinode import run_node

    run_node(
        modal.experimental.get_cluster_info(),
        run_name=run_name,
        model_path=model_path,
        commit=volume.commit,
        config=CONFIG,
    )


@app.local_entrypoint()
def resume(run_name: str, model_path: str = CONFIG.model_path, max_timeout_restarts: int = 10):
    """Keep this local process alive; only Function timeouts start another cluster."""
    from modal.exception import FunctionTimeoutError

    from modal_launchers.multinode import validate_run_name

    validate_run_name(run_name)
    if max_timeout_restarts < 0:
        raise ValueError("max_timeout_restarts must be non-negative")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Export WANDB_API_KEY before starting training")
    for attempt in range(max_timeout_restarts + 1):
        try:
            train.remote(run_name=run_name, model_path=model_path)
            return
        except FunctionTimeoutError:
            if attempt == max_timeout_restarts:
                raise
            print(f"Function timeout: continuing {run_name} ({attempt + 1}/{max_timeout_restarts})", flush=True)
