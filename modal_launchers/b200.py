"""Run the local checkout's AIME recipe on eight B200 GPUs with the easyppo Volume.

The image supplies dependencies; local source and data are uploaded on each run.
Export WANDB_API_KEY locally before running; it is forwarded to the container.
Optionally export WANDB_ENTITY and WANDB_PROJECT to select the W&B destination.
For timeout-only continuation, export WANDB_RUN_ID and WANDB_RESUME=must
(or allow for a new run), and use ::resume in a persistent local terminal.
Image, Volume, and paths are configured in modal_launchers/config.yaml.
Set EASYPPO_MODAL_CONFIG to use another YAML file. The image is public.
Example (starts training):
    modal run --detach modal_launchers/b200.py::train \
        --run-name aime-ours-b200
"""

import json
import os
from pathlib import Path

import modal

from modal_launchers.config import load_modal_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_modal_config()
app = modal.App("easyppo-aime")

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
wandb_env_keys = ["WANDB_API_KEY"]
for key in ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_ID", "WANDB_RESUME"):
    if os.environ.get(key):
        wandb_env_keys.append(key)


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def check_environment():
    """Check the uploaded checkout on CPU without starting training."""
    import subprocess

    subprocess.run(["python3", "scripts/check_environment.py"], cwd=CONFIG.source_root, check=True)


@app.function(
    image=image,
    gpu="B200:8",
    cpu=32,
    memory=262144,  # 256 GiB of host memory, matching the repository launcher.
    timeout=24 * 60 * 60,
    # Ordinary training errors must reach the caller without being retried.
    retries=0,
    single_use_containers=True,
    volumes={CONFIG.volume_mount: volume},
    secrets=[modal.Secret.from_local_environ(wandb_env_keys)] if os.environ.get("WANDB_API_KEY") else [],
    max_containers=1,
)
def train(
    model_path: str = CONFIG.model_path,
    run_name: str = "aime-ours-b200",
):
    import os
    import re
    import subprocess
    from pathlib import Path

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name must contain only letters, digits, underscores and hyphens")

    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Export WANDB_API_KEY locally before starting training")

    if not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError(f"{model_path} must contain config.json, model weights and tokenizer files")

    output = CONFIG.output_path(run_name)

    try:
        subprocess.run(
            [
                "bash",
                "scripts/run_aime_ours.sh",
                f"actor_rollout_ref.model.path={json.dumps(model_path)}",
                # Disable configurable offload. Reference FSDP still forces CPU offload upstream.
                "actor_rollout_ref.model.enable_activation_offload=false",
                "actor_rollout_ref.actor.fsdp_config.param_offload=false",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
                "actor_rollout_ref.ref.fsdp_config.param_offload=false",
                "critic.model.enable_activation_offload=false",
                "critic.fsdp.param_offload=false",
                "critic.fsdp.optimizer_offload=false",
                # Reserve headroom for resident training state on 180 GB B200s.
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.50",
                "actor_rollout_ref.rollout.free_cache_engine=true",
                "trainer.nnodes=1",
                "trainer.n_gpus_per_node=8",
                "trainer.logger=[console,wandb]",
                f"trainer.project_name={json.dumps(os.environ.get('WANDB_PROJECT', 'EasyPPO'))}",
                f"trainer.experiment_name={run_name}",
                f"trainer.default_local_dir={json.dumps(output + '/checkpoints')}",
                f"trainer.rollout_data_dir={json.dumps(output + '/rollouts')}",
                f"trainer.validation_data_dir={json.dumps(output + '/validation')}",
                "trainer.resume_mode=auto",
            ],
            cwd=CONFIG.source_root,
            env=dict(os.environ, **CONFIG.container_env(), PYTHON_BIN="python3"),
            check=True,
        )
    finally:
        volume.commit()


@app.local_entrypoint()
def resume(
    model_path: str = CONFIG.model_path,
    run_name: str = "aime-ours-b200",
    max_timeout_restarts: int = 10,
):
    """Resume after Function timeouts only; keep this local process alive (e.g. tmux)."""
    from modal.exception import FunctionTimeoutError

    if max_timeout_restarts < 0:
        raise ValueError("max_timeout_restarts must be non-negative")
    for key in ("WANDB_API_KEY", "WANDB_RUN_ID"):
        if not os.environ.get(key):
            raise ValueError(f"Export {key} locally before starting resumable training")
    if os.environ.get("WANDB_RESUME") not in ("must", "allow"):
        raise ValueError("Set WANDB_RESUME=must for an existing run, or allow for a new run")

    for attempt in range(max_timeout_restarts + 1):
        try:
            train.remote(model_path=model_path, run_name=run_name)
            return
        except FunctionTimeoutError:
            if attempt == max_timeout_restarts:
                raise
            print(
                f"Training timed out; restarting from the latest saved checkpoint "
                f"({attempt + 1}/{max_timeout_restarts}), run_name={run_name}",
                flush=True,
            )
