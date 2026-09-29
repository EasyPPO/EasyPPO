"""Modal environment check (default) and explicitly invoked EasyPPO training."""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = "/workspace/EasyPPO"
app = modal.App("easyppo-aime")
image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .env({"PYTHONPATH": SOURCE_ROOT})
    .add_local_dir(
        ROOT,
        remote_path=SOURCE_ROOT,
        copy=False,
        ignore=modal.FilePatternMatcher.from_file(ROOT / ".dockerignore"),
    )
)
cache = modal.Volume.from_name("easyppo-cache", create_if_missing=True)
outputs = modal.Volume.from_name("easyppo-outputs", create_if_missing=True)
# Optional named Secret containing HF_TOKEN and/or WANDB_API_KEY.
secrets = [modal.Secret.from_name(os.environ["EASYPPO_SECRET"])] if os.getenv("EASYPPO_SECRET") else []


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def check_environment():
    """Build/check the image on CPU; never load or download a model."""
    import subprocess

    subprocess.run(["python3", "scripts/check_environment.py"], cwd=SOURCE_ROOT, check=True)


@app.function(
    image=image,
    gpu="H200:8",
    cpu=32,
    memory=262144,
    timeout=24 * 60 * 60,
    volumes={"/cache": cache, "/outputs": outputs},
    secrets=secrets,
    max_containers=1,
)
def train(run_name: str = "aime-ours", resume: bool = True):
    """Explicit GPU entrypoint for the user's later tests; not invoked by default."""
    import re
    import subprocess

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name must contain only letters, digits, underscores and hyphens")
    env = dict(os.environ, EASYPPO_OUTPUT_DIR="/outputs")
    try:
        subprocess.run(
            [
                "bash",
                "scripts/run_aime_ours.sh",
                f"trainer.experiment_name={run_name}",
                f"trainer.default_local_dir=/outputs/{run_name}/checkpoints",
                f"trainer.rollout_data_dir=/outputs/{run_name}/rollouts",
                f"trainer.validation_data_dir=/outputs/{run_name}/validation",
                "trainer.resume_mode=auto" if resume else "trainer.resume_mode=disable",
            ],
            cwd=SOURCE_ROOT,
            env=env,
            check=True,
        )
    finally:
        try:
            outputs.commit()
        finally:
            cache.commit()


@app.local_entrypoint()
def main():
    check_environment.remote()
