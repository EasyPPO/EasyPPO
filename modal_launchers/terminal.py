"""Prepare, validate and launch the Claude.md Terminal Bench recipe on Modal.

Example: uv run --no-sync modal run --env YOUR_MODAL_ENV -d -m modal_launchers.terminal \
    --run-id tb21-9b-full-detach-2n-b200-001 --action all

The local entrypoint submits a detached GPU workload without a CPU supervisor.
Model/data artifacts and live trial logs use a v2 Volume. Checkpoints and
evaluation snapshots use a separate v2 Volume with coordinated commit/reload.
"""

import subprocess
from typing import Literal

import modal

from modal_launchers.frontiersmith import _secrets, _training_image
from modal_launchers.terminal_config import TerminalConfig
from modal_launchers.terminus import CPUS_PER_NODE

app = modal.App("easyppo-terminal-launcher")


def _build(config):
    workload = modal.App(f"easyppo-terminal-{config.run_id}")
    volume = modal.Volume.from_name(config.assets_volume, create_if_missing=True, version=2)
    outputs = modal.Volume.from_name(config.volume, create_if_missing=True, version=2)
    image = (
        _training_image(modal, config).pip_install(
            "harbor[modal]==0.24.0", "typed-argument-parser==1.12.0", "tomlkit==0.13.3"
        )
        # The current Qwen attention adapter fixes FA4's causal window;
        # override the older FrontierSmith image's FA4 opt-out.
        .env({"NVTE_FLASH_ATTN_V4": "1"})
    )
    common = dict(
        image=image,
        volumes={config.mount: volume, config.output_mount: outputs},
        secrets=_secrets(modal, config),
        env={
            "EASYPPO_TERMINAL_MODAL_APP": config.sandbox_app,
            "WANDB_DIR": f"{config.mount}/outputs/{config.run_id}/wandb",
        },
        retries=0,
        max_containers=1,
        single_use_containers=True,
        serialized=True,
    )

    @workload.function(cpu=8, memory=32768, timeout=86400, **common)
    def download():
        from modal_launchers.terminal_runtime import check_manifest, prepare_data

        volume.reload()
        check_manifest(config)
        try:
            prepare_data(config)
        finally:
            volume.commit()
            outputs.commit()

    @workload.function(gpu=config.gpu, cpu=16, memory=131072, timeout=7200, **common)
    def convert():
        from modal_launchers.runtime import check_models

        volume.reload()
        try:
            subprocess.run(
                [
                    "python3",
                    "-m",
                    "examples.easyppo.terminus_full_detach",
                    "--prepare-model-only",
                    "--model-dir",
                    config.models,
                ],
                check=True,
            )
            check_models(config)
        finally:
            volume.commit()

    @workload.function(gpu=f"{config.gpu}:2", cpu=8, memory=65536, timeout=3600, **common)
    def validate():
        import os
        from pathlib import Path

        env = dict(os.environ, OMP_NUM_THREADS="1")
        env.pop("MILES_WANDB_KEY_FROM_ENV", None)
        # Tests use synthetic credentials; do not inherit a real key into snapshots.
        env.pop("WANDB_API_KEY", None)
        suites = [
            "modal_launchers/tests",
            "tests/fast/launch_scripts",
            "tests/manual/launch_scripts",
            "tests/fast/test_train_async.py",
            "tests/fast/backends/training_utils/test_joint_checkpoint.py",
            "tests/fast/backends/megatron_utils/test_optimizer_utils.py",
            "tests/fast/rollout/test_checkpoint_eval.py",
            "tests/agentic_critic/test_critic_input.py",
            *map(str, Path("tests/agentic_critic").glob("test_*terminal*.py")),
            "tests/fast/agentic_critic/test_harbor_modal.py",
            "tests/agentic_critic/test_miles_samples.py",
            "tests/agentic_critic/test_agentpick_verifier.py",
            "tests/agentic_critic/test_trial_archive.py",
            "tests/fast/ray/rollout/test_terminal_long_budget.py",
            "tests/fast/ray/rollout/test_memory_critic_minibatches.py",
            "tests/fast/ray/rollout/test_metrics.py",
            "tests/fast/utils/test_metric_utils.py",
            "tests/fast/rollout/test_fully_async_data_buffer.py",
            "tests/fast/rollout/test_fully_async_rollout.py",
            "tests/fast/rollout/test_fully_async_rollout_checkpoint.py",
            "tests/fast/ray/rollout/test_full_history_shards.py",
            "tests/fast/models/test_qwen3_5_detached_memory.py",
            "tests/fast-gpu/test_qwen3_5_detached_memory_kernels.py",
            "tests/fast-gpu/test_fused_log_probs.py",
            "tests/fast-gpu/test_fused_log_probs_in_losses.py",
            "tests/fast-gpu/kernels/softmax/test_token_log_softmax.py",
        ]
        # Validate the live Megatron path here. The standalone HF critic's CPU
        # reference tests require an environment without CUDA-only HF kernels.
        reports = Path(config.mount, "outputs", config.run_id, "validation")
        reports.mkdir(parents=True, exist_ok=True)
        try:
            with (reports / "pytest.log").open("w") as log:
                result = subprocess.run(
                    [
                        "python3",
                        "-m",
                        "pytest",
                        "-q",
                        "-o",
                        "addopts=",
                        "--tb=short",
                        f"--junitxml={reports / 'pytest.xml'}",
                        *suites,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print("\n".join((reports / "pytest.log").read_text().splitlines()[-100:]), flush=True)
            result.check_returncode()
        finally:
            volume.commit()
        print("TERMINAL_VALIDATION_PASSED", flush=True)

    @workload.function(cpu=8, memory=32768, timeout=3600, **common)
    async def preflight():
        from modal_launchers.runtime import check_models
        from modal_launchers.terminal_runtime import check_data, check_manifest, sandbox_smoke

        await volume.reload.aio()
        check_models(config)
        check_data(config)
        check_manifest(config)
        await sandbox_smoke(config)
        await outputs.commit.aio()
        await volume.commit.aio()

    @workload.function(
        gpu=f"{config.gpu}:{config.gpus}",
        cpu=CPUS_PER_NODE,
        memory=config.memory_gib * 1024,
        timeout=86400,
        **(common | {"max_containers": config.nodes}),
    )
    @modal.clustered(size=config.nodes, rdma=True)
    def training():
        from modal_launchers.multinode import run_node
        from modal_launchers.terminal_runtime import train

        _run_training(config, modal.Cluster.from_context(), volume, outputs, run_node=run_node, train_fn=train)

    return workload, download, convert, validate, preflight, training


def _run_training(config, cluster, volume, outputs, *, run_node, train_fn):
    from pathlib import Path

    try:
        volume.reload()
        Path(config.mount, "outputs", config.run_id, "wandb").mkdir(parents=True, exist_ok=True)
        run_node(cluster, config, outputs, "", "", cpus=CPUS_PER_NODE, train_fn=train_fn)
    finally:
        # The GPU job owns its sandboxes. A preemption restarts the whole
        # cluster; no separate CPU container can cancel healthy training.
        try:
            if cluster.container_rank() == 0:
                _cleanup_sandboxes(config)
        finally:
            volume.commit()


def _cleanup_sandboxes(config):
    try:
        owner = modal.App.lookup(config.sandbox_app)
    except modal.exception.NotFoundError:
        return
    for sandbox in modal.Sandbox.list(app_id=owner.app_id):
        if sandbox.poll() is None:
            sandbox.terminate()


def _launch(config, action: str):
    if action not in ("prepare", "validate", "preflight", "train", "all"):
        raise ValueError(f"Unknown phase: {action}")
    workload, download, convert, validate, preflight, training = _build(config)
    with modal.enable_output(), workload.run(detach=True):
        if action in ("prepare", "all"):
            download.remote()
            convert.remote()
            print("TERMINAL_PREPARATION_PASSED", flush=True)
        if action in ("validate", "all"):
            validate.remote()
        if action in ("validate", "preflight", "train", "all"):
            preflight.remote()
        if action in ("train", "all"):
            call = training.spawn()
            print(f"Training app: {workload.app_id}; call: {call.object_id}; run: {config.run_id}", flush=True)
            return {"app_id": workload.app_id, "call_id": call.object_id, "run_id": config.run_id}


@app.local_entrypoint()
def main(
    run_id: str,
    action: Literal["prepare", "validate", "preflight", "train", "all"] = "all",
    mode: Literal["full_detach", "actor-summary"] = "full_detach",
    gpu: str = "B200",
    wandb_secret: str = "wandb-personal",
    wandb_entity: str = "nan-playground",
    save_interval: int = TerminalConfig.save_interval,
):
    config = TerminalConfig(
        run_id=run_id,
        mode=mode,
        gpu=gpu,
        wandb_secret=wandb_secret,
        wandb_entity=wandb_entity,
        save_interval=save_interval,
    )
    _launch(config, action)
