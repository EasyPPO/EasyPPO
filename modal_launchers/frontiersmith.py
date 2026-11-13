"""Two Claude.md experiments on Modal, one run per invocation.

Preview: python -m modal_launchers.frontiersmith --mode full_detach --run-id sync-fs-full-detach-001
Launch:  append --launch after reviewing the plan. Keep the local client alive in tmux.
CPU SDK/storage smoke: python -m modal_launchers.frontiersmith --action cpu-smoke --launch
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from modal_launchers.config import parse_args

ROOT = Path(__file__).resolve().parents[1]
IGNORE = [".git", "**/__pycache__", ".venv", "outputs", "wandb", "*.log"]


def _judge_image(modal):
    root = ROOT / "Frontier-CS/algorithmic"
    return (
        modal.Image.from_dockerfile(
            root / "Dockerfile",
            context_dir=root,
            ignore=["problems/**", "data/**", "submissions/**", "node_modules/**"],
        )
        .entrypoint([])
        .add_local_dir(root / "problems", "/app/problems", copy=True)
    )


def _training_image(modal, config):
    return (
        modal.Image.from_registry(config.image)
        .entrypoint([])
        .apt_install("libibverbs1", "ibverbs-providers")
        .pip_install("harbor==0.24.0", "mini-swe-agent==1.17.1", "transformers==5.12.1", "modal==1.6.2")
        .add_local_dir(ROOT, config.source, ignore=IGNORE, copy=True)
        .run_commands(f"python3 -m pip install --no-deps -e {config.source}")
        .env(
            {
                "PYTHONPATH": f"{config.source}:/root/Megatron-LM",
                "PYTHONUNBUFFERED": "1",
                "HF_HOME": f"{config.mount}/cache/huggingface",
                "EASYPPO_MODAL_VOLUME": config.volume,
            }
        )
        .workdir(config.source)
    )


def _secrets(modal):
    from modal.config import config as modal_config

    # Harbor's Modal backend authenticates from these variables, including in
    # Ray subprocesses. Read the active SDK profile; never log credential values.
    credentials = {
        "MODAL_TOKEN_ID": modal_config.get("token_id"),
        "MODAL_TOKEN_SECRET": modal_config.get("token_secret"),
    }
    if not all(credentials.values()):
        raise ValueError("Configure the Modal CLI or export MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
    keys = [key for key in ("WANDB_API_KEY", "WANDB_ENTITY", "HF_TOKEN") if os.getenv(key)]
    return [modal.Secret.from_dict(credentials)] + ([modal.Secret.from_local_environ(keys)] if keys else [])


def _build_training(modal, config):
    app = modal.App("easyppo-miles-frontiersmith")
    volume = modal.Volume.from_name(config.volume, create_if_missing=True)
    image = _training_image(modal, config)
    common = dict(
        image=image,
        volumes={config.mount: volume},
        secrets=_secrets(modal),
        retries=0,
        single_use_containers=True,
        serialized=True,
        max_containers=1,
    )

    @app.function(cpu=4, memory=16384, timeout=3600, **common)
    def preflight(judge_url):
        from modal_launchers.runtime import check_models, prepare_run

        volume.reload()
        if Path(config.output, "training_complete.json").exists():
            from modal_launchers.config import check_manifest

            check_manifest(
                config,
                wandb_project="miles-run_qwen3_5_27b_sync_frontiersmith",
                wandb_entity=os.environ.get("WANDB_ENTITY"),
            )
            return True
        check_models(config)
        prepare_run(config, judge_url)
        return False

    def training(judge_url, judge_token):
        from modal_launchers.runtime import train

        if config.nodes > 1:
            from modal_launchers.multinode import run_node

            run_node(modal.Cluster.from_context(), config, volume, judge_url, judge_token)
        else:
            volume.reload()
            train(config, judge_url, judge_token)

    if config.nodes > 1:
        training = modal.clustered(size=config.nodes, rdma=True)(training)
    training = app.function(
        gpu=f"{config.gpu}:{config.gpus}",
        cpu=32,
        memory=config.memory_gib * 1024,
        timeout=86400,
        **(common | {"max_containers": config.nodes}),
    )(training)

    return app, volume, preflight, training


def _check_judge(url, token):
    request = urllib.request.Request(url + "/problems", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _attempt(modal, app, config, image, preflight, training):
    judge = modal.Sandbox.create(
        "/app/entrypoint.sh",
        app=app,
        image=image,
        runtime="vm",
        cpu=config.judge_cpus,
        memory=config.judge_memory_gib * 1024,
        timeout=86400,
        env={
            "PORT": "8081",
            "GJ_ADDR": "http://127.0.0.1:5050",
            "JUDGE_WORKERS": str(config.judge_workers),
            "GJ_PARALLELISM": str(config.judge_workers),
            "SAVE_OUTPUTS": "false",
        },
        readiness_probe=modal.Probe.with_tcp(8081),
    )
    deadline = time.monotonic() + 86400 - 600
    call = None
    completed = False
    try:
        judge.wait_until_ready(timeout=600)
        credentials = judge.create_connect_token(port=8081)
        _check_judge(credentials.url, credentials.token)
        if preflight.remote(credentials.url):
            print("This run already completed training and final evaluation.")
            return
        call = training.spawn(credentials.url, credentials.token)
        try:
            call.get(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError as exc:
            if time.monotonic() < deadline:
                raise
            raise JudgeLifetimeTimeout("Restarting before the judge expires") from exc
        completed = True
    finally:
        try:
            if call is not None and not completed:
                call.cancel(terminate_containers=True)
        finally:
            judge.terminate()


class JudgeLifetimeTimeout(TimeoutError):
    pass


def _train(modal, config):
    from modal.exception import FunctionTimeoutError

    app, _, preflight, training = _build_training(modal, config)
    judge_image = _judge_image(modal)
    with modal.enable_output(), app.run():
        for attempt in range(config.max_restarts + 1):
            try:
                _attempt(modal, app, config, judge_image, preflight, training)
                return
            except (FunctionTimeoutError, JudgeLifetimeTimeout):
                if attempt == config.max_restarts:
                    raise
                print(f"Timeout; resuming {config.run_id}, restart {attempt + 1}/{config.max_restarts}", flush=True)


def _prepare(modal, config):
    app = modal.App("easyppo-miles-prepare")
    volume = modal.Volume.from_name(config.volume, create_if_missing=True)
    image = _training_image(modal, config)

    @app.function(
        image=image,
        cpu=4,
        memory=16384,
        timeout=86400,
        retries=0,
        volumes={config.mount: volume},
        secrets=_secrets(modal),
        serialized=True,
    )
    def download():
        from examples.easyppo.run_qwen3_5_27b_sync_frontiersmith import ScriptArgs
        from huggingface_hub import snapshot_download
        from modal_launchers.config import check_manifest

        volume.reload()
        check_manifest(
            config,
            wandb_project="miles-run_qwen3_5_27b_sync_frontiersmith",
            wandb_entity=os.environ.get("WANDB_ENTITY"),
        )
        try:
            snapshot_download(
                ScriptArgs.hf_repo, revision=ScriptArgs.hf_revision, local_dir=f"{config.models}/Qwen3.5-27B"
            )
        finally:
            volume.commit()

    @app.function(
        image=image,
        gpu=f"{config.gpu}:{config.gpus}",
        cpu=16,
        memory=config.memory_gib * 1024,
        timeout=86400,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        volumes={config.mount: volume},
        secrets=_secrets(modal),
        serialized=True,
    )
    def prepare():
        from modal_launchers.config import check_manifest

        volume.reload()
        check_manifest(
            config,
            wandb_project="miles-run_qwen3_5_27b_sync_frontiersmith",
            wandb_entity=os.environ.get("WANDB_ENTITY"),
        )
        volume.commit()
        try:
            subprocess.run(config.command("prepare"), check=True, cwd=config.source)
        finally:
            volume.commit()

    with modal.enable_output(), app.run():
        download.remote()
        prepare.remote()


def _cpu_smoke(modal, config):
    app = modal.App("easyppo-miles-cpu-smoke")
    volume = modal.Volume.from_name(config.volume, create_if_missing=True)
    probe = f"{config.mount}/.smoke-{uuid4().hex}"

    @app.function(
        image=modal.Image.debian_slim(python_version="3.12"),
        cpu=1,
        memory=256,
        timeout=120,
        retries=0,
        max_containers=1,
        single_use_containers=True,
        volumes={config.mount: volume},
        serialized=True,
    )
    def check(write):
        volume.reload()
        path = Path(probe)
        if write:
            path.write_text("checkpoint-commit-probe")
        else:
            assert path.read_text() == "checkpoint-commit-probe"
            path.unlink()
        volume.commit()
        return {"gpu": False, "write" if write else "reload": "passed"}

    with modal.enable_output(), app.run():
        print(check.remote(True))
        print(check.remote(False))


def main(argv=None):
    args, config = parse_args(argv)
    print(
        json.dumps(
            {
                "action": args.action,
                "launch": args.launch,
                "nodes": 0 if args.action == "cpu-smoke" else (1 if args.action == "prepare" else config.nodes),
                "gpus_per_node": 0 if args.action == "cpu-smoke" else config.gpus,
                "gpu_count": (
                    0
                    if args.action == "cpu-smoke"
                    else config.gpus * (1 if args.action == "prepare" else config.nodes)
                ),
                "settings": config.manifest(),
                "max_timeout_restarts": config.max_restarts,
                "command": config.command(args.action if args.action == "prepare" else "train"),
            },
            indent=2,
        )
    )
    if not args.launch:
        return
    # Optional SDK: a preview is stdlib-only and never hydrates Modal resources.
    import modal

    {"train": _train, "prepare": _prepare, "cpu-smoke": _cpu_smoke}[args.action](modal, config)


if __name__ == "__main__":
    main()
