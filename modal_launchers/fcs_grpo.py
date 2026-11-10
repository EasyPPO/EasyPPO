"""FCS GRPO with a detached CPU coordinator, GPU cluster, and CPU judge.

Preview only: python -m modal_launchers.fcs_grpo
Launch: modal run --detach modal_launchers/fcs_grpo.py::train --run-name NEW
Resume on timeouts: modal run modal_launchers/fcs_grpo.py::resume --run-name NEW
WANDB_API_KEY is required. Keep the resume entrypoint alive in tmux.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import modal

from modal_launchers.config import load_modal_config
from modal_launchers.multinode import GPUS_PER_NODE, validate_run_name

ROOT = Path(__file__).resolve().parents[1]
RECIPES = (
    "dr_grpo_frontiersmith_no_overlong_filter",
    "dr_grpo_frontiersmith",
    "grpo_frontiersmith_no_overlong_filter",
    "grpo_frontiersmith",
)
MODEL_ID = "HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
MODEL_REVISION = "016efe4f9b8d1f472b412a26bd43e9b3f690a223"
JUDGE_PORT = 8082
JUDGE_SMOKE_CODE = '#include <iostream>\nint main(){int n,m;std::cin>>n>>m;for(int i=0;i<n;i++)std::cout<<1<<"\\n";}'


def check_judge(url, token=None):
    """Check the real compiler/checker/reward path, without importing CUDA or Ray."""
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(url.rstrip("/") + "/health", headers=headers, timeout=30)
    response.raise_for_status()
    if response.json().get("ok") is not True:
        raise RuntimeError("Judge health check failed")
    spec = importlib.util.spec_from_file_location("fcs_reward_check", ROOT / "verl/utils/reward_score/frontiercs.py")
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    result = reward.compute_score(
        "frontiercs", JUDGE_SMOKE_CODE, "frontiercs_val_182", judge_url=url, judge_token=token
    )
    if not 0 < result["score"] <= 1 or abs(result["score"] * 100 - result["raw_score"]) > 1e-8:
        raise RuntimeError(f"Judge compilation/checker/reward smoke test failed: {result}")
    return result


def training_command(config, *, recipe, model_path, run_name, judge_url, nodes):
    validate_run_name(run_name)
    if recipe not in RECIPES:
        raise ValueError(f"Unsupported FCS recipe: {recipe}")
    output = config.output_path(run_name)
    return [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        f"{config.source_root}/configs",
        "--config-name",
        recipe,
        f"actor_rollout_ref.model.path={json.dumps(model_path)}",
        "actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=8",
        f"reward.custom_reward_function.reward_kwargs.judge_url={json.dumps(judge_url)}",
        f"trainer.nnodes={nodes}",
        "trainer.n_gpus_per_node=8",
        "trainer.save_freq=5",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.logger=[console,file,wandb]",
        f"trainer.project_name={json.dumps(os.environ.get('WANDB_PROJECT', 'EasyPPO-FrontierSmith'))}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={json.dumps(output + '/checkpoints')}",
        f"trainer.rollout_data_dir={json.dumps(output + '/rollouts')}",
        f"trainer.validation_data_dir={json.dumps(output + '/validation')}",
        "trainer.resume_mode=auto",
        "+ray_kwargs.ray_init.address=auto",
    ]


def default_model_path(config):
    return config.volume_path(config.hf_cache_dir) + (
        "/hub/models--HuanzhiMao--Qwen3.5-9B-FrontierCS-SFT-DSv3.1/snapshots/" + MODEL_REVISION
    )


def check_model(model_path):
    """Fail on missing or broken cached files before starting the judge or GPUs."""
    model = Path(model_path)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"Missing model file: {model / name}")
    index = model / "model.safetensors.index.json"
    weights = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in weights:
        path = model / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing model weights: {path}")


def run_training(cluster_train, judge_image, args, *, run_name, model_path, recipe):
    """Execute inside the remote CPU coordinator, including detached runs."""
    validate_run_name(run_name)
    if recipe not in RECIPES:
        raise ValueError(f"Unsupported FCS recipe: {recipe}")
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("Export WANDB_API_KEY before starting training")
    check_model(model_path)
    judge = modal.Sandbox.create(
        "/app/entrypoint.sh",
        image=judge_image,
        runtime="vm",
        cpu=args.judge_cpus,
        memory=args.judge_memory_gib * 1024,
        timeout=86400,
        env={
            "PORT": str(JUDGE_PORT),
            "GJ_ADDR": "http://127.0.0.1:5050",
            "JUDGE_WORKERS": str(args.judge_workers),
            "GJ_PARALLELISM": str(args.judge_workers),
            "SAVE_OUTPUTS": "false",
        },
        readiness_probe=modal.Probe.with_tcp(JUDGE_PORT),
    )
    call = None
    completed = False
    try:
        judge.wait_until_ready(timeout=600)
        credentials = judge.create_connect_token(port=JUDGE_PORT)
        result = check_judge(credentials.url, credentials.token)
        print(f"CPU judge smoke test passed: {json.dumps(result)}", flush=True)
        call = cluster_train.spawn(run_name, model_path, recipe, credentials.url, credentials.token)
        call.get()
        completed = True
    finally:
        try:
            if call is not None and not completed:
                call.cancel(terminate_containers=True)
        finally:
            judge.terminate()


def build_app(args, config):
    """Define SDK objects only; neither import nor a plan allocates resources."""
    app = modal.App("easyppo-fcs-grpo")
    volume = modal.Volume.from_name(config.volume_name)
    source_ignore = modal.FilePatternMatcher.from_file(ROOT / ".dockerignore")
    image = (
        modal.Image.from_registry(config.image)
        .env(config.container_env())
        .add_local_dir(ROOT, remote_path=config.source_root, ignore=source_ignore)
        .add_local_dir(ROOT / "data/frontiercs", remote_path=f"{config.source_root}/data/frontiercs")
    )
    judge_root = ROOT / "Frontier-CS/algorithmic"
    # Using this image for the coordinator also builds/hydrates it before the
    # remote Sandbox.create call; no local build context is needed remotely.
    judge_image = (
        modal.Image.from_dockerfile(
            judge_root / "Dockerfile",
            context_dir=judge_root,
            ignore=["problems/**", "data/**", "submissions/**", "node_modules/**"],
            add_python="3.12",
        )
        .entrypoint([])
        .pip_install("requests==2.34.2", "PyYAML==6.0.3")
        .env(config.container_env())
        .add_local_dir(judge_root / "problems", remote_path="/app/problems", copy=True)
        .add_local_dir(ROOT, remote_path=config.source_root, ignore=source_ignore, copy=True)
    )
    wandb_keys = ["WANDB_API_KEY"] + [key for key in ("WANDB_ENTITY", "WANDB_PROJECT") if os.getenv(key)]
    secrets = [modal.Secret.from_local_environ(wandb_keys)] if os.getenv("WANDB_API_KEY") else []
    mounts = {config.volume_mount: volume}

    @app.function(
        image=image,
        gpu=f"{args.gpu}:{GPUS_PER_NODE}",
        cpu=32,
        memory=262144,
        timeout=86400,
        retries=0,
        single_use_containers=True,
        max_containers=args.nodes,
        volumes=mounts,
        secrets=secrets,
        serialized=True,
    )
    @modal.clustered(size=args.nodes, rdma=True)
    def cluster_train(run_name: str, model_path: str, recipe: str, judge_url: str, judge_token: str):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from modal_launchers.fcs_grpo import check_judge, training_command
        from modal_launchers.multinode import run_node

        os.environ["FRONTIERCS_JUDGE_TOKEN"] = judge_token
        os.environ["FRONTIERCS_JUDGE_URL"] = judge_url
        os.environ["VERL_FILE_LOGGER_ROOT"] = (
            config.output_path(run_name) + "/metrics/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        check_judge(judge_url, judge_token)
        cluster = modal.Cluster.from_context()
        command = training_command(
            config, recipe=recipe, model_path=model_path, run_name=run_name, judge_url=judge_url, nodes=args.nodes
        )
        run_node(
            SimpleNamespace(rank=cluster.container_rank(), container_ipv4_ips=cluster.container_ips(family="ipv4")),
            run_name=run_name,
            model_path=model_path,
            commit=volume.commit,
            config=config,
            nodes=args.nodes,
            command=command,
            project=os.environ.get("WANDB_PROJECT", "EasyPPO-FrontierSmith"),
            require_wandb=True,
            run_metadata={"recipe": recipe, "model": model_path},
        )

    @app.function(
        image=judge_image,
        cpu=1,
        memory=2048,
        timeout=86400,
        retries=0,
        single_use_containers=True,
        volumes=mounts,
        secrets=secrets,
        serialized=True,
    )
    def train(run_name: str, model_path: str = default_model_path(config), recipe: str = RECIPES[0]):
        from modal_launchers.fcs_grpo import run_training

        run_training(cluster_train, judge_image, args, run_name=run_name, model_path=model_path, recipe=recipe)

    return app, train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-judge", metavar="URL", help="Check an existing judge on CPU; no Modal allocation")
    parser.add_argument("--run-name", default="fcs-drgrpo")
    parser.add_argument("--recipe", choices=RECIPES, default=RECIPES[0])
    parser.add_argument("--model-path", default=default_model_path(load_modal_config()))
    parser.add_argument("--gpu", choices=("H200", "B200", "H100"), default=os.getenv("EASYPPO_FCS_GPU", "H200"))
    parser.add_argument(
        "--nodes", type=int, choices=(1, 2, 4, 8, 16, 32), default=int(os.getenv("EASYPPO_FCS_NODES", "2"))
    )
    parser.add_argument("--judge-cpus", type=int, default=int(os.getenv("EASYPPO_FCS_JUDGE_CPUS", "32")))
    parser.add_argument("--judge-memory-gib", type=int, default=int(os.getenv("EASYPPO_FCS_JUDGE_MEMORY_GIB", "64")))
    parser.add_argument("--judge-workers", type=int, default=int(os.getenv("EASYPPO_FCS_JUDGE_WORKERS", "32")))
    args = parser.parse_args(argv)
    validate_run_name(args.run_name)
    if args.gpu not in ("H200", "B200", "H100") or args.nodes not in (1, 2, 4, 8, 16, 32):
        parser.error("unsupported GPU or node count")
    if min(args.judge_cpus, args.judge_memory_gib, args.judge_workers) < 1:
        parser.error("judge resources must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check_judge:
        print(json.dumps(check_judge(args.check_judge, os.getenv("FRONTIERCS_JUDGE_TOKEN")), indent=2))
        return
    config = load_modal_config()
    print(
        json.dumps(
            {
                "launch": False,
                "gpu": args.gpu,
                "nodes": args.nodes,
                "gpus_per_node": GPUS_PER_NODE,
                "total_gpus": args.nodes * GPUS_PER_NODE,
                "gpu_node_cpus": 32,
                "gpu_node_memory_gib": 256,
                "coordinator": {"cpus": 1, "memory_gib": 2},
                "judge": {
                    "runtime": "vm",
                    "cpus": args.judge_cpus,
                    "memory_gib": args.judge_memory_gib,
                    "workers": args.judge_workers,
                },
                "recipe": args.recipe,
                "model_path": args.model_path,
                "image": config.image,
                "run_name": args.run_name,
                "output": config.output_path(args.run_name),
                "save_freq": 5,
                "max_actor_ckpt_to_keep": 2,
                "wandb": True,
                "resume_max_timeout_restarts": 10,
            },
            indent=2,
        )
    )
    print("Plan only: no resources allocated. Use the documented Modal train/resume commands after approval.")


# Serialized remote functions capture their hydrated SDK objects. Helpers can
# be imported remotely without rebuilding local images or reading local files.
if modal.is_local() and __name__ != "__main__":
    app, train = build_app(parse_args([]), load_modal_config())

    @app.local_entrypoint()
    def resume(
        run_name: str,
        model_path: str = default_model_path(load_modal_config()),
        recipe: str = RECIPES[0],
        max_timeout_restarts: int = 10,
    ):
        """Only Function timeouts start another attempt, with a fresh CPU judge."""
        from modal.exception import FunctionTimeoutError

        validate_run_name(run_name)
        if recipe not in RECIPES:
            raise ValueError(f"Unsupported FCS recipe: {recipe}")
        if max_timeout_restarts < 0:
            raise ValueError("max_timeout_restarts must be non-negative")
        if not os.getenv("WANDB_API_KEY"):
            raise ValueError("Export WANDB_API_KEY before starting training")
        for attempt in range(max_timeout_restarts + 1):
            try:
                train.remote(run_name=run_name, model_path=model_path, recipe=recipe)
                return
            except FunctionTimeoutError:
                if attempt == max_timeout_restarts:
                    raise
                print(f"Function timeout: continuing {run_name} ({attempt + 1}/{max_timeout_restarts})", flush=True)


if __name__ == "__main__":
    main()
