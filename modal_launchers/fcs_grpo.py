"""Plan or explicitly launch FCS GRPO with a Modal GPU cluster and CPU judge.

Without --launch, this module only prints the resource plan. --check-judge checks
an already running HTTP judge on CPU and never creates Modal resources.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

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


def training_command(config, *, recipe, model_path, run_name, judge_url, nodes, wandb=False):
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
        "trainer.logger=[console,file,wandb]" if wandb else "trainer.logger=[console,file]",
        f"trainer.project_name={json.dumps(os.environ.get('WANDB_PROJECT', 'EasyPPO-FrontierSmith'))}",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={json.dumps(output + '/checkpoints')}",
        f"trainer.rollout_data_dir={json.dumps(output + '/rollouts')}",
        f"trainer.validation_data_dir={json.dumps(output + '/validation')}",
        "trainer.resume_mode=auto",
        "+ray_kwargs.ray_init.address=auto",
    ]


def build_app(args, config):
    """Define SDK objects only. Allocation happens inside launch()."""
    import modal

    app = modal.App("easyppo-fcs-grpo")
    volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)
    image = (
        modal.Image.from_registry(config.image)
        .env(config.container_env())
        .add_local_dir(
            ROOT,
            remote_path=config.source_root,
            ignore=modal.FilePatternMatcher.from_file(ROOT / ".dockerignore"),
        )
        .add_local_dir(ROOT / "data/frontiercs", remote_path=f"{config.source_root}/data/frontiercs")
    )
    judge_root = ROOT / "Frontier-CS/algorithmic"
    judge_image = (
        modal.Image.from_dockerfile(
            judge_root / "Dockerfile",
            context_dir=judge_root,
            ignore=["problems/**", "data/**", "submissions/**", "node_modules/**"],
        )
        .entrypoint([])
        .add_local_dir(judge_root / "problems", remote_path="/app/problems", copy=True)
    )
    secret_keys = [key for key in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT") if os.getenv(key)]
    secrets = [modal.Secret.from_local_environ(secret_keys)] if secret_keys else []
    mounts = {config.volume_mount: volume}

    @app.function(image=image, cpu=2, memory=8192, timeout=86400, volumes=mounts, secrets=secrets, serialized=True)
    def prepare_model():
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=args.model_id, cache_dir=config.volume_path(config.hf_cache_dir) + "/hub"
        )
        volume.commit()
        return model_path

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
    def train(model_path: str, judge_url: str, judge_token: str):
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from modal_launchers.fcs_grpo import check_judge, training_command
        from modal_launchers.multinode import run_node

        # Set this on every raylet before it starts, so reward actors inherit it.
        os.environ["FRONTIERCS_JUDGE_TOKEN"] = judge_token
        os.environ["FRONTIERCS_JUDGE_URL"] = judge_url
        os.environ["VERL_FILE_LOGGER_ROOT"] = (
            config.output_path(args.run_name) + "/metrics/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        check_judge(judge_url, judge_token)
        cluster = modal.Cluster.from_context()
        command = training_command(
            config,
            recipe=args.recipe,
            model_path=model_path,
            run_name=args.run_name,
            judge_url=judge_url,
            nodes=args.nodes,
            wandb=args.wandb,
        )
        run_node(
            SimpleNamespace(rank=cluster.container_rank(), container_ipv4_ips=cluster.container_ips(family="ipv4")),
            run_name=args.run_name,
            model_path=model_path,
            commit=volume.commit,
            config=config,
            nodes=args.nodes,
            command=command,
            project=os.environ.get("WANDB_PROJECT", "EasyPPO-FrontierSmith"),
            require_wandb=args.wandb,
            run_metadata={"recipe": args.recipe, "model": args.model_id},
        )

    return app, prepare_model, train, judge_image


def launch(args, config):
    import modal

    if args.wandb and not os.getenv("WANDB_API_KEY"):
        raise ValueError("--wandb requires WANDB_API_KEY")
    app, prepare_model, train, judge_image = build_app(args, config)
    with app.run():
        # Download once on CPU before allocating the cluster.
        model_path = prepare_model.remote()
        judge = modal.Sandbox.create(
            "/app/entrypoint.sh",
            app=app,
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
        try:
            judge.wait_until_ready(timeout=600)
            credentials = judge.create_connect_token(port=JUDGE_PORT)
            result = check_judge(credentials.url, credentials.token)
            print(f"CPU judge smoke test passed: {json.dumps(result)}", flush=True)
            train.remote(model_path, credentials.url, credentials.token)
        finally:
            judge.terminate()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--launch", action="store_true", help="Allocate the printed CPU/GPU resources and train")
    mode.add_argument("--check-judge", metavar="URL", help="Check an existing judge on CPU; no Modal allocation")
    parser.add_argument("--run-name", default="fcs-drgrpo")
    parser.add_argument("--recipe", choices=RECIPES, default=RECIPES[0])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--gpu", choices=("H200", "B200", "H100"), default="H200")
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4, 8, 16, 32), default=2)
    parser.add_argument("--judge-cpus", type=int, default=32)
    parser.add_argument("--judge-memory-gib", type=int, default=64)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    validate_run_name(args.run_name)
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
                "launch": args.launch,
                "gpu": args.gpu,
                "nodes": args.nodes,
                "gpus_per_node": GPUS_PER_NODE,
                "total_gpus": args.nodes * GPUS_PER_NODE,
                "gpu_node_cpus": 32,
                "gpu_node_memory_gib": 256,
                "model_download": {"cpus": 2, "memory_gib": 8},
                "judge": {
                    "runtime": "vm",
                    "cpus": args.judge_cpus,
                    "memory_gib": args.judge_memory_gib,
                    "workers": args.judge_workers,
                },
                "recipe": args.recipe,
                "model_id": args.model_id,
                "image": config.image,
                "run_name": args.run_name,
                "output": config.output_path(args.run_name),
                "save_freq": 5,
                "max_actor_ckpt_to_keep": 2,
                "wandb": args.wandb,
            },
            indent=2,
        )
    )
    if args.launch:
        launch(args, config)
    else:
        print("Plan only: no resources allocated. Add --launch only after approving this plan.")


if __name__ == "__main__":
    main()
