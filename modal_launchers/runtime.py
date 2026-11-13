"""Container operations. Heavy dependencies are imported only by remote entrypoints."""

import json
import os
import subprocess
from pathlib import Path

from modal_launchers.config import check_manifest

from miles.utils.file_utils import atomic_write_text


def commit_volume():
    # The hook runs in the training driver, inside the mounted Modal container.
    import modal

    modal.Volume.from_name(os.environ["EASYPPO_MODAL_VOLUME"]).commit()


def prepare_run(config, judge_url):
    check_manifest(
        config,
        wandb_project="miles-run_qwen3_5_27b_sync_frontiersmith",
        wandb_entity=os.environ.get("WANDB_ENTITY"),
    )
    subprocess.run(
        [
            "python3",
            f"{config.source}/scripts/make_miles_frontiersmith_data.py",
            "--source-dir",
            f"{config.source}/Frontier-CS",
            "--tasks-dir",
            f"{config.data}/harbor",
            "--out",
            config.data,
            "--judge-url",
            judge_url,
        ],
        check=True,
    )
    commit_volume()


def check_models(config):
    model = Path(config.models) / "Qwen3.5-27B"
    converted = Path(config.models) / "Qwen3.5-27B_torch_dist"
    for path in (
        model / "config.json",
        model / "tokenizer_config.json",
        converted / "latest_checkpointed_iteration.txt",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run --action prepare first")
    step = (converted / "latest_checkpointed_iteration.txt").read_text().strip()
    if step != "release" or not (converted / "release").is_dir():
        raise ValueError("Base conversion must be a completed Megatron release checkpoint")
    index = model / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if any(not (model / shard).is_file() for shard in shards):
        raise FileNotFoundError("Model weights are incomplete")


def train(config, judge_url, judge_token):
    os.chdir(config.source)
    os.environ.update(FRONTIERCS_JUDGE_TOKEN=judge_token, WANDB_RUN_ID=config.run_id, WANDB_RESUME="allow")
    # Ray inherits the credentials from its parent. Never serialize API keys in
    # --extra-env-vars, since the framework logs that command.
    command = config.command()
    command += [
        "--frontier-judge-url",
        judge_url,
        "--extra-env-vars",
        json.dumps(
            {
                "WANDB_RUN_ID": config.run_id,
                "WANDB_RESUME": "allow",
                "EASYPPO_MODAL_VOLUME": config.volume,
            }
        ),
    ]
    try:
        subprocess.run(command, check=True)
        atomic_write_text(Path(config.output, "training_complete.json"), json.dumps({"rounds": 100}) + "\n")
    finally:
        commit_volume()
