"""Pure configuration and command construction; importing this module allocates nothing."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MODES = ("full_detach", "actor-summary")


@dataclass(frozen=True)
class Config:
    mode: str
    run_id: str
    image: str = "radixark/miles@sha256:af9b05947486d92ab01d29d3c50fe9dc301becfb932eacaa3dd4c92eea943663"
    volume: str = "easyppo-miles"
    mount: str = "/mnt/easyppo"
    source: str = "/workspace/EasyPPO"
    gpu: str = "H200"
    gpus: int = 8
    nodes: int = 1
    memory_gib: int = 1024
    concurrency: int = 128
    max_restarts: int = 3
    judge_cpus: int = 32
    judge_memory_gib: int = 64
    judge_workers: int = 32

    def __post_init__(self):
        if self.mode not in MODES or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.run_id):
            raise ValueError("Choose a critic mode and a run ID containing only letters, digits, _ or -")
        if self.gpu not in ("H200", "B200") or self.gpus not in (4, 8):
            raise ValueError("Use H200 or B200 with 4 or 8 GPUs per node")
        if self.nodes not in (1, 2, 4):
            raise ValueError("nodes must be 1, 2 or 4")
        if self.nodes > 1 and self.gpus != 8:
            raise ValueError("Modal multi-node clusters require 8 GPUs per node")
        if min(self.memory_gib, self.concurrency, self.judge_cpus, self.judge_memory_gib, self.judge_workers) < 1:
            raise ValueError("Resources and concurrency must be positive")
        if self.max_restarts < 0:
            raise ValueError("max-restarts must be non-negative")

    @property
    def output(self):
        return f"{self.mount}/outputs/{self.run_id}"

    @property
    def models(self):
        return f"{self.mount}/models"

    @property
    def data(self):
        return f"{self.output}/data"

    def command(self, action="train"):
        return [
            "python3",
            f"{self.source}/examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py",
            "--action",
            action,
            "--num-nodes",
            str(self.nodes if action == "train" else 1),
            "--num-gpus-per-node",
            str(self.gpus),
            "--no-join-ray-workers",
            "--run-id",
            self.run_id,
            "--critic-context-mode",
            self.mode,
            "--model-dir",
            self.models,
            "--data-dir",
            self.data,
            "--output-dir",
            f"{self.mount}/outputs",
            "--terminal-environment",
            "modal",
            "--num-rollout",
            "100",
            "--num-critic-only-steps",
            "30",
            "--critic-steps-per-rollout",
            "4",
            "--rollout-batch-size",
            "32",
            "--samples-per-prompt",
            "16",
            "--eval-interval",
            "10",
            "--eval-max-tasks",
            "0",
            "--eval-samples-per-prompt",
            "5",
            "--save-interval",
            "5",
            "--max-keep",
            "2",
            "--max-context-tokens",
            "131072",
            "--max-new-tokens",
            "24576",
            "--summary-tokens",
            "3072",
            "--compact-threshold",
            "12000",
            "--max-total-tokens",
            "120000",
            "--continue-on-length",
            "--tensor-model-parallel-size",
            "4",
            "--rollout-gpus-per-engine",
            "4",
            "--sglang-server-concurrency",
            str(self.concurrency),
            "--checkpoint-commit-hook",
            "modal_launchers.runtime.commit_volume",
        ]

    def manifest(self):
        # Retry limits and judge endpoint change across attempts; training identity does not.
        return {key: value for key, value in asdict(self).items() if key != "max_restarts"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preview or launch one synchronous PPO experiment on Modal")
    parser.add_argument("--mode", choices=MODES, default=MODES[0])
    parser.add_argument("--run-id", default="sync-fs-full-detach-001")
    parser.add_argument("--action", choices=("train", "prepare", "cpu-smoke"), default="train")
    parser.add_argument("--launch", action="store_true", help="Allocate the resources printed in the plan")
    for name in ("image", "volume", "gpu"):
        parser.add_argument(f"--{name}", default=getattr(Config, name))
    for name in (
        "gpus",
        "nodes",
        "memory_gib",
        "concurrency",
        "max_restarts",
        "judge_cpus",
        "judge_memory_gib",
        "judge_workers",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(Config, name))
    args = parser.parse_args(argv)
    fields = vars(args).copy()
    fields.pop("action")
    fields.pop("launch")
    config = Config(**fields)
    return args, config


def check_manifest(config, *, wandb_project, wandb_entity):
    root = Path(config.output)
    path = root / "modal_run.json"
    expected = dict(config.manifest(), wandb_project=wandb_project, wandb_entity=wandb_entity)
    if path.exists():
        actual = json.loads(path.read_text())
        actual.setdefault("nodes", 1)  # Existing single-node manifests predate this field.
        if actual != expected:
            raise ValueError(f"Run configuration differs from {path}; use the original settings or a new run ID")
    elif root.exists() and any(root.iterdir()):
        raise ValueError(f"Existing output has no Modal manifest: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(expected, indent=2) + "\n")
        temp.replace(path)
