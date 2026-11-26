"""Pure configuration and command construction; importing this module allocates nothing."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.easyppo.frontiersmith import (
    SYNC_EVAL_SAMPLES_PER_PROMPT,
    SYNC_TERMINAL_ENVIRONMENT,
    Generation9BArgs,
    GenerationArgs,
)

from miles.utils.file_utils import atomic_write_text

MODES = ("full_detach", "actor-summary")
TERMINAL_ENVIRONMENTS = ("shared-judge", "docker")
DOCKER_MEMORY_MB = 16384


@dataclass(frozen=True)
class Config:
    mode: str
    run_id: str
    image: str = json.loads((Path(__file__).resolve().parents[1] / "release-lock.json").read_text())["base_image"]
    volume: str = "easyppo-miles"
    mount: str = "/mnt/easyppo"
    source: str = "/workspace/EasyPPO"
    gpu: str = "H200"
    gpus: int = 8
    nodes: int = 1
    memory_gib: int = 1024
    concurrency: int | None = None
    max_restarts: int = 3
    judge_cpus: int = 32
    judge_memory_gib: int = 64
    judge_workers: int = 32
    terminal_environment: str = SYNC_TERMINAL_ENVIRONMENT
    terminal_memory_mb: int | None = None
    eval_max_tasks: int = 0
    eval_samples_per_prompt: int = SYNC_EVAL_SAMPLES_PER_PROMPT
    skip_eval_before_train: bool = False
    num_rollout: int = 100
    num_critic_only_steps: int = 30
    rollout_batch_size: int = 32
    samples_per_prompt: int = 16
    critic_steps_per_rollout: int = 4
    model_size: str = "27b"
    tensor_model_parallel_size: int | None = None
    save_interval: int = 5
    wandb_secret: str = ""
    wandb_entity: str = ""

    def __post_init__(self):
        if self.model_size not in ("9b", "27b"):
            raise ValueError("Choose model size 9b or 27b")
        if self.mode not in MODES or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", self.run_id):
            raise ValueError("Choose a critic mode and a run ID containing only letters, digits, _ or -")
        if self.tensor_model_parallel_size is None:
            object.__setattr__(self, "tensor_model_parallel_size", self.generation.tensor_model_parallel_size)
        if self.gpu not in ("H200", "B200") or self.gpus not in (2, 4, 8):
            raise ValueError("Use H200 or B200 with 2, 4 or 8 GPUs per node")
        if self.tensor_model_parallel_size not in (1, 2, 4) or self.gpus % self.tensor_model_parallel_size:
            raise ValueError("Training TP must be 1, 2 or 4 and divide the GPUs per node")
        if self.nodes not in (1, 2, 4):
            raise ValueError("nodes must be 1, 2 or 4")
        if self.nodes > 1 and self.gpus != 8:
            raise ValueError("Modal multi-node clusters require 8 GPUs per node")
        if self.terminal_environment not in TERMINAL_ENVIRONMENTS:
            raise ValueError("Choose shared-judge or docker as the terminal environment")
        if self.concurrency is None:
            concurrency = 8 if self.terminal_environment == "docker" else self.generation.sglang_server_concurrency
            object.__setattr__(self, "concurrency", concurrency)
        if self.terminal_environment == "docker":
            if self.terminal_memory_mb is None:
                object.__setattr__(self, "terminal_memory_mb", DOCKER_MEMORY_MB)
            elif self.terminal_memory_mb < 1:
                raise ValueError("Sandbox memory must be positive")
        elif self.terminal_memory_mb is not None:
            raise ValueError("--terminal-memory-mb requires --terminal-environment docker")
        if min(self.memory_gib, self.concurrency, self.judge_cpus, self.judge_memory_gib, self.judge_workers) < 1:
            raise ValueError("Resources and concurrency must be positive")
        if self.max_restarts < 0:
            raise ValueError("max-restarts must be non-negative")
        if self.save_interval < 1:
            raise ValueError("save-interval must be positive")
        if self.eval_samples_per_prompt < 1:
            raise ValueError("Evaluation sample count must be positive")
        if self.eval_max_tasks < 0:
            raise ValueError("Evaluation task limit must be nonnegative")
        if self.num_rollout < 1 or not 0 <= self.num_critic_only_steps <= self.num_rollout:
            raise ValueError("Critic-only steps must be between zero and num-rollout")
        if self.rollout_batch_size < 1 or self.samples_per_prompt < 2 or self.critic_steps_per_rollout < 1:
            raise ValueError(
                "Use a positive rollout batch and critic step count, with at least two samples per prompt"
            )
        if self.rollout_batch_size % self.critic_steps_per_rollout:
            raise ValueError("Critic mini-batches must contain complete prompt groups")
        if (self.rollout_batch_size * self.samples_per_prompt // self.critic_steps_per_rollout) % (
            self.nodes * self.gpus // self.tensor_model_parallel_size
        ):
            raise ValueError("Critic mini-batches must divide evenly across training data-parallel ranks")

    @property
    def output(self):
        return f"{self.mount}/outputs/{self.run_id}"

    @property
    def models(self):
        return f"{self.mount}/models"

    @property
    def data(self):
        return f"{self.output}/data"

    @property
    def generation(self):
        return Generation9BArgs if self.model_size == "9b" else GenerationArgs

    @property
    def launcher_name(self):
        return f"run_qwen3_5_{self.model_size}_sync_frontiersmith"

    @property
    def wandb_project(self):
        return f"miles-{self.launcher_name}"

    def command(self, action="train"):
        terminal_flags = ["--terminal-environment", self.terminal_environment]
        if self.terminal_environment == "docker":
            terminal_flags.extend(
                [
                    "--terminal-docker-compose",
                    f"{self.source}/modal_launchers/docker-compose-host.yaml",
                    "--terminal-memory-mb",
                    str(self.terminal_memory_mb),
                ]
            )
        return [
            "python3",
            f"{self.source}/examples/easyppo/{self.launcher_name}.py",
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
            *terminal_flags,
            *(["--skip-eval-before-train"] if self.skip_eval_before_train else []),
            "--num-rollout",
            str(self.num_rollout),
            "--num-critic-only-steps",
            str(self.num_critic_only_steps),
            "--critic-steps-per-rollout",
            str(self.critic_steps_per_rollout),
            "--rollout-batch-size",
            str(self.rollout_batch_size),
            "--samples-per-prompt",
            str(self.samples_per_prompt),
            "--eval-interval",
            "10",
            "--eval-max-tasks",
            str(self.eval_max_tasks),
            "--eval-samples-per-prompt",
            str(self.eval_samples_per_prompt),
            "--save-interval",
            str(self.save_interval),
            "--max-keep",
            "2",
            "--max-context-tokens",
            "131072",
            "--max-new-tokens",
            "24576",
            "--summary-tokens",
            str(self.generation.summary_tokens),
            "--compact-threshold",
            "12000",
            "--max-total-tokens",
            str(self.generation.max_total_tokens),
            "--continue-on-length",
            "--tensor-model-parallel-size",
            str(self.tensor_model_parallel_size),
            "--sglang-server-concurrency",
            str(self.concurrency),
            "--checkpoint-commit-hook",
            "modal_launchers.runtime.commit_volume",
        ]

    def manifest(self):
        # Retry limits and judge endpoint change across attempts; training identity does not.
        return {
            **{key: value for key, value in asdict(self).items() if key != "max_restarts"},
            "hf_repo": self.generation.hf_repo,
            "hf_revision": self.generation.hf_revision,
            "agent_protocol": "fcs-submit-v1",
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preview or launch one synchronous PPO experiment on Modal")
    parser.add_argument("--terminal-environment", choices=TERMINAL_ENVIRONMENTS, default=Config.terminal_environment)
    parser.add_argument("--model-size", choices=("9b", "27b"), default=Config.model_size)
    parser.add_argument("--mode", choices=MODES, default=MODES[0])
    parser.add_argument("--run-id", default="sync-fs-full-detach-001")
    parser.add_argument("--action", choices=("train", "prepare", "cpu-smoke"), default="train")
    parser.add_argument("--launch", action="store_true", help="Allocate the resources printed in the plan")
    for name in ("image", "volume", "gpu", "wandb_secret", "wandb_entity"):
        parser.add_argument(f"--{name.replace('_', '-')}", default=getattr(Config, name))
    for name in (
        "gpus",
        "tensor_model_parallel_size",
        "nodes",
        "memory_gib",
        "concurrency",
        "max_restarts",
        "judge_cpus",
        "judge_memory_gib",
        "judge_workers",
        "terminal_memory_mb",
        "eval_max_tasks",
        "eval_samples_per_prompt",
        "num_rollout",
        "num_critic_only_steps",
        "rollout_batch_size",
        "samples_per_prompt",
        "critic_steps_per_rollout",
        "save_interval",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(Config, name))
    for name in ("skip_eval_before_train",):
        parser.add_argument(
            f"--{name.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=getattr(Config, name)
        )
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
        original = json.loads(path.read_text())
        actual = dict(original)
        actual.setdefault("model_size", "27b")
        actual.setdefault("tensor_model_parallel_size", 4)  # Earlier runs always used TP4.
        actual.setdefault("nodes", 1)  # Existing single-node manifests predate this field.
        # Manifests predating this field used five evaluation draws.
        actual.setdefault("eval_samples_per_prompt", 5)
        # A manifest without a backend used one Docker sandbox per sample.
        actual.setdefault("terminal_environment", "docker")
        if actual["terminal_environment"] == "docker":
            actual.setdefault("terminal_memory_mb", DOCKER_MEMORY_MB)
        actual.setdefault("save_interval", 5)  # Previously fixed by the Modal launcher.
        # Old manifests predate these controls. Their recipe values match the
        # defaults; the new resource defaults also apply to them.
        for name in (
            "eval_max_tasks",
            "skip_eval_before_train",
            "num_rollout",
            "num_critic_only_steps",
            "rollout_batch_size",
            "samples_per_prompt",
            "critic_steps_per_rollout",
        ):
            actual.setdefault(name, getattr(Config, name))
        actual.setdefault("wandb_secret", "")
        actual.setdefault("wandb_entity", "")
        if actual != expected:
            raise ValueError(f"Run configuration differs from {path}; use the original settings or a new run ID")
        # Persist the normalized identity so later default changes cannot
        # silently reinterpret an old experiment.
        if original == actual:
            return
    elif root.exists() and any(root.iterdir()):
        raise ValueError(f"Existing output has no Modal manifest: {root}")
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(expected, indent=2) + "\n")
