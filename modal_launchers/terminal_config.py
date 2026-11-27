"""Modal infrastructure for the canonical Terminus recipe in Claude.md."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.easyppo.terminus_ppo import ScriptArgs


@dataclass(frozen=True)
class TerminalConfig:
    run_id: str
    mode: str = "full_detach"
    nodes: int = ScriptArgs.num_nodes
    gpus: int = ScriptArgs.num_gpus_per_node
    gpu: str = "B200"
    memory_gib: int = 1024
    volume: str = "easyppo-terminal-runs"
    assets_volume: str = "easyppo-terminal-assets-v2"
    mount: str = "/mnt/easyppo"
    output_mount: str = "/mnt/terminal-runs"
    source: str = "/workspace/EasyPPO"
    wandb_secret: str = "wandb-personal"
    wandb_entity: str = "nan-playground"
    max_keep: int = ScriptArgs.max_keep
    # Preemptible Modal jobs publish a recovery point after every completed batch.
    save_interval: int = 1
    data_revision: str = "e2215165c"
    image: str = json.loads((Path(__file__).resolve().parents[1] / "release-lock.json").read_text())["base_image"]

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.run_id):
            raise ValueError("Use a run ID containing letters, digits, underscores and dashes")
        if self.mode not in ("full_detach", "actor-summary"):
            raise ValueError("Choose full_detach or actor-summary")
        if self.nodes != 2 or self.gpus != 8 or self.gpu not in ("B200", "H200"):
            raise ValueError("This Modal launcher validates the two-node, eight-GPU topology")
        if self.max_keep < 1 or self.memory_gib < 1 or self.save_interval < 1:
            raise ValueError("Resources, checkpoint retention and save interval must be positive")

    @property
    def terminal_environment(self):
        return ScriptArgs.terminal_environment

    @property
    def generation(self):
        return ScriptArgs

    @property
    def models(self):
        return f"{self.mount}/models"

    @property
    def data(self):
        return f"{self.mount}/data/terminal-agentpick-{self.data_revision}"

    @property
    def prompt_data(self):
        return f"{self.data}/harbor/agentpick-802/train.jsonl"

    @property
    def output(self):
        return f"{self.output_mount}/{self.run_id}"

    @property
    def sandbox_app(self):
        return f"easyppo-tasks-{self.run_id}"

    @property
    def wandb_project(self):
        return "miles-terminus_ppo"

    def manifest(self):
        recipe = ScriptArgs(num_nodes=self.nodes, num_gpus_per_node=self.gpus, critic_context_mode=self.mode)
        return {
            **asdict(self),
            "recipe": {
                key: getattr(recipe, key)
                for key in (
                    "hf_repo",
                    "hf_revision",
                    "trainer_num_nodes",
                    "trainer_gpus_per_node",
                    "rollout_num_gpus",
                    "eval_num_gpus",
                    "num_rollout",
                    "rollout_batch_size",
                    "samples_per_prompt",
                    "num_critic_only_steps",
                    "max_tokens_per_gpu",
                    "critic_max_tokens_per_gpu",
                )
            },
        }

    def command(self):
        return [
            "python3",
            "-m",
            "examples.easyppo.terminus_ppo",
            "--critic-context-mode",
            self.mode,
            "--run-id",
            self.run_id,
            "--num-nodes",
            str(self.nodes),
            "--num-gpus-per-node",
            str(self.gpus),
            "--model-dir",
            self.models,
            "--output-dir",
            self.output_mount,
            "--tasks-dir",
            f"{self.data}/harbor",
            "--prompt-data",
            self.prompt_data,
            "--eval-data",
            f"{self.data}/tb21.eval.jsonl",
            "--gpu-arch",
            "blackwell" if self.gpu == "B200" else "hopper",
            "--max-keep",
            str(self.max_keep),
            "--save-interval",
            str(self.save_interval),
            "--checkpoint-commit-hook",
            "modal_launchers.runtime.commit_volume",
            "--trials-dir",
            f"{self.mount}/outputs/{self.run_id}/trials",
        ]
