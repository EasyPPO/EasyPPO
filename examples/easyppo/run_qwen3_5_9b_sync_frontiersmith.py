"""Synchronous FrontierSmith PPO with the pinned Qwen3.5-9B model and Frontier-CS evaluation."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

# examples is not an installed package. File-based invocation must find the
# shared launcher modules independently of the working directory/PYTHONPATH.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from examples.easyppo import sync_ppo as _sync  # noqa: E402
from examples.easyppo.frontiersmith import (
    SYNC_EVAL_SAMPLES_PER_PROMPT,
    SYNC_TERMINAL_ENVIRONMENT,
    Generation9BArgs,
)  # noqa: E402

from miles.utils.external_utils import command_utils as U  # noqa: E402


@dataclass
class ScriptArgs(Generation9BArgs, _sync._SyncScriptArgs):
    dataset: Literal["frontiersmith"] = "frontiersmith"
    terminal_environment: Literal["docker", "daytona", "modal", "shared-judge"] = SYNC_TERMINAL_ENVIRONMENT
    rollout_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.85
    eval_samples_per_prompt: int = SYNC_EVAL_SAMPLES_PER_PROMPT

    def __post_init__(self):
        super().__post_init__()
        _sync._validate_dataset(self, dataset="frontiersmith")


def prepare(args: ScriptArgs) -> None:
    _sync._prepare(args, dataset="frontiersmith")


def execute(args: ScriptArgs) -> None:
    _sync._execute(args, dataset="frontiersmith", script_path=__file__)


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _sync._validate(args, dataset="frontiersmith")
    if args.action in ("prepare", "all"):
        prepare(args)
    if args.action in ("train", "all"):
        execute(args)


if __name__ == "__main__":
    typer.run(main)
