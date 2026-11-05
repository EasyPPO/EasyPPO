"""One-step asynchronous actor-critic on FrontierSmith with Frontier-CS evaluation.

Requires the Miles GPU environment, Docker and a prejoined Ray cluster for
multi-node training. The prepare action downloads the pinned model/data and
converts the Megatron checkpoint. Actor and critic have independent parameters
and support full_detach or actor-summary memory. Official SC/TIS is enabled.

Args:
    --action: prepare, train or all (default).
    --dataset: Fixed to frontiersmith; use the separate TMAX launcher for TMAX.
    --critic-context-mode: full_detach (default) or actor-summary.
    --model-dir / --data-dir / --output-dir: Shared model, task and run paths.
    --no-score-centering: Use the existing asynchronous clipped-PPO baseline.
    --eval-max-tasks / --eval-samples-per-prompt: Full benchmark and five samples by default.

Example (run from the head of a prejoined eight-node Ray cluster):
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \\
        --action train --critic-context-mode full_detach --model-dir /shared/models \\
        --data-dir /shared/data --output-dir /shared/runs
"""

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

from examples.easyppo import async_ppo as _async  # noqa: E402

from miles.utils.external_utils import command_utils as U  # noqa: E402


@dataclass
class ScriptArgs(_async.ScriptArgs):
    dataset: Literal["frontiersmith"] = "frontiersmith"

    def __post_init__(self):
        super().__post_init__()
        _async._validate_dataset(self, dataset="frontiersmith")


def prepare(args: ScriptArgs) -> None:
    _async._prepare(args, dataset="frontiersmith")


def execute(args: ScriptArgs) -> None:
    _async._execute(args, dataset="frontiersmith", script_path=__file__)


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _async._validate_dataset(args, dataset="frontiersmith")
    _async._validate(args)
    if args.action in ("prepare", "all"):
        prepare(args)
    if args.action in ("train", "all"):
        execute(args)


if __name__ == "__main__":
    typer.run(main)
