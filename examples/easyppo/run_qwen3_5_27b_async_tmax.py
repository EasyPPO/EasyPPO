"""One-step asynchronous actor-critic on TMAX-15K with Terminal-Bench 2.1 evaluation.

Requires a configured Miles cluster and Harbor Docker access.
Use MILES_SCRIPT_EXTERNAL_RAY=1 for platform-managed Ray; SSH bootstrap is optional.
The prepare action downloads the pinned model/data and
converts the Megatron checkpoint. Actor and critic have independent parameters
and support full_detach or actor-summary memory. Official SC/TIS is enabled.

Args:
    --action: prepare, train or all (default).
    --ray-hostfile: Optional bootstrap hosts to join over root SSH (default: /root/mpi_rack_hostfile).
    --join-ray-workers / --no-join-ray-workers: Automatically join workers (default: on);
        MILES_SCRIPT_EXTERNAL_RAY=1 skips joining an already managed cluster.
    --dataset: Fixed to tmax; use the FrontierSmith launcher for FrontierSmith.
    --critic-context-mode: full_detach (default) or actor-summary.
    --model-dir / --data-dir / --output-dir: Model, task and run paths available in the job environment.
    --no-score-centering: Use the existing asynchronous clipped-PPO baseline.
    --eval-max-tasks / --eval-samples-per-prompt: Full benchmark and five samples by default.

Example (configured eight-node allocation; set path variables for the site):
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_tmax.py \\
        --action train --critic-context-mode actor-summary --model-dir "$MODEL_DIR" \\
        --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
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
    dataset: Literal["tmax"] = "tmax"

    def __post_init__(self):
        super().__post_init__()
        _async._validate_dataset(self, dataset="tmax")


def prepare(args: ScriptArgs) -> None:
    _async._prepare(args, dataset="tmax")


def execute(args: ScriptArgs) -> None:
    _async._execute(args, dataset="tmax", script_path=__file__)


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    _async._validate_dataset(args, dataset="tmax")
    _async._validate(args)
    if args.action in ("prepare", "all"):
        prepare(args)
    if args.action in ("train", "all"):
        execute(args)


if __name__ == "__main__":
    typer.run(main)
