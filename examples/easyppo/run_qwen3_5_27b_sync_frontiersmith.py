"""Synchronous clipped PPO on FrontierSmith with Frontier-CS evaluation.

Requires a shared Frontier judge, a configured Miles cluster, and Harbor Docker access.
Use MILES_SCRIPT_EXTERNAL_RAY=1 for platform-managed Ray; SSH bootstrap is optional.
The prepare action downloads the pinned model, prepares the bundled data and
converts the Megatron checkpoint. Actor and critic have independent parameters
and support full_detach or actor-summary memory. Score centering is disabled.
Rollout, critic, and actor alternate on all allocated GPUs with CPU offload.

Args:
    --action: prepare, train or all (default).
    --num-nodes: Shared rollout/training allocation (default: 4 nodes, 8 GPUs each).
    --ray-hostfile: Optional bootstrap hosts to join over root SSH (default: /root/mpi_rack_hostfile).
    --join-ray-workers / --no-join-ray-workers: Automatically join workers (default: on);
        MILES_SCRIPT_EXTERNAL_RAY=1 skips joining an already managed cluster.
    --dataset: Fixed to frontiersmith; use the separate TMAX launcher for TMAX.
    --critic-context-mode: full_detach (default) or actor-summary.
    --model-dir / --data-dir / --output-dir: Model, task and run paths available in the job environment.
    --frontier-judge-url / --frontier-judge-timeout: Shared judge endpoint and queue-inclusive timeout baked into prepared tasks.
    --score-centering: Rejected; this entrypoint uses ordinary clipped PPO.
    --eval-max-tasks / --eval-samples-per-prompt: Full benchmark and five samples by default.

Example (configured four-node allocation; set path variables for the site):
    MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \\
        --action train --critic-context-mode full_detach --model-dir "$MODEL_DIR" \\
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

from examples.easyppo import sync_ppo as _sync  # noqa: E402

from miles.utils.external_utils import command_utils as U  # noqa: E402


@dataclass
class ScriptArgs(_sync._SyncScriptArgs):
    dataset: Literal["frontiersmith"] = "frontiersmith"

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
