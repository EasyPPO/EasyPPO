"""Shared synchronous terminal PPO launch helpers; no separate training algorithm."""

import shlex
from dataclasses import dataclass
from typing import Literal

from examples.easyppo import async_ppo as _recipe

from miles.utils.external_utils import command_utils as U


@dataclass
class _SyncScriptArgs(_recipe.ScriptArgs):
    num_nodes: int = 4
    trainer_num_nodes: int = 1
    score_centering: bool = False
    score_centering_candidates: int = 0
    score_centering_is: Literal["none"] = "none"
    score_centering_recompute_loss: bool = False

    def __post_init__(self):
        super().__post_init__()
        _validate_sync_options(self)


def _validate_sync_options(args: _SyncScriptArgs) -> None:
    if args.score_centering:
        raise ValueError("Synchronous PPO launchers do not support --score-centering")
    if (
        args.score_centering_candidates != 0
        or args.score_centering_is != "none"
        or args.score_centering_recompute_loss
    ):
        raise ValueError("Synchronous PPO requires score-centering candidates=0, is=none and recompute-loss disabled")
    # These inherited CLI fields remain for dataclass compatibility, but no SC
    # consumer exists here. Reject overrides instead of silently ignoring them.
    for name in (
        "score_centering_tis_clip",
        "score_centering_mis_low",
        "score_centering_mis_high",
        "score_centering_chunk_size",
    ):
        if getattr(args, name) != getattr(_recipe.ScriptArgs, name):
            raise ValueError(f"Synchronous PPO does not use --{name.replace('_', '-')}; leave its recipe default")


def _validate_dataset(args: _SyncScriptArgs, *, dataset: str) -> None:
    if args.dataset != dataset:
        raise ValueError(f"This synchronous launcher requires --dataset {dataset}")


def _validate(args: _SyncScriptArgs, *, dataset: str) -> None:
    _validate_dataset(args, dataset=dataset)
    _validate_sync_options(args)
    _recipe._validate(args)


def _prepare(args: _SyncScriptArgs, *, dataset: str) -> None:
    _validate(args, dataset=dataset)
    _recipe._prepare(args, dataset=dataset)


def _execute(args: _SyncScriptArgs, *, dataset: str, script_path: str) -> None:
    _validate(args, dataset=dataset)
    argv = (
        _recipe._checkpoint_args(args)
        + _recipe._rollout_args(args)
        + _recipe._terminal_args(args)
        # train.py finishes rollout before updating the actor. Its current
        # forward supplies the PPO denominator, so no old-actor copy is needed.
        + _recipe._algorithm_args(args, keep_old_actor=False)
        + _recipe._resource_args(args)
        + shlex.split(U.get_default_wandb_args(script_path, run_id=args.run_id))
    )
    backend = args.create_backend()
    backend.execute_train(
        train_args=shlex.join(argv),
        train_script="train.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-27B",
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
        before_ray_job_submit=_recipe._ray_worker_startup(args, backend),
    )
