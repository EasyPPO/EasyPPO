import logging
from pathlib import Path

import torch

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if args.save_debug_train_data is not None:
        parallel_state = get_parallel_state()
        # TP peers duplicate the DP shard and only the last PP stage computes the per-token fields
        if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
            return
        save_debug_train_data_for_rank(
            args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
        )


def save_debug_train_data_for_rank(args, *, rollout_id, rollout_data, rank, cp_rank=0, cp_size=1):
    """Write one rank's slice of a rollout.

    ``cp_rank`` / ``cp_size`` are passed in for the same reason ``rank`` is:
    this runs outside the process group in tests and offline tools.
    """
    if (path_template := args.save_debug_train_data) is not None:
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
                cp_rank=cp_rank,
                cp_size=cp_size,
                qkv_format=args.qkv_format,
            ),
            path,
        )


CRITIC_DUMP_KEYS = ("values", "returns", "rewards", "loss_masks", "response_lengths", "sample_indices", "rollout_ids", "value_slices")


def save_debug_critic_values(args, *, rollout_id, rollout_data, rank=None):
    """Write the critic's predictions on a rollout, taken BEFORE the critic trains on it.

    With ``--save-debug-train-data`` set, the file goes next to the actor's dump with a ``.critic.pt`` suffix. The
    values are those of the critic's own samples (one per rollout under ``--critic-context-mode full_history``);
    ``value_slices`` says how they are cut into the actor's samples. Since the dump precedes the optimiser steps,
    the predictions are on data the critic has not fitted.
    """
    if (path_template := args.save_debug_train_data) is None:
        return None
    if rank is None:
        parallel_state = get_parallel_state()
        if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
            return None
        rank = torch.distributed.get_rank()
    path = Path(path_template.format(rollout_id=rollout_id, rank=rank)).with_suffix(".critic.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rollout_id": rollout_id, "rank": rank}
    for key in CRITIC_DUMP_KEYS:
        if key in rollout_data:
            value = rollout_data[key]
            payload[key] = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in value] if isinstance(value, list) else value
    logger.info(f"Save critic values to {path}")
    torch.save(payload, path)
    return path
