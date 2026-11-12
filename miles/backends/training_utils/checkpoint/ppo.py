"""Commit and retain complete actor/critic checkpoints for synchronous PPO."""

import logging
import re
import shutil
from pathlib import Path

from miles.backends.training_utils.checkpoint.tracker import (
    CHECKPOINT_TRACKER_FILENAME,
    read_checkpoint_tracker_iteration,
)
from miles.rollout.data_source import compute_global_dataset_state_path
from miles.utils.file_utils import atomic_write_text

logger = logging.getLogger(__name__)

PPO_CHECKPOINT_TRACKER = "latest_complete_ppo_checkpoint.txt"


def _recover_checkpoint(args) -> None:
    if args.finetune or (args.requested_load, args.critic_load) != (args.save, args.critic_save):
        return

    actor_root, critic_root = Path(args.save), Path(args.critic_save)
    tracker = actor_root / PPO_CHECKPOINT_TRACKER
    actor_step = read_checkpoint_tracker_iteration(actor_root)
    critic_step = read_checkpoint_tracker_iteration(critic_root)
    if not tracker.is_file():
        # Adopt a complete checkpoint written before the joint tracker was introduced.
        if actor_step is not None and actor_step == critic_step:
            state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=actor_step))
            if not args.rollout_global_dataset or state_path.is_file():
                atomic_write_text(tracker, f"{actor_step}\n")
        return

    rollout_id = int(tracker.read_text())
    for root, step in ((actor_root, actor_step), (critic_root, critic_step)):
        if step != rollout_id:
            logger.warning("Recovering %s from rollout %s to complete PPO checkpoint %s", root, step, rollout_id)
            atomic_write_text(root / CHECKPOINT_TRACKER_FILENAME, f"{rollout_id}\n")
        for path in root.glob("iter_*"):
            match = re.fullmatch(r"iter_(\d{7,})", path.name)
            if match and int(match[1]) > rollout_id and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)


def finalize_checkpoint(args, rollout_id: int, max_keep: int) -> None:
    roots = (Path(args.save), Path(args.critic_save))
    state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=rollout_id))
    if any(read_checkpoint_tracker_iteration(root) != rollout_id for root in roots) or (
        args.rollout_global_dataset and not state_path.is_file()
    ):
        raise RuntimeError(f"Checkpoint {rollout_id} is incomplete. Skipping checkpoint pruning.")

    atomic_write_text(roots[0] / PPO_CHECKPOINT_TRACKER, f"{rollout_id}\n")
    complete = sorted(
        int(path.name[5:])
        for path in roots[0].glob("iter_*")
        if re.fullmatch(r"iter_\d{7,}", path.name)
        and path.is_dir()
        and (roots[1] / path.name).is_dir()
        and int(path.name[5:]) <= rollout_id
    )
    for previous in complete[:-max_keep]:
        for root in roots:
            path = root / f"iter_{previous:07d}"
            if not path.is_symlink():
                shutil.rmtree(path)
        Path(compute_global_dataset_state_path(args.save, rollout_id=previous)).unlink(missing_ok=True)


def prepare_checkpoint_retention(args) -> int:
    max_keep = getattr(args, "max_keep", 0)
    if max_keep < 0:
        raise ValueError("--max-keep must be non-negative")
    if max_keep:
        if not args.use_critic or args.megatron_config is not None or not args.save or not args.critic_save:
            raise ValueError("--max-keep requires standard Megatron PPO with actor and critic save directories")
        if Path(args.save).resolve() == Path(args.critic_save).resolve():
            raise ValueError("Actor and critic checkpoint directories must differ")
        # Prune only after both models and the rollout state have finished saving.
        args.save_retain_interval = None
        _recover_checkpoint(args)
    return max_keep


def add_checkpoint_arguments(parser):
    parser.add_argument(
        "--max-keep", type=int, default=0,
        help="Maximum complete PPO checkpoint sets to keep (0 keeps all).",
    )
    return parser
