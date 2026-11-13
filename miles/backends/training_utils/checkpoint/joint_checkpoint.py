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
from miles.utils.function_registry import load_function

logger = logging.getLogger(__name__)

PPO_CHECKPOINT_TRACKER = "latest_complete_ppo_checkpoint.txt"


def _checkpoint_exists(args, rollout_id: int) -> bool:
    return all((Path(root) / f"iter_{rollout_id:07d}").is_dir() for root in (args.save, args.critic_save)) and (
        not args.rollout_global_dataset
        or Path(compute_global_dataset_state_path(args.save, rollout_id=rollout_id)).is_file()
    )


def recover_checkpoint(args) -> None:
    """Reconcile trackers BEFORE resolving fresh-run load/finetune defaults."""
    if not getattr(args, "max_keep", 0):
        return
    if args.critic_save is None and args.save:
        args.critic_save = args.save.rstrip("/") + "_critic"
    if args.critic_load is None:
        args.critic_load = args.critic_save if args.load == args.save else args.load
    _validate_retention(args)
    if args.finetune or (args.load, args.critic_load) != (args.save, args.critic_save):
        return

    actor_root, critic_root = Path(args.save), Path(args.critic_save)
    tracker = actor_root / PPO_CHECKPOINT_TRACKER
    actor_step = read_checkpoint_tracker_iteration(actor_root)
    critic_step = read_checkpoint_tracker_iteration(critic_root)
    if not tracker.is_file():
        if actor_step is not None and actor_step == critic_step and _checkpoint_exists(args, actor_step):
            rollout_id = actor_step
        elif any(root.exists() and any(root.iterdir()) for root in (actor_root, critic_root)):
            raise RuntimeError(
                "Existing PPO checkpoints have no complete joint checkpoint; inspect them before resuming"
            )
        else:
            # Persist the empty transaction before the first model save. A crash
            # during the first save can then unambiguously restart from the base.
            rollout_id = -1
        actor_root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(tracker, f"{rollout_id}\n")
        _commit_storage(args)
    else:
        rollout_id = int(tracker.read_text())
    if rollout_id < -1 or (rollout_id >= 0 and not _checkpoint_exists(args, rollout_id)):
        raise RuntimeError(f"Complete PPO checkpoint {rollout_id} is missing model or rollout state")

    for root, step in ((actor_root, actor_step), (critic_root, critic_step)):
        if rollout_id == -1:
            (root / CHECKPOINT_TRACKER_FILENAME).unlink(missing_ok=True)
        elif step != rollout_id:
            logger.warning("Recovering %s from rollout %s to complete PPO checkpoint %s", root, step, rollout_id)
            atomic_write_text(root / CHECKPOINT_TRACKER_FILENAME, f"{rollout_id}\n")
        for path in root.glob("iter_*"):
            match = re.fullmatch(r"iter_(\d{7,})", path.name)
            if match and int(match[1]) > rollout_id and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
    for path in (actor_root / "rollout").glob("global_dataset_state_dict_*.pt"):
        match = re.fullmatch(r"global_dataset_state_dict_(\d+)\.pt", path.name)
        if match and int(match[1]) > rollout_id:
            path.unlink()


def _commit_storage(args) -> None:
    if hook := getattr(args, "checkpoint_commit_hook", None):
        load_function(hook)()


def finalize_checkpoint(args, rollout_id: int, max_keep: int) -> None:
    roots = (Path(args.save), Path(args.critic_save))
    state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=rollout_id))
    if any(read_checkpoint_tracker_iteration(root) != rollout_id for root in roots) or (
        args.rollout_global_dataset and not state_path.is_file()
    ):
        raise RuntimeError(f"Checkpoint {rollout_id} is incomplete. Skipping checkpoint pruning.")

    if not _checkpoint_exists(args, rollout_id):
        raise RuntimeError(f"Checkpoint {rollout_id} is missing model or rollout state")
    # Durably publish payloads before the completion marker, and the marker
    # before deleting the previous recovery point (e.g. Modal Volume commits).
    _commit_storage(args)
    atomic_write_text(roots[0] / PPO_CHECKPOINT_TRACKER, f"{rollout_id}\n")
    _commit_storage(args)
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


def _validate_retention(args) -> int:
    max_keep = getattr(args, "max_keep", 0)
    if getattr(args, "checkpoint_commit_hook", None) and not max_keep:
        raise ValueError("--checkpoint-commit-hook requires --max-keep > 0")
    if max_keep < 0:
        raise ValueError("--max-keep must be non-negative")
    if max_keep:
        if (
            not (getattr(args, "use_critic", False) or getattr(args, "advantage_estimator", None) == "ppo")
            or args.megatron_config is not None
            or not args.save
            or not args.critic_save
        ):
            raise ValueError("--max-keep requires standard Megatron PPO with actor and critic save directories")
        if Path(args.save).resolve() == Path(args.critic_save).resolve():
            raise ValueError("Actor and critic checkpoint directories must differ")
        # Prune only after both models and the rollout state have finished saving.
        args.save_retain_interval = None
    return max_keep


def prepare_checkpoint_retention(args) -> int:
    return _validate_retention(args)


def add_checkpoint_arguments(parser):
    parser.add_argument(
        "--max-keep",
        type=int,
        default=0,
        help="Maximum complete PPO checkpoint sets to keep (0 keeps all).",
    )
    parser.add_argument(
        "--checkpoint-commit-hook",
        default=None,
        help="Zero-argument storage commit function, called before and after publishing a joint PPO checkpoint.",
    )
    return parser
