from pathlib import Path


def compute_rollout_checkpoint_dir(directory: str | Path, *, rollout_id: int) -> Path:
    return Path(directory) / "rollout" / str(rollout_id)
