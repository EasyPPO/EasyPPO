"""Crash recovery and publication ordering for synchronous PPO checkpoint sets."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.checkpoint import joint_checkpoint as joint
from miles.backends.training_utils.checkpoint.tracker import CHECKPOINT_TRACKER_FILENAME as TRACKER
from miles.rollout.data_source import compute_global_dataset_state_path


def _args(tmp_path):
    actor, critic = str(tmp_path / "actor"), str(tmp_path / "critic")
    return SimpleNamespace(
        save=actor,
        load=actor,
        requested_load=actor,
        critic_save=critic,
        critic_load=critic,
        finetune=False,
        max_keep=2,
        use_critic=True,
        megatron_config=None,
        rollout_global_dataset=True,
        checkpoint_commit_hook=None,
    )


def _model(root, step):
    root = Path(root)
    (root / f"iter_{step:07d}").mkdir(parents=True, exist_ok=True)
    (root / TRACKER).write_text(str(step))


def _save(args, step):
    _model(args.save, step)
    _model(args.critic_save, step)
    path = Path(compute_global_dataset_state_path(args.save, rollout_id=step))
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"closed-rollout-state")


@pytest.mark.parametrize("partial", ["actor", "both", "rollout"])
def test_interruption_during_first_save_restarts_base(tmp_path, partial):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _model(args.save, 4)
    if partial in ("both", "rollout"):
        _model(args.critic_save, 4)
    if partial == "rollout":
        _save(args, 4)
    joint.recover_checkpoint(args)
    assert not (Path(args.save) / TRACKER).exists()
    assert not (Path(args.critic_save) / TRACKER).exists()
    assert not list(Path(args.save).glob("iter_*"))
    assert not list(Path(args.save).glob("rollout/*.pt"))


def test_rollback_keeps_matching_state_and_prunes_only_after_commit(tmp_path, monkeypatch):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    for step in (4, 9):
        _save(args, step)
        joint.finalize_checkpoint(args, step, 2)
    _save(args, 14)
    commits = []

    def commit(_):
        commits.append(
            (
                Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip(),
                sorted(p.name for p in Path(args.save).glob("iter_*")),
            )
        )

    monkeypatch.setattr(joint, "_commit_storage", commit)
    joint.finalize_checkpoint(args, 14, 2)
    assert [c[0] for c in commits] == ["9", "14"]
    assert all("iter_0000004" in c[1] for c in commits)
    assert not Path(args.save, "iter_0000004").exists()
    _model(args.save, 19)
    joint.recover_checkpoint(args)
    assert Path(args.save, TRACKER).read_text().strip() == "14"
    assert not Path(args.save, "iter_0000019").exists()
    assert Path(compute_global_dataset_state_path(args.save, rollout_id=14)).exists()


@pytest.mark.parametrize("missing", ["actor", "critic", "rollout"])
def test_incomplete_set_never_published_or_pruned(tmp_path, missing):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _save(args, 4)
    joint.finalize_checkpoint(args, 4, 2)
    _save(args, 9)
    if missing == "rollout":
        Path(compute_global_dataset_state_path(args.save, rollout_id=9)).unlink()
    else:
        root = args.save if missing == "actor" else args.critic_save
        Path(root, "iter_0000009").rmdir()
    with pytest.raises(RuntimeError):
        joint.finalize_checkpoint(args, 9, 2)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "4"
    assert Path(args.save, "iter_0000004").exists()


def test_missing_committed_payload_is_fatal(tmp_path):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _save(args, 4)
    joint.finalize_checkpoint(args, 4, 2)
    Path(args.critic_save, "iter_0000004").rmdir()
    with pytest.raises(RuntimeError, match="missing model or rollout"):
        joint.recover_checkpoint(args)


def test_complete_legacy_set_is_adopted(tmp_path):
    args = _args(tmp_path)
    _save(args, 4)
    joint.recover_checkpoint(args)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "4"


def test_partial_legacy_set_is_not_silently_discarded(tmp_path):
    args = _args(tmp_path)
    _model(args.save, 4)
    with pytest.raises(RuntimeError, match="no complete joint"):
        joint.recover_checkpoint(args)
    assert Path(args.save, "iter_0000004").exists()


def test_storage_failure_preserves_previous_recovery_point(tmp_path, monkeypatch):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _save(args, 4)
    joint.finalize_checkpoint(args, 4, 2)
    _save(args, 9)

    def fail(_):
        raise OSError("storage unavailable")

    monkeypatch.setattr(joint, "_commit_storage", fail)
    with pytest.raises(OSError):
        joint.finalize_checkpoint(args, 9, 1)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "4"
    assert Path(args.save, "iter_0000004").exists()


def test_recovery_before_derived_ppo_arguments_exist(tmp_path):
    args = _args(tmp_path)
    del args.use_critic
    args.advantage_estimator = "ppo"
    joint.recover_checkpoint(args)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "-1"


def test_full_argument_validation_repairs_before_fresh_fallback(tmp_path):
    from tests.fast.utils.test_arguments import TestMilesValidateArgsCheckpointResolution
    from miles.utils.arguments import miles_validate_args

    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _model(args.save, 4)  # first save interrupted after the actor
    parsed = TestMilesValidateArgsCheckpointResolution._parse(
        [
            "--advantage-estimator",
            "ppo",
            "--load",
            args.save,
            "--save",
            args.save,
            "--critic-load",
            args.critic_save,
            "--critic-save",
            args.critic_save,
        ],
        tmp_path,
    )
    parsed.max_keep = 2
    miles_validate_args(parsed)
    assert parsed.finetune and parsed.start_rollout_id == 0
    assert parsed.load == str(tmp_path)
    assert parsed.requested_load == args.save
