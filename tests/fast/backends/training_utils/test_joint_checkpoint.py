"""Crash recovery and publication ordering for synchronous PPO checkpoint sets."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.checkpoint import compute_rollout_checkpoint_dir
from miles.backends.training_utils.checkpoint import joint_checkpoint as joint
from miles.backends.training_utils.checkpoint.tracker import CHECKPOINT_TRACKER_FILENAME as TRACKER
from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.data_source import RolloutDataSource
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint
from miles.utils.types import Sample


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
    path = Path(compute_rollout_checkpoint_dir(args.save, rollout_id=step))
    path.mkdir(parents=True)
    (path / "state.pt").write_bytes(b"closed-rollout-state")


@pytest.mark.parametrize("partial", ["rollout", "rollout_actor", "complete", "actor", "both"])
def test_interruption_during_first_save_restarts_base(tmp_path, partial):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    if partial in ("rollout", "rollout_actor", "complete"):
        compute_rollout_checkpoint_dir(args.save, rollout_id=4).mkdir(parents=True)
    if partial != "rollout":
        _model(args.save, 4)
    if partial in ("both", "complete"):
        _model(args.critic_save, 4)
    joint.recover_checkpoint(args)
    assert not (Path(args.save) / TRACKER).exists()
    assert not (Path(args.critic_save) / TRACKER).exists()
    assert not list(Path(args.save).glob("iter_*"))
    assert not list(Path(args.critic_save).glob("iter_*"))
    assert not list(Path(args.save).glob("rollout/[0-9]*"))


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
    assert sorted(p.name for p in Path(args.save, "rollout").iterdir()) == ["14", "9"]
    _model(args.save, 19)
    joint.recover_checkpoint(args)
    assert Path(args.save, TRACKER).read_text().strip() == "14"
    assert not Path(args.save, "iter_0000019").exists()
    assert Path(compute_rollout_checkpoint_dir(args.save, rollout_id=14)).exists()


@pytest.mark.parametrize("missing", ["actor", "critic", "rollout"])
def test_incomplete_set_never_published_or_pruned(tmp_path, missing):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _save(args, 4)
    joint.finalize_checkpoint(args, 4, 2)
    _save(args, 9)
    if missing == "rollout":
        shutil.rmtree(compute_rollout_checkpoint_dir(args.save, rollout_id=9))
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
    from miles.utils.arguments import miles_validate_args
    from tests.fast.utils.test_arguments import TestMilesValidateArgsCheckpointResolution

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


def test_storage_hook_makes_remote_checkpoint_visible_before_validation(tmp_path, monkeypatch):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    commits = []

    def synchronize(_):
        commits.append(1)
        if len(commits) == 1:
            # Other nodes already saved these files, but the driver has not
            # reloaded their committed filesystem view yet.
            _save(args, 4)
        else:
            assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "4"

    monkeypatch.setattr(joint, "_commit_storage", synchronize)
    joint.finalize_checkpoint(args, 4, 2)
    assert len(commits) == 2


def test_incomplete_older_pair_does_not_count_towards_retained_sets(tmp_path):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    _save(args, 4)
    joint.finalize_checkpoint(args, 4, 2)
    # A pair with no published rollout directory is not a recovery point.
    _model(args.save, 9)
    _model(args.critic_save, 9)
    _save(args, 14)
    joint.finalize_checkpoint(args, 14, 2)
    assert joint._checkpoint_exists(args, 4)
    assert joint._checkpoint_exists(args, 14)


@pytest.mark.parametrize("global_dataset", [False, True])
def test_joint_save_requires_rollout_state_even_without_global_dataset(tmp_path, global_dataset):
    args = _args(tmp_path)
    args.rollout_global_dataset = global_dataset
    joint.recover_checkpoint(args)
    _model(args.save, 4)
    _model(args.critic_save, 4)
    with pytest.raises(RuntimeError, match="missing model or rollout"):
        joint.finalize_checkpoint(args, 4, 2)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "-1"


class _RolloutState:
    def __init__(self, cursor=0):
        self.cursor = cursor

    def save(self, directory):
        save_simple_checkpoint(directory, {"cursor": self.cursor})

    def load(self, directory):
        self.cursor = load_simple_checkpoint(directory)["cursor"]


def _executor(args):
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = SimpleNamespace(
        save=args.save,
        load=args.load,
        save_debug_event_data=None,
        ci_inject_missing_prefetched_batch_bug=False,
    )
    executor.use_legacy_rollout_v1 = False
    executor.generate_rollout = _RolloutState()
    executor.eval_generate_rollout = _RolloutState()
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = SimpleNamespace(rollout_global_dataset=args.rollout_global_dataset, rollout_shuffle=False)
    source.sample_offset = source.epoch_id = source.sample_group_index = source.sample_index = 0
    source.metadata = {}
    executor.data_source = source
    executor._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=executor.args)
    return executor


@pytest.mark.parametrize("global_dataset", [False, True])
async def test_retention_and_recovery_round_trip_real_rollout_checkpoints(tmp_path, global_dataset):
    args = _args(tmp_path)
    args.rollout_global_dataset = global_dataset
    joint.recover_checkpoint(args)
    executor = _executor(args)
    for step in (4, 9, 14):
        executor.data_source.sample_offset = step * 10
        executor.generate_rollout.cursor = step * 100
        executor.eval_generate_rollout.cursor = step * 1000
        executor._output_snapshotter.capture(
            trainer_model_id=None,
            rollout_id=step + 1,
            data=[Sample(index=step + 100)],
            metadata={"step": step},
        )
        await executor.save(step)
        _model(args.save, step)
        _model(args.critic_save, step)
        joint.finalize_checkpoint(args, step, 2)

    for root in (args.save, args.critic_save):
        assert sorted(path.name for path in Path(root).glob("iter_*")) == ["iter_0000009", "iter_0000014"]
    assert sorted(path.name for path in Path(args.save, "rollout").iterdir()) == ["14", "9"]

    # The next save publishes rollout and actor, then dies before the critic.
    executor.data_source.sample_offset = 190
    await executor.save(19)
    _model(args.save, 19)
    joint.recover_checkpoint(args)
    assert not compute_rollout_checkpoint_dir(args.save, rollout_id=19).exists()
    assert not Path(args.save, "iter_0000019").exists()
    assert Path(args.save, TRACKER).read_text().strip() == "14"
    assert Path(args.critic_save, TRACKER).read_text().strip() == "14"

    restored = _executor(args)
    await restored.load(14)
    assert restored.generate_rollout.cursor == 1400
    assert restored.eval_generate_rollout.cursor == 14000
    assert restored.data_source.sample_offset == (140 if global_dataset else 0)
    pending = restored._output_snapshotter.get(trainer_model_id=None, rollout_id=15)
    assert pending.data[0].index == 114 and pending.metadata == {"step": 14}


async def test_interrupted_atomic_rollout_save_preserves_last_joint_checkpoint(tmp_path, monkeypatch):
    args = _args(tmp_path)
    joint.recover_checkpoint(args)
    executor = _executor(args)
    await executor.save(4)
    _model(args.save, 4)
    _model(args.critic_save, 4)
    joint.finalize_checkpoint(args, 4, 2)

    def fail(directory):
        save_simple_checkpoint(directory, {"partial": True})
        raise OSError("interrupted rollout write")

    monkeypatch.setattr(executor.generate_rollout, "save", fail)
    with pytest.raises(OSError, match="interrupted rollout write"):
        await executor.save(9)
    assert not compute_rollout_checkpoint_dir(args.save, rollout_id=9).exists()
    assert not list(Path(args.save, "rollout").glob(".tmp-*"))
    joint.recover_checkpoint(args)
    await _executor(args).load(4)
    assert Path(args.save, joint.PPO_CHECKPOINT_TRACKER).read_text().strip() == "4"
