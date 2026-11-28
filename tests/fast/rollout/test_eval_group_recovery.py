import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.rollout.eval_dispatch import EvalDispatcher


def dispatcher(root, *, baseline=False, final=3):
    args = SimpleNamespace(
        eval_hf_dir=str(root),
        eval_uses_snapshots=True,
        skip_eval_before_train=not baseline,
        hf_checkpoint="/base",
        eval_keep_snapshots=1,
        eval_max_in_flight=2,
        eval_overflow_policy="backpressure",
        offload_train=False,
        num_rollout=final + 1,
    )
    manager = SimpleNamespace(eval=AsyncMock(return_value={"eval/tb21": 0.5}), report_eval_skip=AsyncMock())
    return EvalDispatcher(args, SimpleNamespace(export_hf=AsyncMock()), manager)


@pytest.mark.asyncio
async def test_resume_baseline_and_step_zero_as_distinct_groups(tmp_path):
    first = dispatcher(tmp_path, baseline=True, final=1)
    started = asyncio.Event()

    async def interrupted(*args, **kwargs):
        started.set()
        await asyncio.Future()

    first.rollout_manager.eval.side_effect = interrupted
    await first.resume_groups(0, previous_eval_due=False)
    await first.dispatch(0)
    await started.wait()
    await first.close(RuntimeError, RuntimeError("interrupt"), None)

    restored = dispatcher(tmp_path, baseline=True, final=1)
    await restored.resume_groups(1, previous_eval_due=True)
    await restored.drain()
    assert [call.kwargs["hf_dir"] for call in restored.rollout_manager.eval.await_args_list] == [
        "/base",
        str(tmp_path / "step_0"),
    ]
    restored.actor_model.export_hf.assert_not_awaited()
    groups = json.loads((tmp_path / "eval-progress.json").read_text())
    assert set(groups) == {"baseline", "step_0"}
    assert all(group["done"] and group["metrics"] == {"eval/tb21": 0.5} for group in groups.values())

    again = dispatcher(tmp_path, baseline=True, final=1)
    await again.resume_groups(1, previous_eval_due=True)
    again.rollout_manager.eval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, RuntimeError("worker died")])
async def test_final_failure_keeps_snapshot_until_successful_resume(tmp_path, failure):
    first = dispatcher(tmp_path)
    await first.resume_groups(3, previous_eval_due=False)
    (tmp_path / "step_3").mkdir()
    if isinstance(failure, Exception):
        first.rollout_manager.eval.side_effect = failure
    else:
        first.rollout_manager.eval.return_value = failure
    await first.dispatch(3, force=True)
    with pytest.raises(RuntimeError):
        await first.close(None, None, None)
    assert (tmp_path / "step_3").exists()
    assert not json.loads((tmp_path / "eval-progress.json").read_text())["step_3"]["done"]

    restored = dispatcher(tmp_path)
    await restored.resume_groups(4, previous_eval_due=True)
    await restored.drain()
    restored.actor_model.export_hf.assert_not_awaited()
    restored.rollout_manager.eval.assert_awaited_once()
    again = dispatcher(tmp_path)
    await again.resume_groups(4, previous_eval_due=True)
    again.rollout_manager.eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_between_training_commit_and_eval_dispatch(tmp_path):
    restored = dispatcher(tmp_path)
    await restored.resume_groups(4, previous_eval_due=True)
    await restored.drain()
    restored.actor_model.export_hf.assert_awaited_once_with(3, str(tmp_path / "step_3"))

    rolled_back = dispatcher(tmp_path)
    await rolled_back.resume_groups(3, previous_eval_due=False)
    await rolled_back.dispatch(3, force=True)
    await rolled_back.drain()
    rolled_back.actor_model.export_hf.assert_awaited_once_with(3, str(tmp_path / "step_3"))


@pytest.mark.asyncio
async def test_retention_prunes_only_completed_snapshots(tmp_path):
    run = dispatcher(tmp_path, final=8)
    await run.resume_groups(0, previous_eval_due=False)
    for step in (1, 2, 3):
        (tmp_path / f"step_{step}").mkdir()
        run.rollout_manager.eval.return_value = False if step == 1 else {"eval/reward": 1.0}
        await run.dispatch(step)
        await run.drain()

    assert (tmp_path / "step_1").is_dir()
    assert not (tmp_path / "step_2").exists()
    assert (tmp_path / "step_3").is_dir()
    retried = dispatcher(tmp_path, final=8)
    await retried.resume_groups(4, previous_eval_due=False)
    await retried.drain()
    assert [c.args[0] for c in retried.rollout_manager.eval.await_args_list] == [1]
    retried.actor_model.export_hf.assert_not_awaited()
