import asyncio
import json
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path

from miles.utils.file_utils import atomic_write_text
from miles.utils.function_registry import load_function
from miles.utils.volume_io import volume_access

logger = logging.getLogger(__name__)


class EvalDispatcher:
    """Dispatches eval points without holding up the training loop.

    Exports a snapshot per point, fires the eval, and reclaims the snapshot once the
    point settles. Failures degrade to a skipped point outside CI; under ``--ci-test``,
    every skipped point fails the run. Shared-engine eval takes the plain blocking call
    instead.
    """

    def __init__(self, args, actor_model, rollout_manager):
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, asyncio.Task, str | None]] = deque()
        self._exported: list[str] = []
        self._progress_path: Path | None = None
        self._groups: dict[str, dict] = {}

    async def resume_groups(self, start_rollout_id: int, *, previous_eval_due: bool) -> None:
        """Recover whole eval groups for a run with joint training checkpoints and dedicated HF exports."""
        self._progress_path = Path(self.args.eval_hf_dir) / "eval-progress.json"
        with volume_access(self._progress_path):
            try:
                self._groups = json.loads(self._progress_path.read_text())
            except FileNotFoundError:
                self._groups = {}

        # The trainer may have rolled back behind an eval. Its next pass through
        # that rollout owns a new snapshot, even if the numerical step is the same.
        stale = [
            key for key, group in self._groups.items() if key != "baseline" and group["rollout_id"] >= start_rollout_id
        ]
        for key in stale:
            del self._groups[key]
        if stale:
            self._save_progress()

        periodic = sorted((g for k, g in self._groups.items() if k != "baseline"), key=lambda g: g["rollout_id"])
        with volume_access(self._progress_path):
            self._exported = [g["hf_dir"] for g in periodic if g["done"] and Path(g["hf_dir"]).is_dir()]
        if not self.args.skip_eval_before_train:
            await self.dispatch(0, hf_dir=self.args.hf_checkpoint)
        for group in periodic:
            if not group["done"]:
                await self.dispatch(group["rollout_id"], force=True)
        # A crash can land between committing the training checkpoint and
        # exporting/registering its eval. The restored actor still has those weights.
        previous = start_rollout_id - 1
        if previous_eval_due and f"step_{previous}" not in self._groups:
            await self.dispatch(previous, force=True)

    def _save_progress(self) -> None:
        with volume_access(self._progress_path):
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._progress_path, json.dumps(self._groups))
        # commit/reload takes the exclusive lease; release shared access first.
        if hook := getattr(self.args, "checkpoint_commit_hook", None):
            load_function(hook)()

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        """A caller-supplied ``hf_dir`` is an existing checkpoint, not one of our exports."""
        if not self.args.eval_uses_snapshots:
            await self.rollout_manager.eval(rollout_id)
            return

        key = "baseline" if hf_dir is not None else f"step_{rollout_id}"
        group = self._groups.get(key)
        if group and group["done"]:
            return
        await self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip(rollout_id, "busy")
                return
            await self._settle(*self.pending.popleft())

        export_time = None
        exported_dir = None
        require_marker = hf_dir is None
        if group:
            hf_dir = group["hf_dir"]
            exported_dir = hf_dir if require_marker else None
        elif hf_dir is None:
            if self.args.eval_hf_dir is None:
                hf_dir = self.args.save_hf.format(rollout_id=rollout_id)
            else:
                hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
                try:
                    export_time = await self._export(rollout_id, hf_dir)
                except Exception as e:
                    logger.error(f"HF snapshot export for eval {rollout_id} failed: {e}")
                    if self._progress_path is not None:
                        raise
                    await self.rollout_manager.report_eval_skip(rollout_id, "export_failed")
                    return
                exported_dir = hf_dir

        if self._progress_path is not None and group is None:
            self._groups[key] = {"rollout_id": rollout_id, "hf_dir": hf_dir, "done": False}
            self._save_progress()
        eval_kwargs = dict(hf_dir=hf_dir, export_time_seconds=export_time, require_marker=require_marker)
        if self._progress_path is None:
            evaluation = self.rollout_manager.eval(rollout_id, **eval_kwargs)
        else:
            evaluation = self._record_result(key, rollout_id, eval_kwargs)
        task = asyncio.ensure_future(evaluation)
        self.pending.append((rollout_id, task, exported_dir))

    async def _record_result(self, key: str, rollout_id: int, eval_kwargs: dict) -> bool:
        metrics = await self.rollout_manager.eval(rollout_id, **eval_kwargs)
        if metrics is False:
            return False
        pending = self._groups[key]
        self._groups[key] = dict(pending, done=True, metrics=metrics)
        try:
            self._save_progress()
        except Exception:
            self._groups[key] = pending
            raise
        return True

    async def drain(self) -> None:
        while self.pending:
            await self._settle(*self.pending.popleft())

    async def close(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            await self.drain()
            return
        # A fatal training error must not keep the allocation alive for a full
        # baseline evaluation. Worker disposal will stop the remote trial owners.
        pending, self.pending = self.pending, deque()
        for _, task, _ in pending:
            task.cancel()
        await asyncio.gather(*(task for _, task, _ in pending), return_exceptions=True)
        # Keep snapshots on failure: a cancelled RPC may still be reading them
        # until the remote workers are shut down.

    async def _export(self, rollout_id: int, hf_dir: str) -> float:
        start = time.time()
        if self.args.offload_train:
            await self.actor_model.onload()
        try:
            await self.actor_model.export_hf(rollout_id, hf_dir)
            # Snapshot readers on other hosts need the same storage handoff as
            # checkpoint readers before the evaluation is dispatched.
            if hook := getattr(self.args, "checkpoint_commit_hook", None):
                load_function(hook)()
        finally:
            if self.args.offload_train:
                await self.actor_model.offload()
        return time.time() - start

    async def _reap_finished(self) -> None:
        while self.pending:
            if not self.pending[0][1].done():
                break
            await self._settle(*self.pending.popleft())

    async def _settle(self, rollout_id: int, task: asyncio.Task, exported_dir: str | None) -> None:
        recoverable = self._progress_path is not None
        final = recoverable and exported_dir is not None and rollout_id == self.args.num_rollout - 1
        completed = False
        try:
            completed = await task is not False
        except Exception:
            if final:
                raise
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip(rollout_id, "crashed")
        finally:
            if completed or not recoverable:
                self._retire(exported_dir)
        if final and not completed:
            raise RuntimeError(f"Final eval {rollout_id} did not complete")

    def _retire(self, exported_dir: str | None) -> None:
        """Every settled point retires exactly one snapshot, whatever its outcome."""
        if exported_dir is None:
            return
        self._exported.append(exported_dir)
        while len(self._exported) > self.args.eval_keep_snapshots:
            victim = self._exported.pop(0)
            with volume_access(victim):
                shutil.rmtree(victim, ignore_errors=True)
            logger.info(f"GC'd consumed eval snapshot {victim}")
