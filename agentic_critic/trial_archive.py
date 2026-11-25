"""Bounded node-local staging and immutable diagnostic archives (not checkpoints)."""

import asyncio
import logging
import os
import shutil
import tarfile
from collections import deque
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def _write_shard(target: Path, roots: list[Path]) -> None:
    import zstandard as zstd

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with tmp.open("xb") as raw:
            with zstd.ZstdCompressor(level=1).stream_writer(raw, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", bufsize=1024 * 1024) as tar:
                    for root in roots:
                        tar.add(root, arcname=root.name, recursive=True)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # Never remove staging before the finished shard is published.
    for root in roots:
        try:
            shutil.rmtree(root)
        except OSError:
            logger.exception("Could not remove archived local trial %s", root)


class TrialArchiveWriter:
    """One async writer per Ray node/role; one compression thread per writer."""

    def __init__(self, directory, *, shard_trials=128, max_pending=256, max_delay=30.0):
        if not 0 < shard_trials <= max_pending or max_delay <= 0:
            raise ValueError("Invalid trial archive batching limits")
        self.directory = Path(directory)
        self.shard_trials = shard_trials
        self.max_pending = max_pending
        self.max_delay = max_delay
        self._condition = asyncio.Condition()
        self._open = []
        self._ready = deque()
        self._target = None
        self._deadline = None
        self._pending = 0
        self._worker = None
        self._error = None

    def _raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Trial archive writer failed; local staging retained") from self._error

    def _seal(self):
        if self._open:
            self._ready.append((self._target, self._open))
            self._open = []
            self._target = None
            self._deadline = None

    async def enqueue(self, root):
        """Return the final archive path after enqueue, not after compression."""
        async with self._condition:
            await self._condition.wait_for(lambda: self._error is not None or self._pending < self.max_pending)
            self._raise_if_failed()
            if not self._open:
                self._target = self.directory / f"shard_{uuid4().hex}.tar.zst"
                self._deadline = asyncio.get_running_loop().time() + self.max_delay
            target = self._target
            self._open.append(Path(root))
            self._pending += 1
            if len(self._open) >= self.shard_trials:
                self._seal()
            if self._worker is None:
                self._worker = asyncio.create_task(self._run())
            self._condition.notify_all()
            return str(target)

    async def _run(self):
        while True:
            async with self._condition:
                while not self._ready:
                    if not self._open:
                        await self._condition.wait()
                        continue
                    remaining = self._deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        self._seal()
                        break
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except TimeoutError:
                        self._seal()
                target, roots = self._ready.popleft()
            try:
                await asyncio.to_thread(_write_shard, target, roots)
            except Exception as exc:
                logger.exception("Failed to publish trial archive %s", target)
                async with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                return
            async with self._condition:
                self._pending -= len(roots)
                self._condition.notify_all()

    async def flush(self):
        """Call after the producer stops (or an evaluation point finishes)."""
        async with self._condition:
            self._seal()
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._error is not None or self._pending == 0)
            self._raise_if_failed()
