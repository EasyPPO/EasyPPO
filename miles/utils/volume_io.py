"""Opt-in process-shared exclusion for files on a reloadable volume.

The platform sets MILES_VOLUME_IO_ROOT and MILES_VOLUME_IO_LOCK. The lock file
must live on node-local storage, outside the mount that disappears on reload.
All filesystem users take shared access; the platform takes exclusive access
on every node before reloading any mount, and releases only after all reloads.
Cross-node operations must acquire shared access on the coordinator node before
issuing remote I/O. The platform acquires that node's exclusive lock first.
"""

import asyncio
import fcntl
import os
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

LOCK_TIMEOUT = 900.0


def _lock_path(path: str | Path | None) -> str | None:
    root = os.environ.get("MILES_VOLUME_IO_ROOT")
    lock = os.environ.get("MILES_VOLUME_IO_LOCK")
    if root is None and lock is None:
        return None
    if not root or not lock:
        raise ValueError("MILES_VOLUME_IO_ROOT and MILES_VOLUME_IO_LOCK must be set together")
    # Do not resolve symlinks/stat the mount before acquiring the lock.
    root, lock = os.path.abspath(root), os.path.abspath(lock)
    if Path(lock).is_relative_to(root):
        raise ValueError("The volume I/O lock must be outside the reloadable mount")
    if path is not None and not Path(os.path.abspath(path)).is_relative_to(root):
        return None
    return lock


def _acquired(file, *, exclusive: bool, deadline: float) -> bool:
    try:
        fcntl.flock(file, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for volume filesystem access") from None
        return False


@contextmanager
def volume_access(path: str | Path | None = None, *, exclusive: bool = False):
    with _file_access(_lock_path(path), exclusive=exclusive):
        yield


@contextmanager
def event_snapshot_access(path: str | Path, *, exclusive: bool = False):
    """Freeze only audit appenders; never wait for an async weight-load lease.

    Callers also hold volume_access. This separate short-lived lock cannot be
    held across an await, so snapshotting on an event loop cannot deadlock an
    evaluation coroutine on that same loop.
    """
    lock = _lock_path(path)
    with _file_access(None if lock is None else lock + ".events", exclusive=exclusive):
        yield


@contextmanager
def _file_access(lock: str | None, *, exclusive: bool):
    if lock is None:
        yield
        return
    with open(lock, "a") as file:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while not _acquired(file, exclusive=exclusive, deadline=deadline):
            time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


@asynccontextmanager
async def async_volume_access(path: str | Path | None = None):
    if (lock := _lock_path(path)) is None:
        yield
        return
    # Poll nonblocking flock on the event loop: cancellation must close the fd,
    # rather than leaving a background thread that later acquires a leaked lock.
    with open(lock, "a") as file:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while not _acquired(file, exclusive=False, deadline=deadline):
            await asyncio.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)
