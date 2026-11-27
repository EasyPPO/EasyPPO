import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from miles.utils.volume_io import async_volume_access, volume_access


@pytest.fixture
def mount(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    root.mkdir()
    monkeypatch.setenv("MILES_VOLUME_IO_ROOT", str(root))
    monkeypatch.setenv("MILES_VOLUME_IO_LOCK", str(tmp_path / "access.lock"))
    return root


def test_append_waits_while_reload_removes_directory(mount):
    events = mount / "events"
    events.mkdir()
    path = events / "worker.jsonl"
    started, completed = Event(), Event()

    def append():
        started.set()
        with volume_access(path):
            with path.open("a") as file:
                file.write('{"step":1}\n')
        completed.set()

    with ThreadPoolExecutor() as pool:
        with volume_access(exclusive=True):
            # Reproduce the documented empty view during a real reload.
            hidden = mount.parent / "hidden"
            events.rename(hidden)
            future = pool.submit(append)
            assert started.wait(2)
            assert not completed.wait(0.05)
            hidden.rename(events)
        future.result(timeout=2)
    assert path.read_text() == '{"step":1}\n'


async def test_cancelling_a_waiter_never_leaves_a_lock_behind(mount):
    async def read():
        async with async_volume_access(mount / "weights"):
            pytest.fail("the exclusive owner still holds the lock")

    with volume_access(exclusive=True):
        waiter = asyncio.create_task(read())
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    with volume_access(exclusive=True):
        pass


def test_preempted_process_releases_its_lease(mount):
    code = (
        "import time\nfrom miles.utils.volume_io import volume_access\n"
        "with volume_access(exclusive=True):\n print('locked', flush=True)\n time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
    finally:
        child.terminate()
        child.wait(timeout=5)
        child.stdout.close()
    with volume_access(exclusive=True):
        pass


def test_other_volumes_remain_accessible_during_reload(mount):
    with ThreadPoolExecutor() as pool, volume_access(exclusive=True):

        def read_assets():
            with volume_access(mount.parent / "assets"):
                return True

        assert pool.submit(read_assets).result(timeout=2)


def test_lock_file_cannot_live_on_the_reloaded_volume(mount, monkeypatch):
    monkeypatch.setenv("MILES_VOLUME_IO_LOCK", str(mount / "bad.lock"))
    with pytest.raises(ValueError, match="outside"):
        with volume_access():
            pass


def test_missing_half_of_configuration_fails_closed(monkeypatch):
    monkeypatch.setenv("MILES_VOLUME_IO_ROOT", "/volume")
    monkeypatch.delenv("MILES_VOLUME_IO_LOCK", raising=False)
    with pytest.raises(ValueError, match="set together"):
        with volume_access():
            pass


def test_head_access_waits_for_the_slowest_node_reload(mount, monkeypatch):
    from threading import Lock, local
    from types import SimpleNamespace
    from miles.utils import volume_io
    from modal_launchers import multinode

    node = local()
    original_lock_path = volume_io._lock_path
    monkeypatch.setattr(volume_io, "_lock_path", lambda path: original_lock_path(path) + f".{node.rank}")
    monkeypatch.setattr(multinode, "STARTUP_TIMEOUT", 5)
    slow_reload, release_reload, wrote = Event(), Event(), Event()
    values, lock = {}, Lock()

    class Store:
        def set(self, key, value):
            with lock:
                values[key] = str(value).encode()

        def get(self, key):
            with lock:
                return values[key]

        def check(self, keys):
            with lock:
                return all(key in values for key in keys)

    store = Store()

    def worker():
        node.rank = 1

        def reload():
            slow_reload.set()
            assert release_reload.wait(3)

        multinode._worker_loop(store, None, SimpleNamespace(commit=lambda: None, reload=reload), rank=1, nodes=2)

    def head():
        node.rank = 0
        multinode._commit_all(store, SimpleNamespace(commit=lambda: None, reload=lambda: None), 2, "snapshot")

    def append():
        node.rank = 0
        with volume_access(mount / "events"):
            wrote.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        worker_task = pool.submit(worker)
        head_task = pool.submit(head)
        try:
            assert slow_reload.wait(2)
            append_task = pool.submit(append)
            assert not wrote.wait(0.05)
            release_reload.set()
            head_task.result(timeout=3)
            append_task.result(timeout=3)
            assert wrote.is_set()
        finally:
            release_reload.set()
            store.set("finished", "1")
        worker_task.result(timeout=3)


async def test_audit_snapshot_does_not_deadlock_an_async_weight_read(mount):
    from miles.utils.volume_io import event_snapshot_access

    async with async_volume_access(mount / "weights"):
        # Both operations run on the same event loop. Snapshotting must not
        # request an exclusive volume lease held by its sibling coroutine.
        with volume_access(mount / "events"), event_snapshot_access(mount / "events", exclusive=True):
            (mount / "snapshot").write_text("complete events")
