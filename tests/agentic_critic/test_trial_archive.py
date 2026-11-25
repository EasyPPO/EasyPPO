import asyncio
import tarfile
from pathlib import Path

import pytest

from agentic_critic.trial_archive import TrialArchiveWriter


def _trial(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    (root / "result.json").write_text(name)
    return root


async def _stop(writer):
    if writer._worker is not None:
        writer._worker.cancel()
        await asyncio.gather(writer._worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_archive_batches_and_flushes_tail(tmp_path):
    zstd = pytest.importorskip("zstandard")
    writer = TrialArchiveWriter(tmp_path / "archives", shard_trials=2, max_pending=4, max_delay=60)
    roots = [_trial(tmp_path, f"trial-{i}") for i in range(3)]
    active = _trial(tmp_path, "still-running")
    try:
        paths = [await writer.enqueue(root) for root in roots]
        assert paths[0] == paths[1] and paths[1] != paths[2]
        await writer.flush()
        assert not any(root.exists() for root in roots)
        assert active.exists()
        for path, expected in [(paths[0], roots[:2]), (paths[2], roots[2:])]:
            with open(path, "rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    data = {item.name: tar.extractfile(item).read() for item in tar if item.isfile()}
            assert data == {f"{root.name}/result.json": root.name.encode() for root in expected}
    finally:
        await _stop(writer)


@pytest.mark.asyncio
async def test_archive_timer(tmp_path):
    pytest.importorskip("zstandard")
    writer = TrialArchiveWriter(tmp_path / "archives", shard_trials=128, max_delay=0.02)
    root = _trial(tmp_path, "finished")
    try:
        path = Path(await writer.enqueue(root))
        async with asyncio.timeout(5):
            while root.exists():
                await asyncio.sleep(0.01)
        assert path.is_file()
        await writer.flush()
    finally:
        await _stop(writer)


@pytest.mark.asyncio
async def test_archive_failure_retains_staging(tmp_path, monkeypatch):
    from agentic_critic import trial_archive

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(trial_archive, "_write_shard", fail)
    writer = TrialArchiveWriter(tmp_path / "archives", shard_trials=1)
    root = _trial(tmp_path, "finished")
    try:
        await writer.enqueue(root)
        with pytest.raises(RuntimeError, match="local staging retained"):
            await asyncio.wait_for(writer.flush(), timeout=5)
        assert root.exists()
    finally:
        await _stop(writer)
