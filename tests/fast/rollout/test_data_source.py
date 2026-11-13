import logging
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.data_source import RolloutDataSource


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _bare_source(**overrides) -> RolloutDataSource:
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = _make_args(**overrides)
    source.metadata = {}
    return source


def test_load_says_so_when_it_finds_no_state(tmp_path: Path, caplog) -> None:
    """A dataset silently starting over is a run replaying samples its trainers already trained on."""
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no dataset state under" in caplog.text


def test_load_says_so_when_the_run_names_no_load_directory(tmp_path: Path, caplog) -> None:
    """A run told to write but not to read still has to say that its dataset starts from the beginning."""
    source = _bare_source(rollout_global_dataset=True, load=None)

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "no --load" in caplog.text


def test_load_says_so_when_the_run_keeps_no_global_dataset(tmp_path: Path, caplog) -> None:
    """A custom rollout function keeps its own state, and the operator has to know this one restored none."""
    source = _bare_source(rollout_global_dataset=False, load=str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="miles.rollout.data_source"):
        source.load(rollout_id=3)

    assert "rollout-global-dataset" in caplog.text


def test_load_restores_the_state_it_finds(tmp_path: Path) -> None:
    """This is the ordinary resume, and the position it restores is what keeps a run off samples it has seen."""
    import torch

    from miles.rollout.data_source import compute_global_dataset_state_path

    path = Path(compute_global_dataset_state_path(str(tmp_path), rollout_id=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"sample_offset": 7, "epoch_id": 1}, path)
    source = _bare_source(rollout_global_dataset=True, load=str(tmp_path))

    source.load(rollout_id=3)

    assert (source.sample_offset, source.epoch_id) == (7, 1)


def test_save_and_resume_preserve_dataset_position(tmp_path):
    source = _bare_source(rollout_global_dataset=True, save=str(tmp_path), load=str(tmp_path))
    source.sample_offset, source.epoch_id = 123, 2
    source.sample_group_index, source.sample_index = 45, 67
    source.metadata = {"seed": 42}
    source.save(3)
    restored = _bare_source(rollout_global_dataset=True, load=str(tmp_path))
    restored.load(3)
    assert (restored.sample_offset, restored.epoch_id, restored.sample_group_index, restored.sample_index) == (
        123,
        2,
        45,
        67,
    )
    assert restored.metadata == source.metadata
    assert not list((tmp_path / "rollout").glob("*.tmp"))


def test_failed_save_does_not_publish_partial_rollout_state(tmp_path, monkeypatch):
    import pytest
    import torch

    source = _bare_source(rollout_global_dataset=True, save=str(tmp_path))
    source.sample_offset = source.epoch_id = source.sample_group_index = source.sample_index = 0
    source.save(3)
    path = tmp_path / "rollout/global_dataset_state_dict_3.pt"
    before = path.read_bytes()

    def fail_save(state, file):
        file.write(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted write"):
        source.save(3)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))
