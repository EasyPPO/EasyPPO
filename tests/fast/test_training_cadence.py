"""Explicit cadence keeps costly terminal evaluation independent of dataset epochs."""

import pytest

from miles.utils.async_utils import with_disposer


@pytest.mark.parametrize("driver_name", ["test_train", "test_train_async"])
@pytest.mark.parametrize("on_epoch", [False, True])
async def test_eval_and_checkpoint_cadence(monkeypatch, driver_name, on_epoch):
    from importlib import import_module

    harness = import_module(f"tests.fast.{driver_name}")
    driver = harness.train_driver if driver_name == "test_train" else harness.train_async_driver
    events = []
    args = harness._make_args(
        num_rollout=12,
        eval_interval=10,
        save_interval=10,
        eval_on_epoch=on_epoch,
        save_on_epoch=on_epoch,
        eval_uses_snapshots=False,
    )
    components = harness._install_driver_fakes(monkeypatch, args, events)

    await with_disposer(driver.train, args)

    expected = [3, 7, 9, 11] if on_epoch else [9, 11]
    assert components.actor_model.saved == expected
    assert [e for e in events if e.startswith("eval:")] == ["eval:0", *[f"eval:{i}" for i in expected]]
