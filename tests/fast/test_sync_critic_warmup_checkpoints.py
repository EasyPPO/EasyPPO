"""Synchronous critic warmup checkpoints keep the frozen actor resumable."""

import pytest
import train as train_driver
from tests.fast.fixtures.driver_fakes import FakeObjectStore
from tests.fast.test_train import _install_driver_fakes, _make_args

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store
from miles.utils.async_utils import with_disposer


@pytest.mark.parametrize("offload", [False, True])
async def test_warmup_saves_actor_and_critic_at_the_same_rollout_without_actor_updates(monkeypatch, offload):
    events = []
    args = _make_args(
        num_rollout=30,
        num_critic_only_steps=30,
        use_critic=True,
        offload_train=offload,
        save_interval=10,
        save_on_epoch=False,
    )
    components = _install_driver_fakes(monkeypatch, args, events)
    monkeypatch.setattr(object_store, "_INSTANCE", FakeObjectStore())
    for rollout_id in range(30):
        components.critic_model.train_outputs[rollout_id] = [TrainStepOutput(outcome=TrainStepOutcome.NORMAL)]

    await with_disposer(train_driver.train, args)

    assert components.actor_model.trained == []
    assert components.critic_model.trained == list(range(30))
    assert components.actor_model.saved == components.critic_model.saved == [9, 19, 29]
    assert [event for event in events if event.startswith("executor_save:")] == [
        "executor_save:9",
        "executor_save:19",
        "executor_save:29",
    ]
    for rollout_id in (9, 19, 29):
        assert events.index(f"critic_train:{rollout_id}") < events.index(f"actor_save:{rollout_id}")
        assert events.index(f"actor_save:{rollout_id}") < events.index(f"critic_save:{rollout_id}")
    if offload:
        assert events.count("actor_onload") == 3
        assert events.count("actor_offload") == 3


async def test_synchronous_run_without_critic_keeps_actor_checkpoint_cadence(monkeypatch):
    events = []
    args = _make_args(num_rollout=30, save_interval=10, save_on_epoch=False)
    components = _install_driver_fakes(monkeypatch, args, events)

    await with_disposer(train_driver.train, args)

    assert components.critic_model is None
    assert components.actor_model.trained == list(range(30))
    assert components.actor_model.saved == [9, 19, 29]
