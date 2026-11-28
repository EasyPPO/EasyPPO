import argparse
import asyncio
import runpy
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import train_async as train_async_driver
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeObjectStore,
    FakeRolloutExecutor,
    FakeTrainingModel,
    FakeWorkerManager,
)

from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray import placement_group as placement_group_mod
from miles.ray import wiring
from miles.utils import object_store
from miles.utils.async_utils import with_disposer
from miles.utils.data import RolloutDataPack


def test_async_entrypoint_accepts_joint_checkpoint_flags(monkeypatch):
    from miles.utils import arguments

    parsed = []

    def parse_cli(add_custom_arguments=None):
        parser = argparse.ArgumentParser()
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)
        args = parser.parse_args(["--max-keep", "2", "--checkpoint-commit-hook", "modal_launchers.runtime.commit_volume"])
        parsed.append(args)
        return args

    monkeypatch.setattr(arguments, "parse_args", parse_cli)
    monkeypatch.setattr(asyncio, "run", lambda coroutine: coroutine.close())
    runpy.run_path(train_async_driver.__file__, run_name="__main__")
    assert parsed[0].max_keep == 2
    assert parsed[0].checkpoint_commit_hook == "modal_launchers.runtime.commit_volume"


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        **parser_defaults(),
    )
    defaults: dict[str, Any] = dict(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate=False,
        debug_exit_after_rollout=None,
        eval_hf_dir=None,
        eval_interval=None,
        eval_max_in_flight=2,
        eval_overflow_policy="skip",
        eval_uses_snapshots=True,
        fully_async=False,
        ft_components=[],
        hf_checkpoint=None,
        keep_old_actor=False,
        num_critic_only_steps=0,
        num_rollout=0,
        offload_train=False,
        save_hf=None,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        update_weights_interval=1,
        use_critic=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )
    for key, value in {**defaults, **overrides}.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
        api_server_calls=[],
        cell_operations=object(),
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(
        _args: Any, _model: Any, _executor: Any, _inference_controller: Any, *, rollout_id: int | None = None
    ) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_async_driver, "init_orchestration_script", lambda _args, *, disposer: None)
    monkeypatch.setattr(train_async_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_async_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_async_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_async_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_async_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    # the driver reaches the server through maybe_start_api_server, whose gate the tests exercise
    monkeypatch.setattr(
        placement_group_mod, "start_api_server", lambda **kwargs: components.api_server_calls.append(kwargs)
    )
    monkeypatch.setattr(
        placement_group_mod,
        "get_backend_capability",
        lambda _args: SimpleNamespace(cell_operations=lambda: components.cell_operations),
    )
    return components


class TestApiServer:
    async def test_api_server_receives_the_refactored_driver_handles(self, monkeypatch: pytest.MonkeyPatch):
        """The API server acts on the live actor and inference controller, so it must be handed those objects."""
        events: list[str] = []
        args = _make_args(api_server_port=8123, ft_components=["rollout"])
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        (call,) = components.api_server_calls
        assert list(call["trainer_models"]) == ["actor"]
        assert call["trainer_models"]["actor"] is components.actor_model
        assert call["inference_controller"] is components.inference_controller
        assert call["port"] == 8123
        assert call["ft_components"] == ["rollout"]

    async def test_no_api_server_without_a_port(self, monkeypatch: pytest.MonkeyPatch):
        """An unrequested API server would expose a control plane the operator never asked for."""
        events: list[str] = []
        args = _make_args(api_server_port=None)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert components.api_server_calls == []


async def test_fatal_training_error_cancels_pending_eval(monkeypatch):
    events = []
    args = _make_args(num_rollout=1, eval_interval=1, hf_checkpoint="base", skip_eval_before_train=False)
    components = _install_driver_fakes(monkeypatch, args, events)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def evaluate(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fail(*args, **kwargs):
        await started.wait()
        raise RuntimeError("fatal optimizer failure")

    monkeypatch.setattr(components.rollout_executor, "eval", evaluate)
    monkeypatch.setattr(components.actor_model, "train", fail)
    with pytest.raises(RuntimeError, match="fatal optimizer failure"):
        await asyncio.wait_for(with_disposer(train_async_driver.train, args), timeout=2)
    assert cancelled.is_set()


@pytest.mark.parametrize("fail_at", [None, "actor", "critic"])
async def test_async_checkpoint_publishes_only_after_complete_synchronous_saves(monkeypatch, tmp_path, fail_at):
    events = []
    args = _make_args(num_rollout=2, debug_exit_after_rollout=1, use_critic=True, keep_old_actor=True,
                      num_critic_only_steps=2, save_interval=1, max_keep=2,
                      save=str(tmp_path / "actor"), critic_save=str(tmp_path / "critic"))
    components = _install_driver_fakes(monkeypatch, args, events)
    monkeypatch.setattr(train_async_driver, "remove_train_output_refs", lambda *_: None)
    monkeypatch.setattr(train_async_driver, "prepare_rollout_checkpoint", lambda *_: events.append("refresh_events"))

    def save(role):
        async def callback(rollout_id, force_sync):
            assert force_sync and rollout_id == 0
            events.append(f"{role}_save")
            if fail_at == role:
                raise RuntimeError("incomplete checkpoint")
        return callback

    monkeypatch.setattr(components.actor_model, "save_model", save("actor"))
    monkeypatch.setattr(components.critic_model, "save_model", save("critic"))
    monkeypatch.setattr(train_async_driver, "finalize_checkpoint", lambda *_: events.append("published"))
    if fail_at:
        with pytest.raises(RuntimeError, match="incomplete checkpoint"):
            await with_disposer(train_async_driver.train, args)
        assert "published" not in events
    else:
        await with_disposer(train_async_driver.train, args)
        assert events.index("refresh_events") < events.index("executor_save:0") < events.index("actor_save") < events.index("critic_save") < events.index("published")
        assert components.actor_model.trained == []


@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("writer_fails", [False, True])
async def test_joint_async_writer_overlaps_next_rollout(monkeypatch, tmp_path, cpu_offload, writer_fails):
    events = []
    args = _make_args(
        num_rollout=2, use_critic=True, fully_async=True, use_rollout_logprobs=True,
        async_save=True, optimizer_cpu_offload=cpu_offload, train_backend="megatron",
        max_keep=2, save_interval=1, save=str(tmp_path / "actor"), critic_save=str(tmp_path / "critic"),
    )
    components = _install_driver_fakes(monkeypatch, args, events)
    monkeypatch.setattr(train_async_driver, "remove_train_output_refs", lambda *_: None)
    monkeypatch.setattr(train_async_driver, "finalize_checkpoint", lambda _, step, keep: events.append(f"publish:{step}"))

    def install_writer(model, role):
        async def start(step):
            assert step == 0
            events.append(f"start:{role}")
            return role, "workers"

        async def finish(cell, workers):
            assert (cell, workers) == (role, "workers")
            events.append(f"finish:{role}")
            if role == "critic" and writer_fails:
                raise OSError("shard write failed")

        monkeypatch.setattr(model, "start_async_checkpoint", start, raising=False)
        monkeypatch.setattr(model, "finish_async_checkpoint", finish, raising=False)

    install_writer(components.actor_model, "actor")
    install_writer(components.critic_model, "critic")
    if writer_fails:
        with pytest.raises(OSError, match="shard write failed"):
            await with_disposer(train_async_driver.train, args)
        assert "publish:0" not in events
    else:
        await with_disposer(train_async_driver.train, args)
        assert events.index("finish:rollout") < events.index("finish:critic") < events.index("publish:0") < events.index("actor_save:1")
        assert events.index("critic_save:1") < events.index("publish:1")
    assert events.index("start:critic") < events.index("generate_done:1") < events.index("finish:actor")
    if cpu_offload:
        assert "critic_train:1" not in events or events.index("finish:critic") < events.index("critic_train:1")
    else:
        assert events.index("critic_train:1") < events.index("finish:actor")


class TestWeightEqualityCheck:
    async def test_weight_equality_check_is_routed_to_the_inference_controller(self, monkeypatch: pytest.MonkeyPatch):
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]


class TestPipelinedGeneration:
    async def test_inflight_next_rollout_finishes_before_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        """Generation for the next rollout starts while this one trains, but must settle before new weights ship."""
        events: list[str] = []
        args = _make_args(num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)
        held_generation = asyncio.Event()
        components.rollout_executor.generation_gates[1] = held_generation

        driver = asyncio.create_task(with_disposer(train_async_driver.train, args))
        await asyncio.wait_for(components.actor_model.train_started[0].wait(), timeout=10)

        assert "generate_start:1" in events
        assert "generate_done:1" not in events
        assert "update_weights:0" not in events

        held_generation.set()
        await asyncio.wait_for(driver, timeout=10)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("generate_done:1") < events.index("update_weights:0")
        assert components.actor_model.trained == [0, 1]

    async def test_fully_async_next_drain_starts_after_weight_publication(self, monkeypatch: pytest.MonkeyPatch):
        """The persistent producer needs no lookahead drain that captures the previous weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=2, update_weights_interval=1)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert events.index("actor_train:0") < events.index("update_weights:0")
        assert events.index("update_weights:0") < events.index("generate_start:1")
        assert "generate_start:2" not in events
        assert components.actor_model.trained == [0, 1]

    async def test_fully_async_keeps_lookahead_between_weight_updates(self, monkeypatch: pytest.MonkeyPatch):
        """A drain may still overlap training when that step cannot change the current weight version."""
        events: list[str] = []
        args = _make_args(fully_async=True, num_rollout=3, update_weights_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert events.index("generate_start:1") < events.index("actor_train:0")
        assert events.index("actor_train:1") < events.index("update_weights:1")
        assert events.index("update_weights:1") < events.index("generate_start:2")
        assert components.actor_model.trained == [0, 1, 2]


class TestCriticValuesHandoff:
    async def test_episode_critic_receives_its_own_shards(self, monkeypatch):
        events = []
        args = _make_args(num_rollout=1, use_critic=True, keep_old_actor=True)
        components = _install_driver_fakes(monkeypatch, args, events)
        store = FakeObjectStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        actor_ref, critic_ref = store.put("actor-shard"), store.put("critic-episode-shard")
        pack = RolloutDataPack(sample_indices=[0], data_ref=[actor_ref], critic_data_ref=[critic_ref])
        components.rollout_executor.generation_packs.append(pack)
        captured = {}

        async def critic_train(rollout_id, data):
            captured["critic"] = data
            return []

        async def actor_train(rollout_id, data, external_data):
            captured["actor"] = data

        monkeypatch.setattr(components.critic_model, "train", critic_train)
        monkeypatch.setattr(components.actor_model, "train", actor_train)
        await with_disposer(train_async_driver.train, args)
        assert captured["critic"].data_ref == [critic_ref]
        assert captured["actor"] is pack and captured["actor"].data_ref == [actor_ref]

    async def test_async_critic_outputs_reach_the_actor_and_are_released_after_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pipelined critic value references stay live through the actor step that consumes them."""
        events: list[str] = []
        args = _make_args(num_rollout=1, use_critic=True, keep_old_actor=True)
        components = _install_driver_fakes(monkeypatch, args, events)
        store = FakeObjectStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        ref = store.put("critic-values")
        values = [TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref)]
        components.critic_model.train_outputs[0] = values

        def consume_critic_values(external_data: list[TrainStepOutput]) -> None:
            assert external_data is values
            assert store.get(external_data[0].values).value == "critic-values"

        components.actor_model.consume_external_data = consume_critic_values

        await with_disposer(train_async_driver.train, args)

        assert store.consumed == [ref]
        assert not store.contains(ref)


class TestTerminalLifecycle:
    @pytest.mark.parametrize("audit_fails", [False, True])
    async def test_final_audit_runs_with_live_api_and_cells(
        self, monkeypatch: pytest.MonkeyPatch, audit_fails: bool
    ) -> None:
        """Final audits retain a working observation endpoint even when the audit raises."""
        events: list[str] = []
        args = _make_args()
        components = _install_driver_fakes(monkeypatch, args, events)

        def close_api() -> None:
            events.append("api_closed")

        async def final_audit() -> None:
            assert "api_closed" not in events
            assert "actor_dispose" not in events
            assert "inference_dispose" not in events
            events.append("audit")
            if audit_fails:
                raise RuntimeError("audit failed")

        monkeypatch.setattr(train_async_driver, "maybe_start_api_server", lambda *args, **kwargs: close_api)
        monkeypatch.setattr(components.rollout_executor, "_dispose", final_audit)

        if audit_fails:
            with pytest.raises(RuntimeError, match="audit failed"):
                await with_disposer(train_async_driver.train, args)
        else:
            await with_disposer(train_async_driver.train, args)

        assert events.count("audit") == 1
        assert events.index("audit") < events.index("api_closed") < events.index("actor_dispose")
        assert "inference_dispose" in events

    async def test_model_creation_failure_still_disposes_rollout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Startup failure before the API exists still releases the rollout components."""
        events: list[str] = []
        args = _make_args()
        _install_driver_fakes(monkeypatch, args, events)

        async def fail_creation(*args: Any) -> None:
            raise RuntimeError("model creation failed")

        monkeypatch.setattr(train_async_driver, "create_training_models", fail_creation)
        with pytest.raises(RuntimeError, match="model creation failed"):
            await with_disposer(train_async_driver.train, args)

        assert events == ["executor_dispose", "inference_dispose"]

    async def test_async_train_drains_eval_and_disposes_all_component_controllers(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A run ends only once its in-flight eval has settled and every component it created is disposed."""
        events: list[str] = []
        args = _make_args(use_critic=True, keep_old_actor=True, eval_interval=1, hf_checkpoint="/ckpt/hf")
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_async_driver.train, args)

        assert "eval:0" in events
        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]

    async def test_async_train_releases_its_worker_manager_after_eval_and_component_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The async driver releases its own manager after eval draining and component cleanup."""
        events: list[str] = []
        args = _make_args(use_critic=True, keep_old_actor=True, eval_interval=1, hf_checkpoint="/ckpt/hf")
        manager = FakeWorkerManager(events)
        _install_driver_fakes(monkeypatch, args, events)

        def init_orchestration_script(_args: SimpleNamespace, *, disposer: Any) -> FakeWorkerManager:
            disposer.add(partial(wiring.shutdown_worker_manager, manager))
            return manager

        monkeypatch.setattr(train_async_driver, "init_orchestration_script", init_orchestration_script)
        monkeypatch.setattr(wiring.ray, "kill", manager.kill)

        await with_disposer(train_async_driver.train, args)

        assert manager.killed == [manager]
        assert events.index("eval:0") < events.index("manager_shutdown")
        assert all(
            events.index(event) < events.index("manager_shutdown") for event in events if event.endswith("_dispose")
        )
        assert events.index("manager_shutdown") < events.index("manager_kill")


async def test_resume_at_final_joint_checkpoint_only_runs_final_eval(monkeypatch):
    args = _make_args(
        start_rollout_id=5, num_rollout=5, use_critic=True, keep_old_actor=True, max_keep=3, save="actor", critic_save="critic",
        megatron_config=None, eval_interval=20, eval_uses_snapshots=False, skip_eval_before_train=True,
    )
    events = []
    _install_driver_fakes(monkeypatch, args, events)
    await with_disposer(train_async_driver.train, args)
    assert "eval:4" in events
    assert not any(event.startswith("generate_start:") for event in events)
