"""Fail before training on unsupported shapes; retain useful output on errors."""

import asyncio
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agentic_critic_pipeline import validation, validation_driver
from miles.ray.rollout import critic_validation
from miles.utils.data import RolloutDataPack
from tests.agentic_critic.cpu_validation_harness import isolated_definition, launch_args, make_worker
from tests.fast.ray.rollout.test_critic_shards import _episode as samples_of


@pytest.mark.parametrize(
    "flag",
    [
        None,
        "delay_split_train_data_by_dp",
        "enable_witness",
        "use_dynamic_global_batch_size",
        "load_debug_rollout_data",
    ],
)
def test_validation_launch_checks_supported_configuration(monkeypatch, flag):
    args = SimpleNamespace(
        train_backend="megatron",
        use_critic=True,
        fully_async=False,
        entry="train",
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        critic_validation_interval=5,
        critic_validation_output="out",
        critic_validation_plan="plan",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=1,
        load_debug_rollout_data="replay/rollout_{rollout_id}.pt",
    )
    monkeypatch.setattr(validation, "load_plan", lambda *a, **kw: {"group_size": 32})
    if flag is None:
        validation.validate_launch(args)
        return
    setattr(args, flag, None if flag == "load_debug_rollout_data" else True)
    with pytest.raises(ValueError, match=flag.replace("_", "-")):
        validation.validate_launch(args)


def test_partial_store_upload_releases_successful_shards(monkeypatch):
    args = launch_args("shared")
    _, _, _, store = make_worker(monkeypatch, args)
    samples = [s for i in range(32) for s in samples_of(i, i * 100)[:2]]
    for sample in samples:
        sample.group_index = 0
    monkeypatch.setattr(critic_validation, "load_debug_rollout_data", lambda *a: (samples, {}))
    put = store.put

    def fail_third(**kwargs):
        if store.next_id == 2:
            raise OSError("store full")
        return put(**kwargs)

    monkeypatch.setattr(store, "put", fail_third)
    with pytest.raises(OSError, match="store full"):
        critic_validation.validation_pack(args, replay_path="unused", group_size=32, parallel={"dp_size": 8})
    assert not store.rows


def test_partial_validation_directory_is_not_reused(monkeypatch, tmp_path):
    output = tmp_path / "batch_0005"
    output.mkdir()
    (output / "interrupted.critic.pt").write_text("previous attempt")
    monkeypatch.setattr(validation_driver, "load_plan", lambda *a: {"replay": [], "group_size": 32})
    args = SimpleNamespace(
        critic_validation_plan="plan", critic_validation_output=str(tmp_path), critic_context_mode="shared"
    )
    with pytest.raises(FileExistsError, match="Validation"):
        asyncio.run(validation_driver.evaluate(args, None, None, 4))
    assert (output / "interrupted.critic.pt").read_text() == "previous attempt"


@pytest.mark.parametrize("stage", ["forward", "dump"])
def test_failed_validation_restores_rng_modes_and_releases_borrowed_data(monkeypatch, tmp_path, stage):
    args = launch_args("shared")
    args.load_debug_rollout_data = str(tmp_path / "replay/rollout_{rollout_id}.pt")
    worker, evaluate, _, store = make_worker(monkeypatch, args, dp_size=1)
    samples = list(samples_of(0, 0)[:2])
    for sample in samples:
        sample.group_index = 0
    monkeypatch.setattr(critic_validation, "load_debug_rollout_data", lambda *a: (samples, {}))
    pack = critic_validation.validation_pack(args, replay_path="unused", group_size=1, parallel={"dp_size": 1})
    model = worker.model[0]
    model.dropout.eval()  # Preserve a deliberately mixed child-module mode.
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())

    def fail(*a, **kw):
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError(f"{stage} failed")

    if stage == "forward":
        monkeypatch.setattr(model, "forward", fail)
    else:
        monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        evaluate(batch=5, rollout_data_ref=pack.data_ref, output=str(tmp_path / "{rank}.pt"))
    assert model.training and not model.dropout.training
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
    assert store.releases == 1
    assert not list(tmp_path.glob("*.pt"))


def test_driver_failure_removes_shards_and_does_not_publish_metrics(monkeypatch, tmp_path):
    args = launch_args("shared")
    vars(args).update(critic_validation_plan="plan", critic_validation_output=str(tmp_path))
    _, _, _, store = make_worker(monkeypatch, args)
    refs = [store.put(value={}, value_spec={}) for _ in range(8)]
    monkeypatch.setattr(validation_driver, "load_plan", lambda *a: {"replay": [{"path": "chunk"}], "group_size": 32})

    async def get(*args):
        return RolloutDataPack(data_ref=refs)

    async def fail(**kwargs):
        raise RuntimeError("rank failed")

    with pytest.raises(RuntimeError, match="rank failed"):
        asyncio.run(
            validation_driver.evaluate(
                args,
                SimpleNamespace(evaluate_critic=fail),
                SimpleNamespace(get_critic_validation=get),
                4,
            )
        )
    assert not store.rows
    assert not (tmp_path / "batch_0005/metrics.json").exists()


@pytest.mark.parametrize("fault", [None, "source", "overlap", "duplicate", "problem"])
def test_training_validation_pairing(tmp_path, fault):
    plan = dict(training_source_sha256="same", episodes={"validation:0": {"problem_id": "frontiersmith:0"}})
    rows = [{"episode_id": "train:0", "problem": "frontiersmith:0"}]
    metadata = dict(source_sha256="same", episodes=rows)
    if fault == "source":
        metadata["source_sha256"] = "different"
    elif fault == "overlap":
        rows[0]["episode_id"] = "validation:0"
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "problem":
        rows[0]["problem"] = "frontiersmith:8"
    (tmp_path / "episodes.identities.json").write_text(json.dumps(metadata))
    if fault is None:
        validation.validate_training_replay(plan, tmp_path)
    else:
        with pytest.raises(ValueError):
            validation.validate_training_replay(plan, tmp_path)


@pytest.mark.parametrize("stage", ["executor", "controller", "worker"])
def test_validation_rpc_payload_roundtrip(stage):
    from miles.utils.object_store import StoreObjectRef, _MooncakeStoreObjectRef
    from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs

    refs = [_MooncakeStoreObjectRef(payload=i) for i in range(8)]
    pack = RolloutDataPack(sample_indices=[1, 2], data_ref=refs)
    if stage == "executor":
        path, owner, name = "miles/ray/rollout/rollout_executor.py", "RolloutExecutor", "get_critic_validation"
        query, result = dict(replay_path="/validation/rollout_0.pt", group_size=32), pack
    elif stage == "controller":
        path, owner, name = "miles/ray/train/group.py", "TrainerController", "evaluate_critic"
        query, result = dict(batch=5, rollout_data_pack=pack, output="/validation/{rollout_id}_{rank}.pt"), None
    else:
        path, owner, name = "miles/backends/megatron_utils/actor.py", "MegatronTrainRayActor", "evaluate_critic"
        query, result = dict(batch=5, rollout_data_ref=refs, output="/validation/{rollout_id}_{rank}.pt"), None
    method = isolated_definition(
        path, name, dict(StoreObjectRef=StoreObjectRef, RolloutDataPack=RolloutDataPack), owner=owner
    )
    cls = type(owner, (), {name: method})
    serializer = collect_rpc_method_specs(cls)[name].serializer
    wire = json.loads(json.dumps(serializer.encode_query(query)))
    assert serializer.decode_query(wire) == query
    wire_result = json.loads(json.dumps(serializer.encode_result(result)))
    assert serializer.decode_result(wire_result) == result
