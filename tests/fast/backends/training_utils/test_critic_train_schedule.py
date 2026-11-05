"""EasyPPO critic mini-batches: `get_critic_train_schedule` re-splits a rank's rollout shard."""

import ast
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.fast.fixtures.driver_fakes import FakeObjectStore

from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.data.rollout import get_critic_train_schedule, get_data_iterator
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("critic-train-schedule") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _trivial_parallel_state() -> None:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=trivial_group,
            intra_dp_cp=trivial_group,
            cp=trivial_group,
            tp=trivial_group,
            pp=trivial_group,
            ep=trivial_group,
            etp=trivial_group,
            indep_dp=trivial_group,
        )
    )


def _rollout_data(total_lengths: list[int]) -> dict:
    return {"total_lengths": total_lengths, "sample_ids": list(range(len(total_lengths)))}


def _drain(data_iterator, num_microbatches: list[int]) -> list[list[list[int]]]:
    """Sample ids of every micro-batch, grouped by critic step."""
    [iterator] = data_iterator
    iterator.reset()
    return [[iterator.get_next(["sample_ids"])["sample_ids"] for _ in range(count)] for count in num_microbatches]


def test_static_schedule_visits_every_sample_once_in_equal_steps():
    args = Namespace(critic_num_steps_per_rollout=4, seed=1234, use_dynamic_batch_size=False, micro_batch_size=2)
    rollout_data = _rollout_data([10] * 16)

    data_iterator, num_microbatches, num_rollouts = get_critic_train_schedule(
        args, rollout_data, rollout_id=3, num_rollouts=[16]
    )

    assert num_microbatches == [2, 2, 2, 2]
    assert num_rollouts == [4, 4, 4, 4]
    steps = _drain(data_iterator, num_microbatches)
    assert all(len(micro_batch) == 2 for step in steps for micro_batch in step)
    visited = [i for step in steps for micro_batch in step for i in micro_batch]
    assert sorted(visited) == list(range(16))
    assert visited != list(range(16)), "critic mini-batches are shuffled"


def test_schedule_depends_on_the_rollout_but_is_reproducible():
    args = Namespace(critic_num_steps_per_rollout=4, seed=1234, use_dynamic_batch_size=False, micro_batch_size=1)

    def visited(rollout_id: int) -> list[int]:
        data_iterator, num_microbatches, _ = get_critic_train_schedule(
            args, _rollout_data([10] * 32), rollout_id=rollout_id, num_rollouts=[32]
        )
        return [i for step in _drain(data_iterator, num_microbatches) for micro_batch in step for i in micro_batch]

    assert visited(0) == visited(0)
    assert visited(0) != visited(1)


def test_dynamic_schedule_packs_each_step_under_the_token_budget(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    args = Namespace(critic_num_steps_per_rollout=2, seed=7, use_dynamic_batch_size=True, max_tokens_per_gpu=100)
    total_lengths = [60, 60, 30, 30, 30, 30, 90, 10]

    data_iterator, num_microbatches, num_rollouts = get_critic_train_schedule(
        args, _rollout_data(total_lengths), rollout_id=0, num_rollouts=[8]
    )

    assert num_rollouts == [4, 4]
    steps = _drain(data_iterator, num_microbatches)
    assert [sum(len(micro_batch) for micro_batch in step) for step in steps] == [4, 4]
    assert sorted(i for step in steps for micro_batch in step for i in micro_batch) == list(range(8))


def test_step_count_must_divide_the_rollout_count():
    args = Namespace(critic_num_steps_per_rollout=4, seed=0, use_dynamic_batch_size=False, micro_batch_size=1)
    with pytest.raises(AssertionError, match="does not divide"):
        get_critic_train_schedule(args, _rollout_data([10] * 6), rollout_id=0, num_rollouts=[6])


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_prompt_group_schedule_is_consumed_by_both_native_iterator_entries(mode):
    args = Namespace(
        critic_context_mode=mode,
        critic_num_steps_per_rollout=2,
        use_dynamic_global_batch_size=False,
        qkv_format="thd",
    )
    data = {
        **_rollout_data([10] * 8),
        "micro_batch_indices": [[3, 0], [6], [2, 7], [1, 4, 5]],
        "num_microbatches": [2, 2],
        "num_rollouts": [2, 2],
    }
    forward, forward_counts = get_data_iterator(args, [], data)
    training, counts, rollouts = get_critic_train_schedule(args, data, rollout_id=123, num_rollouts=[4])
    assert forward_counts == counts == [2, 2] and rollouts == [2, 2]
    assert _drain(training, counts) == _drain(forward, forward_counts) == [[[3, 0], [6]], [[2, 7], [1, 4, 5]]]


def test_memory_schedule_refuses_missing_or_mismatched_precomputed_steps():
    args = Namespace(
        critic_context_mode="full_detach", critic_num_steps_per_rollout=2, use_dynamic_global_batch_size=False
    )
    with pytest.raises(ValueError, match="rollout-computed"):
        get_critic_train_schedule(args, _rollout_data([10] * 4), rollout_id=0, num_rollouts=[4])
    data = {**_rollout_data([10] * 4), "micro_batch_indices": [[0], [1], [2], [3]], "num_microbatches": [4]}
    with pytest.raises(ValueError, match="does not match"):
        get_critic_train_schedule(args, data, rollout_id=0, num_rollouts=[4])


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_native_critic_fixes_all_old_values_before_four_preplanned_updates(monkeypatch, mode):
    # Exercise the native orchestration body without importing optional
    # Megatron/CUDA initialization into this CPU schedule regression suite.
    source = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_train_critic")
    method.decorator_list = []
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: SimpleNamespace(vpp_size=1))
    store = FakeObjectStore()
    monkeypatch.setattr(object_store, "_INSTANCE", store)
    args = Namespace(
        global_batch_size=4,
        critic_context_mode=mode,
        critic_num_steps_per_rollout=4,
        use_dynamic_global_batch_size=False,
        qkv_format="thd",
    )
    worker = SimpleNamespace(
        args=args, model=None, optimizer=None, opt_param_scheduler=None, _heartbeat=SimpleNamespace(bump=lambda: None)
    )
    data = {
        "sample_ids": [0, 1, 2, 3],
        "total_lengths": [2] * 4,
        "micro_batch_indices": [[3], [0], [2], [1]],
        "num_microbatches": [1] * 4,
        "num_rollouts": [1] * 4,
        "value_slices": [(2, 0, 1), (0, 0, 1), (3, 0, 1), (1, 0, 1)],
    }
    events = []
    old_values = [torch.tensor([float(row + 1)]) for row in range(4)]

    def forward(_compute, _args, _model, iterators, counts, **_kwargs):
        events.append("all-old-values")
        assert _drain(iterators, counts) == [[[3]], [[0]], [[2]], [[1]]]
        return {"values": old_values}

    def advantages(_args, rollout_data):
        events.append("all-returns")
        assert rollout_data["values"] is old_values

    def train(_rollout, _model, _optimizer, _scheduler, iterators, counts, rollouts, **_kwargs):
        assert worker.args.loss_type == "value_loss" and counts == rollouts == [1] * 4
        assert events == ["all-old-values", "all-returns"]
        for step, microbatches in enumerate(_drain(iterators, counts)):
            events.append(f"update-{step}")
            assert data["values"] is old_values
            row = microbatches[0][0]
            assert data["values"][row].item() == row + 1
        return TrainStepOutcome.NORMAL

    namespace = {
        "RolloutBatch": dict,
        "TrainStepOutput": TrainStepOutput,
        "TrainStepOutcome": TrainStepOutcome,
        "get_data_iterator": get_data_iterator,
        "get_critic_train_schedule": get_critic_train_schedule,
        "get_num_rollouts": data_utils.get_num_rollouts,
        "get_parallel_state": lambda: SimpleNamespace(is_pp_last_stage=True),
        "forward_only": forward,
        "get_values": object(),
        "compute_advantages_and_returns": advantages,
        "train_dump": SimpleNamespace(save_debug_critic_values=lambda *a, **k: None),
        "train": train,
        "object_store": object_store,
        "CRITIC_VALUES_VALUE_SPEC": {},
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    output = namespace["_train_critic"](worker, rollout_id=7, rollout_data=data)
    with store.get(output.values) as stored:
        assert [value.item() for value in stored["values"]] == [3.0, 1.0, 4.0, 2.0]
    assert events == ["all-old-values", "all-returns", "update-0", "update-1", "update-2", "update-3"]
