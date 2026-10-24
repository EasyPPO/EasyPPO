"""Exercise the actual synchronous train driver and validation data path on CPU."""

import asyncio
import copy
import gzip
import json
import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agentic_critic_pipeline import validation_driver
from agentic_critic_pipeline.validation import prepare
from miles.ray.rollout.critic_validation import validation_pack
from miles.utils.data import RolloutDataPack, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.misc import should_run_periodic_action
from tests.agentic_critic.cpu_validation_harness import isolated_definition, launch_args, make_worker
from tests.agentic_critic.test_validation import _episode, _write


def consume_rng(args, model, prefix):
    """A supported forward hook must not change the next training update's RNG."""
    random.random()
    np.random.random()
    torch.rand(3)


@pytest.fixture
def frozen_plan(tmp_path):
    training, validation = [], []
    for problem in range(2):
        problem_id = f"frontiersmith:{problem}"
        training.append(_episode(f"train:{problem}", split="train", problem=problem_id, lengths=(3,)))
        for i in range(32):
            ep = _episode(f"validation:{problem}:{i}", problem=problem_id, lengths=(1040,) + (7,) * (i % 3))
            # Non-binary fractions catch float32 rounding or global/local target misalignment.
            ep["reward"] = (i % 5) / 7
            ep["scores"] = [ep["reward"] * 100]
            for j, segment in enumerate(ep["segments"]):
                length = len(segment["loss_mask"])
                response = [4 + j, 20] + [5 + i] * (length - 2)
                segment["input_ids"][2:] = response
                segment["loss_mask"][1] = 0
                # Keep feedback in the context but exclude it from every metric.
                ep["events"][2 * j + 1]["tokens"] = response[:1]
            events = []
            for j, segment in enumerate(ep["segments"]):
                response = segment["input_ids"][2:]
                events.extend(
                    [
                        ep["events"][2 * j],
                        ep["events"][2 * j + 1],
                        dict(tokens=response[1:2], actor=False, segment=j, kind="feedback"),
                        dict(tokens=response[2:], actor=True, segment=j, kind="action"),
                    ]
                )
            ep["events"] = events
            validation.append(ep)
    _write(tmp_path / "train.jsonl", training)
    _write(tmp_path / "validation.jsonl", validation)
    return prepare(tmp_path / "validation.jsonl", tmp_path / "train.jsonl", tmp_path / "replay")


async def noop(*args, **kwargs):
    pass


def run_six_batches(monkeypatch, tmp_path, mode, *, with_validation):
    args = launch_args(mode)
    vars(args).update(
        fully_async=False,
        use_critic=True,
        colocate_memory_peak_device="cpu",
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=6,
        start_rollout_id=0,
        eval_interval=None,
        num_critic_only_steps=50,
        save_trigger_sentinel=None,
        save_interval=None,
        debug_exit_after_rollout=None,
        offload_train=True,
        load_debug_rollout_data=str(tmp_path / "training_replay/rollout_{rollout_id}.pt"),
        critic_validation_plan=str(tmp_path / "replay/validation_plan.json") if with_validation else None,
        critic_validation_interval=5,
        critic_validation_output=str(tmp_path / mode),
        custom_megatron_before_log_prob_hook_path=f"{__name__}.consume_rng",
    )
    worker, evaluate, state, store = make_worker(monkeypatch, args)
    worker._asleep = False
    worker.wake_up = lambda: setattr(worker, "_asleep", False)
    model = worker.model[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    calls = []
    step5_weight = None

    async def train_batch(rollout_id, pack):
        nonlocal step5_weight
        worker.wake_up()
        optimizer.zero_grad()
        x = torch.rand(1, 5) + random.random() + np.random.random()
        model(x).square().mean().backward()
        optimizer.step()
        calls.append(("train", rollout_id + 1))
        if rollout_id == 4:
            step5_weight = model.weight.detach().item()
        return []

    async def evaluate_pack(*, batch, rollout_data_pack, output):
        assert not worker._asleep
        for rank in range(8):
            state.effective_dp.rank = rank
            evaluate(batch=batch, rollout_data_ref=rollout_data_pack.data_ref, output=output)
        calls.append(("validation", batch))

    async def get_validation(path, size):
        return validation_pack(args, replay_path=path, group_size=size, parallel={"dp_size": 8})

    async def get_training(rollout_id):
        return RolloutDataPack(data_ref=[])

    async def offload():
        worker._asleep = True

    critic = SimpleNamespace(train=train_batch, evaluate_critic=evaluate_pack, offload=offload)
    executor = SimpleNamespace(get=get_training, get_critic_validation=get_validation)
    inference = SimpleNamespace(prepare_rollout=noop)

    async def components(args):
        return inference, executor, 50

    async def models(args, rollout_executor):
        return None, critic

    train = isolated_definition(
        "train.py",
        "train",
        dict(
            init_orchestration_script=lambda *a, **kw: None,
            create_rollout_components=components,
            create_training_models=models,
            maybe_start_api_server=lambda *a, **kw: None,
            maybe_start_mini_ft_controller=lambda *a: None,
            update_weights=noop,
            EvalDispatcher=lambda *a: SimpleNamespace(drain=noop),
            remove_rollout_data_refs=remove_rollout_data_refs,
            remove_train_output_refs=remove_train_output_refs,
            should_run_periodic_action=should_run_periodic_action,
            os=os,
            logger=logging.getLogger(__name__),
        ),
    )
    monkeypatch.setattr(validation_driver.tracking, "define_step_key_metric_group", lambda *a: None)
    monkeypatch.setattr(validation_driver.tracking, "log", lambda *a, **kw: None)
    original_args = copy.deepcopy(vars(args))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    asyncio.run(train(args, disposer=SimpleNamespace(add=lambda *a: None)))
    assert not store.rows
    assert vars(args) == original_args
    assert model.training
    return dict(
        weight=model.weight.detach().clone(),
        optimizer=optimizer.state_dict(),
        calls=calls,
        random=(random.random(), np.random.random(), torch.rand(1).item()),
        step5_weight=step5_weight,
        releases=store.releases,
    )


@pytest.mark.parametrize("mode", ["shared", "full_history"])
def test_real_train_driver_resumes_after_validation_with_identical_state(monkeypatch, tmp_path, frozen_plan, mode):
    control = run_six_batches(monkeypatch, tmp_path, mode, with_validation=False)
    actual = run_six_batches(monkeypatch, tmp_path, mode, with_validation=True)
    assert actual["calls"] == [("train", i) for i in range(1, 6)] + [("validation", 5)] * 2 + [("train", 6)]
    assert actual["releases"] == 16
    assert torch.equal(control["weight"], actual["weight"])
    assert actual["optimizer"] == control["optimizer"]
    assert actual["random"] == control["random"]
    output = tmp_path / mode / "batch_0005"
    summary = json.loads((output / "metrics.json").read_text())
    assert summary["episodes"] == 64
    assert summary["profiles"] == {}
    assert not (output / "compaction_events.jsonl.gz").exists()
    assert len(list(output.glob("chunk_*/*.critic.pt"))) == 16
    with gzip.open(output / "selected_trajectories.jsonl.gz", "rt") as stream:
        traces = [json.loads(line) for line in stream]
    assert {row["episode_id"] for row in traces} == set(frozen_plan["selected_episode_ids"])
    with gzip.open(output / "values.jsonl.gz", "rt") as stream:
        values = {row["episode_id"]: row for row in map(json.loads, stream)}
    for path in output.glob("chunk_*/*.critic.pt"):
        dump = torch.load(path, weights_only=False)
        assert dump["schema_version"] == 2 and dump["value_bin_size"] == 100
        assert "values" not in dump and "loss_masks" not in dump and "returns" not in dump
        for row in dump["episodes"]:
            assert all(len(s["values"]) == (s["tokens"] + 99) // 100 for s in row["prediction"]["segments"])
    assert all(len(row["value_prediction"]) <= 13 for row in traces)
    token_mode = "full_history" if mode == "full_history" else "actor_summary"
    token_root = tmp_path / "training_replay/validation_values" / token_mode / "batch_0005"
    token_files = list(token_root.glob("chunk_*/*.critic.pt"))
    assert len(token_files) == 16
    token_predictions = {}
    for path in token_files:
        dump = torch.load(path, weights_only=False)
        keys = dump["rollout_ids" if mode == "full_history" else "sample_indices"]
        for key, value, mask in zip(keys, dump["values"], dump["loss_masks"], strict=True):
            assert key not in token_predictions
            token_predictions[key] = (value, torch.as_tensor(mask, dtype=torch.bool))
    all_predictions, all_targets = [], []
    for chunk in frozen_plan["replay"]:
        replay = torch.load(chunk["path"], weights_only=False)["samples"]
        for sample in replay:
            start = len(sample["tokens"]) - sample["response_length"]
            tokens = sample["tokens"]
            if mode == "full_history":
                tokens = sample["critic_tokens"]
                start = len(tokens) - sample["critic_response_length"] + sample["critic_span"][0]
            expected = [
                float(torch.tensor(tokens[start + i - 1]) * actual["step5_weight"])
                for i, bit in enumerate(sample["loss_mask"])
                if bit
            ]
            native_value, native_mask = token_predictions[
                sample["rollout_id" if mode == "full_history" else "index"]
            ]
            if mode == "full_history":
                offset, length = sample["critic_span"]
                native_value = native_value[offset : offset + length]
                native_mask = native_mask[offset : offset + length]
            assert native_mask.tolist() == sample["loss_mask"]
            assert native_value[native_mask].tolist() == pytest.approx(expected)
            row = values[sample["metadata"]["sample"]]
            segment = row["segments"][sample["metadata"]["segment"]]
            assert segment["values"] == pytest.approx(
                [np.mean(expected[i : i + 100]) for i in range(0, len(expected), 100)]
            )
            assert sum(segment["token_counts"]) == len(expected)
            assert "response_positions" not in segment
            all_predictions.extend(expected)
            all_targets.extend([sample["reward"]] * len(expected))
    ev = 1 - np.var(np.array(all_targets) - all_predictions) / np.var(all_targets)
    assert summary["ev"]["all"]["ev"] == pytest.approx(ev)
