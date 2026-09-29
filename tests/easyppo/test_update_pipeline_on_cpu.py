# Copyright 2026 EasyPPO contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Exercise the real trainer ordering and mini-batch dispatch without any model."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from transfer_queue import interface as tq_interface

from verl.trainer.ppo.v1 import trainer_base


@pytest.mark.parametrize("fail_actor", [False, True])
def test_v1_critic_sees_all_rows_actor_masks_then_restores(monkeypatch, fail_actor):
    masks = {"response_mask": torch.ones(1, 4), "loss_mask": torch.ones(1, 6)}
    events = []
    batch = KVBatchMeta(keys=["short", "long"], tags=[{"response_len": 2}, {"response_len": 4}], partition_id="train")

    def get(**kwargs):
        return TensorDict({key: masks[key].clone() for key in kwargs["select_fields"]}, batch_size=[1])

    def put(**kwargs):
        masks.update({key: value.clone() for key, value in kwargs["fields"].items()})

    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", put)

    def critic(b, **kwargs):
        assert masks["response_mask"].sum() == 4
        events.append("critic")
        return b

    def actor(b, **kwargs):
        assert masks["response_mask"].sum() == 0
        assert masks["loss_mask"].sum() == 0
        events.append("actor")
        if fail_actor:
            raise RuntimeError("test actor failure")
        return b

    identity = lambda b, **kwargs: b
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "data": {"filter_overlong_responses": True, "filter_overlong_responses_critic": False},
                "actor_rollout_ref": {"rollout": {"response_length": 4, "temperature": 1}},
                "trainer": {"critic_warmup": 31},
            }
        ),
        global_steps=31,
        replay_buffer=SimpleNamespace(sample=lambda **kwargs: (batch, {})),
        on_sample_begin=lambda: None,
        on_sample_end=lambda: None,
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[True]),
        use_critic=True,
        use_reference_policy=True,
        _balance_batch=identity,
        _compute_old_log_prob=identity,
        _compute_ref_log_prob=identity,
        _compute_values=identity,
        _compute_advantage=identity,
        _update_critic=critic,
        _update_actor=actor,
    )
    if fail_actor:
        with pytest.raises(RuntimeError, match="test actor failure"):
            trainer_base.PPOTrainer._step_once(trainer, {}, {}, 2)
    else:
        trainer_base.PPOTrainer._step_once(trainer, {}, {}, 2)
    assert events == ["critic", "actor"]
    assert masks["response_mask"].sum() == 4
    assert masks["loss_mask"].sum() == 6


@pytest.mark.parametrize(
    "filter_mode,global_step,long_length,fail_actor",
    [
        ("actor_only", 31, 4, False),
        ("actor_only", 31, 4, True),
        ("actor_only", 30, 4, False),
        ("actor_only", 31, 3, False),
        ("both", 31, 4, False),
        ("disabled", 31, 4, False),
    ],
)
def test_v1_filter_survives_transferqueue_writebacks(monkeypatch, filter_mode, global_step, long_length, fail_actor):
    # Keep TransferQueue's real kv_batch_put: only its storage transport is fake.
    # A field-only write returns persisted tags, not annotations on the input batch.
    keys = ["short", "long"]
    stored_tags = {"short": {"response_len": 2}, "long": {"response_len": long_length}}
    response_mask = torch.arange(4).unsqueeze(0) < torch.tensor([2, long_length]).unsqueeze(1)
    initial = TensorDict(
        {
            "response_mask": response_mask.long(),
            "loss_mask": torch.cat([torch.zeros(2, 2, dtype=torch.long), response_mask.long()], dim=1),
            "log_probs": torch.zeros(2, 4),
            "entropy": torch.ones(2, 4),
        },
        batch_size=[2],
    )
    stored = {key: initial[i].clone() for i, key in enumerate(keys)}
    events = []

    def retrieve_meta(keys, partition_id, create):
        return SimpleNamespace(
            keys=list(keys),
            size=len(keys),
            custom_meta=[deepcopy(stored_tags[key]) for key in keys],
            field_names=list(stored[keys[0]].keys()),
            extra_info={},
        )

    def store_fields(fields, metadata, **kwargs):
        for i, key in enumerate(metadata.keys):
            stored[key].update(fields[i].clone())
        metadata.field_names = list(stored[metadata.keys[0]].keys())
        return metadata

    def get(keys, partition_id, select_fields):
        return torch.stack([stored[key].select(*select_fields).clone() for key in keys])

    client = SimpleNamespace(kv_retrieve_meta=retrieve_meta, put=store_fields)
    monkeypatch.setattr(tq_interface, "_maybe_create_tq_client", lambda: client)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_get", get)
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", tq_interface.kv_batch_put)
    # Model outputs in this test are already response-aligned padded tensors.
    monkeypatch.setattr(trainer_base, "response_from_nested", lambda tensor, mask: tensor)

    def balance(batch, **kwargs):
        batch.reorder([1, 0])
        return batch

    def old_log_prob(batch, metrics):
        output = trainer_base.PPOTrainer._compute_old_log_prob(trainer, batch, metrics)
        assert all("overlong_filtered" not in tag for tag in output.tags)
        events.append("old_log_prob_write")
        return output

    def advantage(batch, **kwargs):
        # Exercise the second metadata replacement used by _compute_advantage.
        output = trainer_base.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict({"advantages": torch.ones(2, 4), "returns": torch.ones(2, 4)}, batch_size=[2]),
        )
        assert all("overlong_filtered" not in tag for tag in output.tags)
        events.append("advantage_write")
        return output

    is_overlong = long_length == 4

    def critic(batch, **kwargs):
        expected = 0 if filter_mode == "both" and is_overlong else long_length
        assert stored["long"]["response_mask"].sum().item() == expected
        assert stored["long"]["loss_mask"].sum().item() == expected
        events.append("critic")
        return batch

    def actor(batch, **kwargs):
        expected = 0 if filter_mode != "disabled" and is_overlong else long_length
        assert stored["long"]["response_mask"].sum().item() == expected
        assert stored["long"]["loss_mask"].sum().item() == expected
        assert stored["short"]["response_mask"].sum().item() == 2
        assert stored["short"]["loss_mask"].sum().item() == 2
        events.append("actor")
        if fail_actor:
            raise RuntimeError("test actor failure")
        return batch

    batch = KVBatchMeta(keys=keys, tags=[deepcopy(stored_tags[key]) for key in keys], partition_id="train")
    identity = lambda batch, **kwargs: batch
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "data": {
                    "filter_overlong_responses": filter_mode != "disabled",
                    "filter_overlong_responses_critic": filter_mode == "both",
                },
                "algorithm": {"rollout_correction": {"bypass_mode": False}},
                "actor_rollout_ref": {
                    "rollout": {"response_length": 4, "temperature": 1, "calculate_log_probs": False},
                    "actor": {"loss_agg_mode": "token-mean", "loss_scale_factor": None},
                },
                "trainer": {"critic_warmup": 31},
            }
        ),
        global_steps=global_step,
        replay_buffer=SimpleNamespace(sample=lambda **kwargs: (batch, {})),
        on_sample_begin=lambda: None,
        on_sample_end=lambda: None,
        reward_loop_manager=SimpleNamespace(reward_loop_worker_handles=[True]),
        use_critic=True,
        use_reference_policy=True,
        actor_rollout_wg=SimpleNamespace(compute_log_prob=identity),
        _balance_batch=balance,
        _compute_old_log_prob=old_log_prob,
        _compute_ref_log_prob=identity,
        _compute_values=identity,
        _compute_advantage=advantage,
        _update_critic=critic,
        _update_actor=actor,
    )
    if fail_actor:
        with pytest.raises(RuntimeError, match="test actor failure"):
            trainer_base.PPOTrainer._step_once(trainer, {}, {}, 2)
    else:
        trainer_base.PPOTrainer._step_once(trainer, {}, {}, 2)
    assert events == ["old_log_prob_write", "advantage_write", "critic"] + (["actor"] if global_step >= 31 else [])
    for key in keys:
        expected = 0 if key == "long" and filter_mode == "both" and is_overlong else stored_tags[key]["response_len"]
        assert stored[key]["response_mask"].sum().item() == expected
        assert stored[key]["loss_mask"].sum().item() == expected


def test_critic_dispatches_four_minibatches_and_one_epoch():
    config = OmegaConf.create(
        {
            "critic": {"ppo_mini_batch_size": 8, "ppo_epochs": 1, "data_loader_seed": 42, "shuffle": False},
            "actor_rollout_ref": {"rollout": {"n": 16, "temperature": 1}},
        }
    )
    worker = Mock()
    worker.train_mini_batch.return_value.get.return_value = {"metrics": {"mfu": [0.0]}}
    trainer = SimpleNamespace(config=config, critic_wg=worker)
    batch = SimpleNamespace(extra_info={})
    trainer_base.PPOTrainer._update_critic(trainer, batch, {})
    assert batch.extra_info["mini_batch_size"] == 128
    assert 512 // batch.extra_info["mini_batch_size"] == 4
    assert batch.extra_info["epochs"] == 1


def test_engine_recomputes_and_clips_each_update():
    from verl.workers.engine.base import BaseEngine
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    # A scalar parameter and known gradients verify the real engine's clipping path.
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    module = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    seen = []

    def backward(data, loss_function, forward_only):
        seen.append(parameter.item())
        (parameter * 100).backward()
        return {"metrics": {}}

    engine = SimpleNamespace(
        optimizer=optimizer,
        module=module,
        optimizer_config=SimpleNamespace(clip_grad=1.0),
        optimizer_zero_grad=optimizer.zero_grad,
        forward_backward_batch=backward,
        is_mp_src_rank_with_outputs=lambda: True,
        _qat_enabled=False,
    )
    engine.optimizer_step = lambda: FSDPEngine.optimizer_step(engine)
    for _ in range(4):
        BaseEngine.train_batch(engine, TensorDict({}, batch_size=[]), None)
    assert seen == pytest.approx([0.0, -0.1, -0.2, -0.3])
    assert parameter.item() == pytest.approx(-0.4)
