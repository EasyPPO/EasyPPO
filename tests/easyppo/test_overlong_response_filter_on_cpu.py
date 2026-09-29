# Copyright 2026 FrontierReward contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import numpy as np
import pytest
import torch
from transfer_queue import KVBatchMeta

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    compute_overlong_response_mask,
    filter_overlong_response_kv_batch,
    filter_overlong_responses,
    mask_overlong_response_rows,
    set_initial_response_clip_ratio,
)


def _make_batch() -> DataProto:
    tensors = {
        "responses": torch.tensor(
            [
                [11, 12, 0, 0],
                [21, 22, 23, 24],
                [0, 0, 0, 0],
                [31, 32, 33, 0],
            ],
            dtype=torch.long,
        ),
        "response_mask": torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
        "loss_mask": torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
        "token_level_scores": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={"uid": np.array(["keep_short", "drop_overlong", "keep_aborted", "keep_near_limit"], dtype=object)},
    )


def test_initial_clip_ratio_uses_pre_filter_counts():
    data_metrics = {"response_length/clip_ratio": 0.0}
    filter_metrics = {
        "training/overlong_response_filter/total": 8,
        "training/overlong_response_filter/dropped": 3,
    }

    updated = set_initial_response_clip_ratio(data_metrics, filter_metrics)

    assert updated["response_length/clip_ratio"] == pytest.approx(3 / 8)


def test_compute_overlong_response_mask_marks_max_length_only():
    batch = _make_batch()

    overlong_mask = compute_overlong_response_mask(batch, max_response_length=4)

    assert overlong_mask.tolist() == [False, True, False, False]


def test_training_filter_masks_overlong_responses_and_preserves_batch_shape():
    batch = _make_batch()

    filtered, metrics = filter_overlong_responses(
        batch,
        enabled=True,
        max_response_length=4,
        metrics_prefix="training",
    )

    assert len(filtered) == 4
    assert filtered.non_tensor_batch["uid"].tolist() == [
        "keep_short",
        "drop_overlong",
        "keep_aborted",
        "keep_near_limit",
    ]
    assert filtered.batch["responses"].tolist() == [
        [11, 12, 0, 0],
        [21, 22, 23, 24],
        [0, 0, 0, 0],
        [31, 32, 33, 0],
    ]
    assert filtered.batch["response_mask"].tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
    ]
    assert filtered.batch["loss_mask"].tolist() == [
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 0],
    ]
    assert filtered.batch["overlong_filtered"].tolist() == [False, True, False, False]
    assert metrics == {
        "training/overlong_response_filter/total": 4,
        "training/overlong_response_filter/kept": 3,
        "training/overlong_response_filter/dropped": 1,
        "training/overlong_response_filter/drop_ratio": 0.25,
    }


def test_actor_only_filter_defers_masking_until_after_critic_update():
    batch = _make_batch()
    original_response_mask = batch.batch["response_mask"].clone()
    original_loss_mask = batch.batch["loss_mask"].clone()

    filtered, metrics = filter_overlong_responses(
        batch,
        enabled=True,
        max_response_length=4,
        metrics_prefix="training",
        mask_dropped_rows=False,
    )

    assert torch.equal(filtered.batch["response_mask"], original_response_mask)
    assert torch.equal(filtered.batch["loss_mask"], original_loss_mask)
    assert filtered.batch["overlong_filtered"].tolist() == [False, True, False, False]
    assert metrics["training/overlong_response_filter/dropped"] == 1

    masked = mask_overlong_response_rows(filtered)
    assert masked.batch["response_mask"][1].tolist() == [0, 0, 0, 0]
    assert masked.batch["loss_mask"][1].tolist() == [0, 0, 0, 0]
    assert masked.batch["response_mask"][0].tolist() == [1, 1, 0, 0]


def test_training_filter_uses_configured_max_response_length_not_tensor_width():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 1, dtype=torch.long),
            "response_mask": torch.ones(2, 1, dtype=torch.long),
        },
        non_tensors={"uid": np.array(["short-a", "short-b"], dtype=object)},
    )

    filtered, metrics = filter_overlong_responses(
        batch,
        enabled=True,
        max_response_length=32768,
        metrics_prefix="training",
    )

    assert len(filtered) == 2
    assert metrics == {
        "training/overlong_response_filter/total": 2,
        "training/overlong_response_filter/kept": 2,
        "training/overlong_response_filter/dropped": 0,
        "training/overlong_response_filter/drop_ratio": 0,
    }


def test_validation_filter_is_noop_even_when_enabled():
    batch = _make_batch()

    filtered, metrics = filter_overlong_responses(
        batch,
        enabled=True,
        max_response_length=4,
        metrics_prefix="validation",
    )

    assert len(filtered) == 4
    assert filtered.non_tensor_batch["uid"].tolist() == batch.non_tensor_batch["uid"].tolist()
    assert metrics == {}


def test_filter_raises_when_every_training_response_is_overlong():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 4, dtype=torch.long),
            "response_mask": torch.ones(2, 4, dtype=torch.long),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )

    with pytest.raises(ValueError, match="removed every training sample"):
        filter_overlong_responses(batch, enabled=True, max_response_length=4, metrics_prefix="training")


def test_kv_training_filter_masks_rows_and_preserves_cardinality_and_metadata():
    batch = KVBatchMeta(
        partition_id="train",
        keys=["a_0_0", "b_0_0", "c_0_0"],
        tags=[
            {"response_len": 2, "seq_len": 6},
            {"response_len": 4, "seq_len": 8},
            {"response_len": 3, "seq_len": 7},
        ],
        fields=["response_mask"],
        extra_info={"temperature": 0.7},
    )
    cleared = {}

    filtered, metrics = filter_overlong_response_kv_batch(
        batch,
        enabled=True,
        max_response_length=4,
        metrics_prefix="training",
        mask_dropped_fn=lambda **kwargs: cleared.update(kwargs),
    )

    assert filtered.keys == ["a_0_0", "b_0_0", "c_0_0"]
    assert filtered.tags == [
        {"response_len": 2, "seq_len": 6, "overlong_filtered": False},
        {"response_len": 4, "seq_len": 8, "overlong_filtered": True},
        {"response_len": 3, "seq_len": 7, "overlong_filtered": False},
    ]
    assert filtered.partition_id == "train"
    assert filtered.fields == ["response_mask"]
    assert filtered.extra_info == {"temperature": 0.7}
    assert cleared == {"keys": ["b_0_0"], "partition_id": "train"}
    assert metrics == {
        "training/overlong_response_filter/total": 3,
        "training/overlong_response_filter/kept": 2,
        "training/overlong_response_filter/dropped": 1,
        "training/overlong_response_filter/drop_ratio": 1 / 3,
    }


def test_kv_validation_filter_is_noop_even_when_enabled():
    batch = KVBatchMeta(
        partition_id="val",
        keys=["a_0_0", "b_0_0"],
        tags=[{"response_len": 4}, {"response_len": 1}],
    )

    filtered, metrics = filter_overlong_response_kv_batch(
        batch,
        enabled=True,
        max_response_length=4,
        metrics_prefix="validation",
    )

    assert filtered.keys == ["a_0_0", "b_0_0"]
    assert metrics == {}
