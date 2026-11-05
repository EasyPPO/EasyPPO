"""Independent critic episode packing preserves every actor response position."""

import math
from collections import defaultdict
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from agentic_critic.critic_input import Segment
from agentic_critic.miles_samples import critic_episode
from tests.fast.fixtures.driver_fakes import FakeObjectStore
from tests.fast.fixtures.score_centering_fixtures import _args as _conversion_args

from miles.ray.rollout import rollout_executor as executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.rollout.train_data_conversion import build_critic_samples, split_full_history_shards
from miles.utils import object_store
from miles.utils.types import Sample


def _samples():
    actor = [Segment([1, 2, 3, 4], 2, [1, 1]), Segment([1, 2, 3, 4, 5, 6, 7], 2, [0, 1])]
    memory = [actor[0], Segment([90, 6, 7], 2, [0, 1])]
    episode, spans = critic_episode(actor)
    output = []
    for i, (part, span) in enumerate(zip(actor, spans, strict=True)):
        sample = Sample(
            index=i,
            rollout_id=100,
            tokens=part.tokens,
            response_length=part.response_length,
            loss_mask=part.loss_mask,
            metadata={"segment": i},
            reward=0.8,
        )
        sample.critic_tokens, sample.critic_response_length = episode.tokens, episode.response_length
        sample.critic_loss_mask, sample.critic_span = episode.loss_mask, span
        sample.critic_segments = [asdict(part) for part in memory]
        sample.critic_memory_exact = True
        output.append(sample)
    return output


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_memory_modes_build_own_episode_and_return_actor_spans(mode):
    samples = _samples()
    critic, spans = build_critic_samples(samples, context_mode=mode)
    assert len(critic) == 1
    expected_tokens = [1, 2, 3, 4, 90, 6, 7] if mode == "full_detach" else samples[0].tokens + samples[1].tokens
    assert critic[0].tokens == expected_tokens
    args = SimpleNamespace(
        critic_context_mode=mode,
        use_dynamic_batch_size=True,
        global_batch_size=1,
        max_tokens_per_gpu=100,
        critic_num_steps_per_rollout=1,
        balance_data=False,
    )

    def rows(items):
        return {"tokens": [s.tokens for s in items], "response_lengths": [s.response_length for s in items]}

    actor_shards, critic_shards = split_full_history_shards(args, rows(samples), rows(critic), spans, {"dp_size": 1})
    response = critic[0].tokens[-critic[0].response_length :]
    shipped = [response[start : start + length] for _, start, length in critic_shards[0]["value_slices"]]
    assert shipped == [sample.tokens[-sample.response_length :] for sample in samples]
    assert actor_shards[0]["tokens"] == [sample.tokens for sample in samples]


def test_full_detach_rejects_untracked_template_boundary():
    samples = _samples()
    samples[1].critic_memory_exact = False
    with pytest.raises(ValueError, match="explicit compaction boundaries"):
        build_critic_samples(samples, context_mode="full_detach")


def test_wrong_critic_response_is_rejected():
    samples = _samples()
    samples[0].critic_segments[-1]["tokens"][-1] = 99
    with pytest.raises(ValueError, match="exactly match"):
        build_critic_samples(samples, context_mode="full_detach")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
async def test_executor_sc_actor_and_memory_critic_have_separate_conversion_contracts(monkeypatch, mode):
    """Exercise get()'s actual conversions and DP packaging, not a config-copy helper."""
    samples = _samples()
    for sample in samples:
        sample.status = Sample.Status.COMPLETED
        sample.rollout_log_probs = [-math.log(128) if mask else 0 for mask in sample.loss_mask]
        sample.rollout_topk_token_ids = np.tile(np.arange(128, dtype=np.int32), (sample.response_length, 1))
        sample.rollout_topk_log_probs = np.full((sample.response_length, 128), -math.log(128), dtype=np.float32)
        for row, mask in enumerate(sample.loss_mask):
            if not mask:
                sample.rollout_topk_token_ids[row] = -1
                sample.rollout_topk_log_probs[row] = -np.inf
    args = _conversion_args(
        loss_type="score_centering",
        advantage_estimator="ppo",
        critic_context_mode=mode,
        rollout_top_logprobs_num=128,
        use_dynamic_batch_size=True,
        global_batch_size=1,
        max_tokens_per_gpu=100,
        critic_num_steps_per_rollout=1,
        delay_split_train_data_by_dp=False,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_skip_weight_update=False,
        update_weights_interval=1,
        lora_rank=0,
    )
    executor = object.__new__(RolloutExecutor)
    executor.args = args
    executor.data_source = SimpleNamespace()
    executor._rollouts_since_publish_of_model_id = defaultdict(int)
    executor._train_parallel_configs_of_model_id = {None: {"dp_size": 1}}
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor._get_rollout_data = AsyncMock(return_value=(samples, {}, {}))
    store = FakeObjectStore()
    monkeypatch.setattr(object_store, "_INSTANCE", store)
    monkeypatch.setattr(executor_module, "save_debug_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(executor_module, "log_rollout_data", lambda *a, **kw: None)
    pack = await executor.get(0)
    actor = store.get(pack.data_ref[0]).value
    critic = store.get(pack.for_critic().data_ref[0]).value
    assert args.rollout_top_logprobs_num == 128  # Only the critic's copied args disable candidates.
    assert len(actor["tokens"]) == 2 and len(critic["tokens"]) == 1
    for i, sample in enumerate(samples):
        np.testing.assert_array_equal(actor["rollout_topk_token_ids"][i], sample.rollout_topk_token_ids)
        np.testing.assert_array_equal(actor["rollout_topk_log_probs"][i], sample.rollout_topk_log_probs)
        assert actor["rollout_log_probs"][i] == sample.rollout_log_probs
    assert not any(key.startswith("rollout_topk") or key.startswith("rollout_sampling_mask") for key in critic)
    assert "rollout_log_probs" not in critic
    assert "critic_segments" in critic and len(critic["critic_segments"][0]) == 2
    expected = (
        samples[0].critic_segments
        if mode == "full_detach"
        else [dict(tokens=s.tokens, response_length=s.response_length, loss_mask=s.loss_mask) for s in samples]
    )
    assert critic["critic_segments"][0] == expected
    response = critic["tokens"][0][-critic["response_lengths"][0] :]
    values = [response[start : start + length] for _, start, length in critic["value_slices"]]
    assert values == [s.tokens[-s.response_length :] for s in samples]
