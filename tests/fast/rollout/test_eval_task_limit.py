"""Fixed evaluation task budgets reach both rollout backends before task creation."""

import base64
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.rollout import sglang_rollout
from miles.rollout.inference_rollout import inference_rollout_eval
from miles.utils.arguments import _resolve_eval_datasets
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.types import Sample


def test_eval_config_merges_dataset_limits_and_defaults():
    document = {
        "defaults": {"max_samples": 16, "sample_seed": 42},
        "datasets": [
            {"name": "tb21_quick", "path": "/tb.jsonl"},
            {"name": "fcs_quick", "path": "/fcs.jsonl", "max_samples": 8, "sample_seed": 7},
        ],
    }
    args = SimpleNamespace(
        eval_config="base64:" + base64.b64encode(json.dumps(document).encode()).decode(),
        eval_prompt_data=None,
    )
    configs = _resolve_eval_datasets(args)
    assert [(c.max_samples, c.sample_seed) for c in configs] == [(16, 42), (8, 7)]
    original = list(range(89))
    first = configs[0].select_samples(original)
    assert len(first) == len(set(first)) == 16
    assert first == configs[0].select_samples(original) and original == list(range(89))
    assert first != EvalDatasetConfig("other", "/tb.jsonl", max_samples=16, sample_seed=7).select_samples(original)


def test_fullset_eval_inherits_five_samples_per_prompt_instead_of_training_fanout():
    document = {"datasets": [{"name": "fcs", "path": "/fcs.jsonl", "max_samples": None}]}
    args = SimpleNamespace(
        eval_config="base64:" + base64.b64encode(json.dumps(document).encode()).decode(),
        eval_prompt_data=None,
        n_samples_per_eval_prompt=5,
        n_samples_per_prompt=16,
    )
    [config] = _resolve_eval_datasets(args)
    assert config.n_samples_per_eval_prompt == 5
    assert config.max_samples is None
    original = list(range(179))
    assert config.select_samples(original) is original


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_task_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="max_samples"):
        EvalDatasetConfig("bad", "/data", max_samples=limit)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_subset_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="sample_seed"):
        EvalDatasetConfig("bad", "/data", max_samples=16, sample_seed=seed)


@pytest.mark.parametrize("backend", ["native", "legacy"])
@pytest.mark.parametrize("task_count,fanout", [(89, 1), (89, 5), (179, 5)])
async def test_subset_and_fullset_fanout_create_independent_episodes_without_mutating_cache(
    monkeypatch, backend, task_count, fanout
):
    module = inference_rollout_eval if backend == "native" else sglang_rollout
    raw = [Sample(prompt=f"task-{i}", metadata={"task_id": i}) for i in range(task_count)]
    created, sampled = [], []

    class Dataset:
        def __init__(self, **kwargs):
            self.samples = raw
            created.append(self)

    async def generate(_state, sample, *, sampling_params, evaluation):
        assert evaluation
        assert sampling_params["temperature"] == sampling_params["top_p"] == 1
        assert sampling_params["top_k"] == -1
        sampled.append(sample.metadata["task_id"])
        sample.reward, sample.response = 1.0, "done"
        return sample

    monkeypatch.setattr(module, "Dataset", Dataset)
    monkeypatch.setattr(module, "load_tokenizer", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_processor", lambda *a, **kw: None)
    monkeypatch.setattr(module, "generate_and_rm", generate)
    args = SimpleNamespace(
        group_rm=False,
        hf_checkpoint="/model",
        apply_chat_template=False,
        chat_template_path=None,
        eval_max_prompt_len=None,
        multimodal_keys=None,
        apply_chat_template_kwargs=None,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_router_policy="random",
        eval_reward_key=None,
        reward_key=None,
    )
    cache = {}
    if backend == "legacy":
        monkeypatch.setattr(module, "EVAL_PROMPT_DATASET", cache)

    async def run(config):
        if backend == "native":
            return await module.eval_rollout_single_dataset(SimpleNamespace(args=args), config, cache)
        return await module.eval_rollout_single_dataset(args, 0, config)

    def config(name, limit):
        return EvalDatasetConfig(
            name,
            "/tb.jsonl",
            max_samples=limit,
            sample_seed=42,
            n_samples_per_eval_prompt=fanout,
            temperature=1,
            top_p=1,
            top_k=-1,
            max_response_len=8192,
        )

    limited = await run(config("tb21_quick", 16))
    assert len(limited["tb21_quick"]["samples"]) == len(sampled) == 16 * fanout
    assert len(Counter(sampled)) == 16
    assert set(Counter(sampled).values()) == {fanout}
    chosen = list(sampled)
    sampled.clear()
    await run(config("tb21_quick", 16))
    assert sorted(sampled) == sorted(chosen)
    sampled.clear()
    full = await run(config("tb21_quick", None))
    episodes = full["tb21_quick"]["samples"]
    assert len(episodes) == len(sampled) == task_count * fanout
    assert Counter(sampled) == Counter({task: fanout for task in range(task_count)})
    assert [sample.index for sample in episodes] == list(range(task_count * fanout))
    assert len({id(sample) for sample in episodes}) == task_count * fanout
    assert len(full["tb21_quick"]["rewards"]) == len(full["tb21_quick"]["truncated"]) == task_count * fanout
    assert len(created) == 1  # Selection is per-call; the full cached dataset is intact.
    assert all(sample.index is None and sample.reward is None for sample in raw)
    assert [sample.metadata for sample in raw] == [{"task_id": i} for i in range(task_count)]
