import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.rollout.base_types import RolloutFnEvalInput
from miles.rollout.inference_rollout import inference_rollout_eval as evaluation
from miles.rollout.inference_rollout.eval_progress import InitialEvalProgress
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.types import Sample


def _fixture(monkeypatch, tmp_path):
    args = SimpleNamespace(
        group_rm=False,
        hf_checkpoint=str(tmp_path / "model"),
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
        rollout_seed=42,
    )
    raw = [Sample(prompt=f"task-{i}") for i in range(3)]
    monkeypatch.setattr(evaluation, "Dataset", lambda **kw: SimpleNamespace(samples=raw))
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *a, **kw: None)
    monkeypatch.setattr(evaluation, "load_processor", lambda *a, **kw: None)
    cfg = EvalDatasetConfig(
        "test", "/tasks.jsonl", n_samples_per_eval_prompt=1, temperature=1.0, top_p=1.0, top_k=-1, max_response_len=100
    )
    return SimpleNamespace(args=args), cfg, raw


async def test_failed_trial_does_not_discard_completed_work_and_restart_only_runs_missing(monkeypatch, tmp_path):
    state, cfg, raw = _fixture(monkeypatch, tmp_path)
    calls = []

    async def generate(state, sample, **kw):
        calls.append(sample.index)
        if sample.index == 1 and calls.count(1) == 1:
            raise RuntimeError("sandbox unavailable")
        await asyncio.sleep(0.01)
        sample.reward = sample.index / 2
        sample.status = Sample.Status.COMPLETED
        return [sample]

    monkeypatch.setattr(evaluation, "generate_and_rm", generate)
    kwargs = dict(progress_dir=str(tmp_path / "progress"))
    with pytest.raises(ExceptionGroup, match="1 evaluation trial"):
        await evaluation.eval_rollout_single_dataset(state, cfg, {}, **kwargs)
    assert sorted(calls) == [0, 1, 2]
    assert len(list((tmp_path / "progress").rglob("*.pt"))) == 2
    # New dataset cache and KV namespace emulate a fresh executor process.
    output = await evaluation.eval_rollout_single_dataset(state, cfg, {}, kv_cache_namespace="fresh", **kwargs)
    assert sorted(calls) == [0, 1, 1, 2]
    assert output["test"]["rewards"] == [0.0, 0.5, 1.0]
    assert all(s.kv_cache_namespace == "fresh" for s in output["test"]["samples"])
    assert all(s.reward is None for s in raw)


async def test_cache_invalidates_sampling_prompt_and_model_changes_and_recovers_corrupt_files(monkeypatch, tmp_path):
    state, cfg, raw = _fixture(monkeypatch, tmp_path)
    calls = []

    async def generate(state, sample, **kw):
        calls.append(sample.index)
        sample.reward, sample.status = 1.0, Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(evaluation, "generate_and_rm", generate)
    directory = tmp_path / "progress"

    async def run(config=cfg):
        return await evaluation.eval_rollout_single_dataset(state, config, {}, progress_dir=str(directory))

    await run()
    await run()
    assert len(calls) == 3
    next(directory.rglob("*.pt")).write_bytes(b"partial")
    await run()
    assert len(calls) == 4
    raw[0].prompt = "changed task"
    await run()
    assert len(calls) == 5
    await run(replace(cfg, temperature=0.7))
    assert len(calls) == 8
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"new weights")
    await run()
    assert len(calls) == 11


async def test_eval_cancellation_waits_for_trial_cleanup(monkeypatch, tmp_path):
    state, cfg, _ = _fixture(monkeypatch, tmp_path)
    started, cleaned = set(), set()
    ready = asyncio.Event()

    async def generate(state, sample, **kw):
        started.add(sample.index)
        if len(started) == 3:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.add(sample.index)

    monkeypatch.setattr(evaluation, "generate_and_rm", generate)
    task = asyncio.create_task(evaluation.eval_rollout_single_dataset(state, cfg, {}))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == started == {0, 1, 2}


@pytest.mark.parametrize(
    "initial,skip,rollout_id,expected",
    [(True, False, 0, "cache"), (False, False, 0, None), (0, True, 0, None), (0, False, 1, None)],
)
async def test_only_first_untrained_eval_can_resume(monkeypatch, initial, skip, rollout_id, expected):
    observed = []

    async def run(*a, **kw):
        observed.append(kw["progress_dir"])
        return {}

    monkeypatch.setattr(evaluation, "run_eval_datasets", run)
    fn = object.__new__(InferenceRolloutFn)
    fn.state = SimpleNamespace(
        args=SimpleNamespace(
            start_rollout_id=None,
            skip_eval_before_train=skip,
            eval_progress_dir="cache",
            namespaced_radix_cache=False,
        )
    )
    fn.eval_prompt_dataset_cache = {}
    fn._initial_eval_pending = True
    await fn._call_eval(RolloutFnEvalInput(rollout_id=rollout_id, initial_weights=initial))
    await fn._call_eval(RolloutFnEvalInput(rollout_id=rollout_id, initial_weights=initial))
    assert observed == [expected, None]


async def test_shared_eval_preserves_initial_checkpoint_identity():
    from unittest.mock import AsyncMock

    manager = SimpleNamespace(eval=AsyncMock())
    dispatcher = EvalDispatcher(SimpleNamespace(eval_uses_snapshots=False), None, manager)
    await dispatcher.dispatch(0, hf_dir="/model")
    manager.eval.assert_awaited_once_with(0, hf_dir="/model")


def test_task_identity_tracks_verifier_changes_but_ignores_judge_transport(monkeypatch, tmp_path):
    state, cfg, _ = _fixture(monkeypatch, tmp_path)
    task = tmp_path / "task"
    task.mkdir()
    config = task / "task.toml"
    verifier = task / "test.sh"
    verifier.write_text("original verifier")
    sample = Sample(index=0, metadata={"task_path": str(task)})

    def path(judge):
        config.write_text(f'[verifier.env]\nJUDGE_URL="{judge}"\nFRONTIERCS_JUDGE_TOKEN="changing token"\n')
        return InitialEvalProgress(tmp_path / "progress", state.args, cfg, {}).path(sample, {})

    original = path("https://old-judge")
    assert path("https://new-judge") == original
    verifier.write_text("new verifier")
    assert path("https://new-judge") != original


@pytest.mark.parametrize(
    "status,reward", [(Sample.Status.ABORTED, None), (Sample.Status.FAILED, 0.0), (Sample.Status.COMPLETED, None)]
)
def test_infrastructure_failures_are_never_cached_as_scores(monkeypatch, tmp_path, status, reward):
    state, cfg, _ = _fixture(monkeypatch, tmp_path)
    progress = InitialEvalProgress(tmp_path / "progress", state.args, cfg, {})
    sample = Sample(index=0, status=status, reward=reward)
    path = progress.path(sample, {})
    with pytest.raises(ValueError, match="completed, scored"):
        progress.save(path, sample)
    assert not path.exists()
