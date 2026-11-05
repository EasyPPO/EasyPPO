"""Collector aborts can precede the terminal hook and its registry lookup."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("minisweagent")
pytest.importorskip("harbor")

from agentic_critic import harbor_terminal, miles_terminal
from tests.agentic_critic.test_terminal_topk import _args, _task, _Tokenizer

from miles.utils import http_utils
from miles.utils.types import Sample


def _input(tmp_path, *, evaluation=False, candidate_count=0):
    _task(tmp_path)
    return SimpleNamespace(
        args=_args(tmp_path, candidate_count=candidate_count),
        sample=Sample(index=1, reward=0.75, metadata={"task_path": "task", "original": True}),
        state=SimpleNamespace(tokenizer=_Tokenizer(), aborted=False),
        evaluation=evaluation,
        sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 32},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_collector_abort_converts_late_trial_failure_before_hook(monkeypatch, tmp_path, failure_type):
    input = _input(tmp_path)
    started, fail = asyncio.Event(), asyncio.Event()

    async def trial(**kwargs):
        started.set()
        await fail.wait()
        raise failure_type("Engine stopped during collector abort")

    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    pending = asyncio.create_task(miles_terminal.generate(input))
    await asyncio.wait_for(started.wait(), timeout=10)
    input.state.aborted = True
    fail.set()
    output = await asyncio.wait_for(pending, timeout=10)
    assert not miles_terminal._ACTIVE_TRIALS
    # The generic collector can invoke the plugin hook only after cleanup and
    # registry removal. It must not be required to classify this discarded tail.
    await miles_terminal.abort(input.args)
    sample = output.samples
    assert sample.status == Sample.Status.ABORTED and sample.reward is None
    assert sample.metadata["terminal_abort_failure_type"] == failure_type.__name__
    assert sample.metadata["terminal_abort_reason"] == "Engine stopped during collector abort"
    assert sample.metadata["original"] is True
    assert input.sample.reward == 0.75 and input.sample.status != Sample.Status.ABORTED


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation,aborted", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError])
async def test_errors_without_training_collector_abort_propagate(
    monkeypatch, tmp_path, evaluation, aborted, failure_type
):
    input = _input(tmp_path, evaluation=evaluation)
    input.state.aborted = aborted

    async def trial(**kwargs):
        raise failure_type("Real infrastructure failure")

    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    with pytest.raises(failure_type, match="Real infrastructure failure"):
        await miles_terminal.generate(input)
    assert not miles_terminal._ACTIVE_TRIALS


@pytest.mark.asyncio
async def test_engine_abort_skips_invalid_topk_reply_and_waits_for_cleanup(monkeypatch, tmp_path):
    input = _input(tmp_path, candidate_count=3)
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
    requests, cleaned = [], []

    async def post(url, payload, **kwargs):
        requests.append(payload)
        input.state.aborted = True
        return {
            "text": "",
            "meta_info": {"output_top_logprobs": "invalid discarded output", "finish_reason": {"type": "abort"}},
        }

    async def trial(**kwargs):
        try:
            await kwargs["decode"]([1], 2)
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleaned.append(True)

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    pending = asyncio.create_task(miles_terminal.generate(input))
    await asyncio.wait_for(cleanup_started.wait(), timeout=10)
    assert not pending.done() and not cleaned
    release_cleanup.set()
    sample = (await asyncio.wait_for(pending, timeout=10)).samples
    assert len(requests) == 1 and cleaned == [True]
    assert sample.status == Sample.Status.ABORTED and sample.reward is None
    assert sample.metadata["terminal_abort_failure_type"] == "CancelledError"
    assert "during decode" in sample.metadata["terminal_abort_reason"]
    assert not miles_terminal._ACTIVE_TRIALS


@pytest.mark.asyncio
async def test_collector_abort_prevents_next_decode_request(monkeypatch, tmp_path):
    input = _input(tmp_path, candidate_count=3)

    async def post(*args, **kwargs):
        pytest.fail("A discarded terminal trial must not start another decode")

    async def trial(**kwargs):
        input.state.aborted = True
        await kwargs["decode"]([1], 2)

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    sample = (await miles_terminal.generate(input)).samples
    assert sample.status == Sample.Status.ABORTED and sample.reward is None
    assert "before decode" in sample.metadata["terminal_abort_reason"]
    assert not miles_terminal._ACTIVE_TRIALS


@pytest.mark.asyncio
async def test_evaluation_decode_ignores_training_collector_abort(monkeypatch, tmp_path):
    input = _input(tmp_path, evaluation=True, candidate_count=3)
    input.state.aborted = True
    requests = []

    async def post(url, payload, **kwargs):
        requests.append(payload)
        return {
            "text": "A",
            "meta_info": {"output_token_logprobs": [(-0.5, 65, None)], "finish_reason": {"type": "stop"}},
        }

    async def trial(**kwargs):
        completion = await kwargs["decode"]([1], 2)
        assert completion.tokens == [65]
        raise RuntimeError("Evaluation verifier failed")

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    with pytest.raises(RuntimeError, match="Evaluation verifier failed"):
        await miles_terminal.generate(input)
    assert len(requests) == 1 and not miles_terminal._ACTIVE_TRIALS


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_outcome", ["propagate", "error", "return"])
async def test_external_cancellation_wins_over_collector_abort(monkeypatch, tmp_path, cancel_outcome):
    input = _input(tmp_path)
    started, cleaned = asyncio.Event(), []

    async def trial(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if cancel_outcome == "error":
                raise RuntimeError("Trial teardown failed") from None
            if cancel_outcome == "return":
                return None
            raise
        finally:
            await asyncio.sleep(0)
            cleaned.append(True)

    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    pending = asyncio.create_task(miles_terminal.generate(input))
    await asyncio.wait_for(started.wait(), timeout=10)
    input.state.aborted = True
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cleaned == [True] and not miles_terminal._ACTIVE_TRIALS
