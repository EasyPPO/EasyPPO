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


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("failure_type", ["ImageBuildError", "EnvironmentStartTimeoutError", "AgentSetupTimeoutError", "RuntimeError"])
async def test_modal_failure_at_startup(monkeypatch, tmp_path, evaluation, failure_type):
    inp = _input(tmp_path, evaluation=evaluation)
    inp.args.terminal_environment = "modal"
    in_setup = failure_type in ("RuntimeError", "AgentSetupTimeoutError")
    message = "Failed to start tmux session" if failure_type == "RuntimeError" else f"Modal startup failed with {failure_type}"
    trial_dir = tmp_path / "trial"

    attempts = []
    monkeypatch.setattr(miles_terminal.random, "uniform", lambda *_: 0)

    async def create(config):
        attempts.append(config.trial_name)
        async def run():
            return SimpleNamespace(
                environment_setup=object(), agent_setup=object() if in_setup else None, agent_execution=None,
                exception_info=SimpleNamespace(exception_type=failure_type, exception_message=message),
            )

        return SimpleNamespace(
            agent=SimpleNamespace(bind=lambda **kwargs: None, trace=None), run=run,
            config=config, paths=SimpleNamespace(trial_dir=trial_dir),
        )

    monkeypatch.setattr(harbor_terminal.Trial, "create", create)
    from miles.rollout.filter_hub.common_filters import apply_aborted_filter

    sample = (await miles_terminal.generate(inp)).samples
    assert sample.status == Sample.Status.ABORTED
    assert sample.reward == (0.0 if evaluation else None)
    assert sample.metadata["terminal_abort_failure_type"] == failure_type
    assert str(trial_dir) in sample.metadata["terminal_abort_reason"]
    assert len(attempts) == (2 if failure_type == "EnvironmentStartTimeoutError" else 1)
    assert not apply_aborted_filter(inp.args, [inp.sample, sample]).keep


@pytest.mark.asyncio
async def test_transient_start_recovers_without_replaying_group(monkeypatch):
    attempts = []
    async def run(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise harbor_terminal.SandboxInfrastructureFailure("ServiceError", "unavailable", retry_start=True)
        return "finished"

    monkeypatch.setattr(miles_terminal.random, "uniform", lambda *_: 0)
    monkeypatch.setattr(harbor_terminal, "run_trial", run)
    result = await miles_terminal._run_cancellable_trial(args=object(), evaluation=False, task_path="task")
    assert result == "finished" and attempts == [{"task_path": "task"}] * 2


@pytest.mark.parametrize("kind,trace,start,expected", [
    ("ServiceError", "modal.exception.ServiceError: unavailable", True, True),
    ("ServiceError", "modal.exception.ServiceError: unavailable", False, False),
    ("ConnectionError", harbor_terminal._MODEL_ERROR_NOTE, False, None),
    ("AuthError", "modal.exception.AuthError: bad token", True, None),
    ("SandboxLikelyOutOfMemoryError", "", False, False),
    ("DownloadVerifierDirError", "", False, False),
    ("VerifierTimeoutError", "", False, False),
    ("PreviouslyUnknownFailure", "", False, False),
])
def test_trial_infrastructure_error_boundaries(tmp_path, kind, trace, start, expected):
    result = SimpleNamespace(environment_setup=object(), agent_setup=None if start else object(),
        agent_execution=None if start else object(), exception_info=SimpleNamespace(
            exception_type=kind, exception_message="message", exception_traceback=trace))
    error = harbor_terminal._modal_trial_failure(result.exception_info, tmp_path, starting=start)
    assert (error.retry_start if error else None) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("kind, frame", [
    ("NeverBeforeSeenError", "new_remote_method"),
    ("RuntimeError", "_sdk_exec"),
    ("RuntimeError", "_sdk_upload_dir"),
    ("RuntimeError", "_sdk_download_dir"),
    ("TaskIntegrityCheckError", "protected_digests"),
])
async def test_task_environment_failure_isolated_after_start(monkeypatch, tmp_path, evaluation, kind, frame):
    inp = _input(tmp_path, evaluation=evaluation)
    inp.args.terminal_environment = "modal"
    attempts = []
    async def create(config):
        attempts.append(config)
        async def run():
            return SimpleNamespace(environment_setup=object(), agent_setup=object(), agent_execution=object(),
                exception_info=SimpleNamespace(exception_type=kind, exception_message="task environment failed",
                    exception_traceback=f'  File "/lib/harbor/environments/modal.py", line 1, in {frame}\n'))
        return SimpleNamespace(agent=SimpleNamespace(bind=lambda **kw: None, trace=None),
            add_hook=lambda *args: None, run=run, paths=SimpleNamespace(trial_dir=tmp_path))
    monkeypatch.setattr(harbor_terminal.Trial, "create", create)
    sample = (await miles_terminal.generate(inp)).samples
    assert len(attempts) == 1
    assert sample.status == Sample.Status.ABORTED
    assert sample.reward == (0.0 if evaluation else None)
    assert sample.metadata["terminal_abort_failure_type"] == kind
    from miles.rollout.filter_hub.common_filters import apply_aborted_filter
    assert not apply_aborted_filter(inp.args, [inp.sample, sample]).keep


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "run", "model"])
async def test_new_modal_error_boundary(monkeypatch, tmp_path, phase):
    class PreviouslyUnseenError(Exception):
        pass

    inp = _input(tmp_path)
    inp.args.terminal_environment = "modal"

    async def fail(*args, **kwargs):
        raise PreviouslyUnseenError("new error")

    async def create(config):
        if phase == "create":
            return await fail()
        bound = {}
        async def run():
            if phase == "run":
                return await fail()
            try:
                await bound["decode"]([1], 2)
            except Exception as exc:
                return SimpleNamespace(environment_setup=object(), agent_setup=object(), agent_execution=object(),
                                       exception_info=harbor_terminal.ExceptionInfo.from_exception(exc))
        return SimpleNamespace(config=config, agent=SimpleNamespace(bind=lambda **kw: bound.update(kw), trace=None),
                               add_hook=lambda *a: None, run=run, paths=SimpleNamespace(trial_dir=tmp_path))

    monkeypatch.setattr(harbor_terminal.Trial, "create", create)
    monkeypatch.setattr(http_utils, "post", fail)
    if phase == "model":
        with pytest.raises(RuntimeError, match="PreviouslyUnseenError"):
            await miles_terminal.generate(inp)
    else:
        sample = (await miles_terminal.generate(inp)).samples
        assert sample.status == Sample.Status.ABORTED and sample.reward is None
        assert sample.metadata["terminal_infra_failure_type"] == "PreviouslyUnseenError"


@pytest.mark.asyncio
@pytest.mark.parametrize("external", [False, True])
async def test_dispatched_train_cancel_awaits_teardown_and_preserves_eval(tmp_path, external):
    inp = _input(tmp_path)
    events = [asyncio.Event() for _ in range(4)]
    started, cleaning, cleaned, eval_done = events

    class Train:
        async def run(self, *a):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await cleaned.wait()

    class Eval:
        async def run(self, *a):
            await eval_done.wait()
            return "eval-result"

    inp.args.terminal_train_workers_per_node = inp.args.terminal_eval_workers_per_node = 1
    inp.state.terminal_train_workers, inp.state.terminal_eval_workers = Train(), Eval()
    ev = SimpleNamespace(**(vars(inp) | {"evaluation": True}))
    train = asyncio.create_task(miles_terminal.generate(inp))
    evaluation = asyncio.create_task(miles_terminal.generate(ev))
    await started.wait()
    if external:
        train.cancel()
    else:
        abort = asyncio.create_task(miles_terminal.abort(inp.args))
    await cleaning.wait()
    assert not train.done() and not evaluation.done()
    cleaned.set()
    if external:
        with pytest.raises(asyncio.CancelledError):
            await train
    else:
        await abort
        sample = (await train).samples
        assert sample.status is Sample.Status.ABORTED and sample.reward is None
    eval_done.set()
    assert await evaluation == "eval-result"
    assert not miles_terminal._ACTIVE_TRIALS
