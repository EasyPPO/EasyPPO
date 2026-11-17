"""CPU integration contracts for submit-only FCS episodes and Harbor grading."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
pytest.importorskip("minisweagent")

from harbor.models.agent.context import AgentContext

from agentic_critic.fcs_env import COMPACT_PROMPT, TASK_PROMPT
from agentic_critic.fcs_judge import JudgeResult
from agentic_critic.fcs_terminal import FcsSubmissionAgent
from agentic_critic.harbor_fcs import HarborFcsAgent
from agentic_critic.harbor_terminal import run_trial
from agentic_critic.terminal_env import TerminalConfig
from scripts import frontiersmith_verifier as verifier
from tests.agentic_critic.test_terminal_data import make_task
from tests.agentic_critic.test_terminal_env import Tokenizer, completion
from tests.agentic_critic.test_terminal_thinking import tokenizer  # noqa: F401


def episode(replies, scores, **config):
    replies, scores = iter(replies), iter(scores)
    prompts, codes = [], []

    async def decode(prompt, limit):
        prompts.append(Tokenizer().decode(prompt))
        return completion(next(replies))

    async def submit(code):
        codes.append(code)
        return JudgeResult(next(scores), "done")

    agent = FcsSubmissionAgent(
        tokenizer=Tokenizer(),
        decode=decode,
        submit=submit,
        config=TerminalConfig(**config),
        max_submissions=2,
        max_turns=3,
    )
    return agent, prompts, codes


@pytest.mark.asyncio
async def test_only_cpp_submits_and_feedback_keeps_the_best_score():
    agent, prompts, codes = episode(
        [
            "```bash\nrm -rf /app\n```",
            "```cpp\nignored\n```</think>```cpp\nfirst\n```",
            "```cpp\nsecond\n```",
        ],
        [80, 20],
    )
    trace = await agent.run_async("statement")
    assert codes == ["first", "second"]
    assert prompts[0] == "<user>" + TASK_PROMPT.format(statement="statement") + "</user><assistant>"
    assert "No ```cpp code block" in prompts[1]
    assert "score=80.00/100" in prompts[2] and "Best score so far: 80.00/100" in prompts[2]
    assert trace.submission_scores == [80, 20]
    assert trace.n_format_errors == 1 and trace.n_commands == 0
    assert trace.exit_status == "Submitted"
    assert sum(sum(s.loss_mask) for s in trace.segments) == trace.generated_tokens


@pytest.mark.asyncio
async def test_summary_uses_reference_prompts_and_keeps_trainable_tokens():
    agent, prompts, codes = episode(
        ["```cpp\n" + "x" * 1100 + "\n```", "hidden</think>keep this plan", "```cpp\nsecond\n```"],
        [80, 20],
        max_context_tokens=6000,
        compact_threshold=1200,
        max_new_tokens=1500,
        summary_tokens=200,
    )
    trace = await agent.run_async("statement")
    assert len(codes) == 2 and trace.n_compactions == 1
    assert COMPACT_PROMPT in prompts[1]
    assert "keep this plan" in prompts[2] and "hidden" not in prompts[2]
    assert "x" * 1100 not in prompts[2] and "Best score so far" not in prompts[2]
    assert [call["kind"] for call in trace.calls] == ["action", "summary", "action"]
    assert sum(sum(s.loss_mask) for s in trace.segments) == trace.generated_tokens
    fields = trace.sample_fields()
    assert len(fields) == 2 and all(f["critic_memory_exact"] for f in fields)


@pytest.mark.asyncio
async def test_turn_limit_does_not_submit_invalid_answers():
    agent, _, codes = episode(["no code"] * 3, [])
    trace = await agent.run_async("statement")
    assert not codes and not trace.truncated and trace.n_format_errors == 3


@pytest.mark.asyncio
async def test_harbor_uploads_long_source_and_verifier_uses_best_submission(monkeypatch, tmp_path):
    container = tmp_path / "container"
    container.mkdir()
    commands, sources = [], []
    responses = iter([{"status": "error"}, {"status": "done", "score": 80, "scoreUnbounded": 150}])

    def judge(url, pid, code, *, timeout):
        sources.append(code)
        return next(responses)

    monkeypatch.setattr(verifier, "_judge", judge)

    async def upload_file(source_path, target_path):
        (container / Path(target_path).name).write_bytes(Path(source_path).read_bytes())

    async def execute(**kwargs):
        commands.append(kwargs)
        response = verifier._submit(
            solution=container / "solution.cpp",
            ledger=container / "fcs-submissions.json",
            judge_url="http://judge",
            problem_id="0",
            timeout=30,
        )
        return SimpleNamespace(return_code=0, stdout=json.dumps(response), stderr="")

    replies = iter(["```cpp\n" + "x" * 70000 + "\n```", "```cpp\nsecond\n```"])

    async def decode(prompt, limit):
        return completion(next(replies))

    agent = HarborFcsAgent(logs_dir=tmp_path / "agent")
    agent.bind(
        tokenizer=Tokenizer(),
        decode=decode,
        config=TerminalConfig(
            max_context_tokens=200000,
            max_total_tokens=150000,
            max_new_tokens=80000,
            compact_threshold=190000,
            compaction="none",
        ),
    )
    agent.bind_judge(
        verifier_env={
            "JUDGE_URL": "http://judge",
            "PROBLEM_ID": "0",
            "MAX_POLL_TIME": "30",
            "FCS_ITERATIVE_SUBMISSIONS": "1",
        },
        max_submissions=2,
        max_turns=16,
    )
    environment = SimpleNamespace(upload_file=upload_file, exec=execute)
    await agent.setup(environment)
    await agent.run("statement", environment, AgentContext())
    verifier._verify_submissions(ledger=container / "fcs-submissions.json", logs=tmp_path / "verifier")
    assert json.loads((tmp_path / "verifier/reward.json").read_text()) == {"reward": 0.8}
    assert len(sources) == 2 and len(sources[0]) == 70000
    assert all(c["command"] == "python3 /app/easyppo-fcs-submit.py --submit" for c in commands)
    assert agent.trace.submission_scores == [0, 80]


@pytest.mark.parametrize(
    "result",
    [
        {"status": "queued"},
        {"status": "done", "score": float("nan")},
    ],
)
def test_failed_submission_never_enters_reward_ledger(monkeypatch, tmp_path, result):
    source, ledger = tmp_path / "solution.cpp", tmp_path / "ledger.json"
    source.write_text("code")
    ledger.write_text("[]")
    monkeypatch.setattr(verifier, "_judge", lambda *a, **kw: result)
    with pytest.raises((ValueError, RuntimeError)):
        verifier._submit(solution=source, ledger=ledger, judge_url="http://judge", problem_id="0", timeout=1)
    assert ledger.read_text() == "[]"


def test_empty_submission_and_no_submission_are_valid_zeros(tmp_path):
    source, ledger = tmp_path / "solution.cpp", tmp_path / "ledger.json"
    source.write_text("")
    ledger.write_text("[]")
    for submit in (False, True):
        if submit:
            verifier._submit(solution=source, ledger=ledger, judge_url="unused", problem_id="0", timeout=1)
        verifier._verify_submissions(ledger=ledger, logs=tmp_path)
        assert (tmp_path / "reward.txt").read_text() == "0.0"
    ledger.unlink()
    with pytest.raises(FileNotFoundError):
        verifier._verify_submissions(ledger=ledger, logs=tmp_path)
    assert not (tmp_path / "reward.txt").exists()


@pytest.mark.asyncio
async def test_cancellation_during_submission_propagates():
    agent, _, _ = episode(["```cpp\ncode\n```"], [])

    async def cancel(code):
        raise asyncio.CancelledError

    agent.submit = cancel
    with pytest.raises(asyncio.CancelledError):
        await agent.run_async("statement")


@pytest.mark.asyncio
async def test_native_trial_selects_submit_agent_and_resolves_judge_configuration(monkeypatch, tmp_path):
    from harbor.trial.trial import Trial

    task = make_task(tmp_path / "tasks")
    with (task / "task.toml").open("a") as config:
        config.write(
            '\n[verifier.env]\nFCS_ITERATIVE_SUBMISSIONS="1"\n'
            'FRONTIERCS_JUDGE_TOKEN="${FRONTIERCS_JUDGE_TOKEN:-}"\n'
        )
    monkeypatch.setenv("FRONTIERCS_JUDGE_TOKEN", "test-token")

    async def run(self):
        try:
            assert isinstance(self.agent, HarborFcsAgent)
            assert (self.agent.max_submissions, self.agent.max_turns) == (3, 7)
            assert self.agent.judge_env["FRONTIERCS_JUDGE_TOKEN"] == "test-token"
            return SimpleNamespace(exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": 0.5}))
        finally:
            self._close_logger_handler()

    monkeypatch.setattr(Trial, "run", run)
    _, rewards, _ = await run_trial(
        task_path=task,
        trials_dir=tmp_path / "trials",
        environment="docker",
        tokenizer=Tokenizer(),
        decode=None,
        config=TerminalConfig(),
        agent_protocol="fcs-submit",
        fcs_max_submissions=3,
        fcs_max_turns=7,
    )
    assert rewards == {"reward": 0.5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "truncated_call,budget_stop", [(None, False), (0, False), (1, False), (2, False), (2, True), (None, True)]
)
async def test_fcs_samples_match_reference_with_real_template(
    monkeypatch, tmp_path, request, truncated_call, budget_stop
):
    tokenizer_impl = request.getfixturevalue("tokenizer")

    from agentic_critic import harbor_terminal, miles_terminal
    from agentic_critic.fcs_env import ContextFull, TurnFormat, run_episode
    from agentic_critic.miles_fcs import episode_samples
    from miles.utils import http_utils
    from miles.utils.types import Sample
    from tests.agentic_critic.test_miles_terminal import args_for

    texts = [
        "plan</think>\n\n```cpp\n// " + "x " * 800 + "\nint main(){}\n```",
        "summarize</think>\n\nKeep the first algorithm.",
        "improve</think>\n\n```cpp\nint main(){return 0;}\n```",
    ]
    token_lists = [
        tokenizer_impl.encode(
            text + ("" if i == truncated_call else tokenizer_impl.eos_token), add_special_tokens=False
        )
        for i, text in enumerate(texts)
    ]
    reference_prompts = []
    reference_calls = iter(enumerate(token_lists))
    reference_scores = iter([80, 20])

    def generate(prompt):
        if budget_stop and len(reference_prompts) == len(token_lists):
            raise ContextFull
        reference_prompts.append(prompt)
        i, tokens = next(reference_calls)
        return tokens, i != truncated_call

    reference = run_episode(
        statement="statement",
        fmt=TurnFormat(tokenizer_impl),
        generate=generate,
        judge=lambda code: JudgeResult(next(reference_scores), "done"),
        max_submissions=3 if budget_stop else 2,
        compact_threshold_tokens=1200,
    )
    expected = episode_samples(reference, [[-0.5] * len(tokens) for tokens in token_lists])
    requests = []
    calls = iter(enumerate(token_lists))

    async def post(url, payload, **kwargs):
        requests.append(payload["input_ids"])
        i, tokens = next(calls)
        return {
            "text": texts[i],
            "meta_info": {
                "output_token_logprobs": [(-0.5, token, None) for token in tokens],
                "finish_reason": {"type": "length" if i == truncated_call else "stop"},
            },
        }

    async def trial(**kwargs):
        scores = iter([80, 20])

        async def submit(code):
            return JudgeResult(next(scores), "done")

        agent = FcsSubmissionAgent(
            tokenizer=kwargs["tokenizer"],
            decode=kwargs["decode"],
            submit=submit,
            config=kwargs["config"],
            max_submissions=kwargs["fcs_max_submissions"],
            max_turns=16,
        )
        return await agent.run_async("statement"), {"reward": 0.8}, "/trial"

    monkeypatch.setattr(http_utils, "post", post)
    monkeypatch.setattr(harbor_terminal, "run_trial", trial)
    make_task(tmp_path)
    args = args_for(tmp_path)
    args.terminal_max_new_tokens = 4096
    args.terminal_summary_tokens = 1024
    args.terminal_max_context_tokens = args.rollout_max_context_len = 16000
    args.terminal_continue_on_length = True
    args.fcs_max_submissions = 3 if budget_stop else 2
    if budget_stop:
        args.terminal_max_total_tokens = sum(map(len, token_lists))
    result = await miles_terminal.generate(
        SimpleNamespace(
            args=args,
            sample=Sample(index=1, metadata={"task_path": "task", "agent_protocol": "fcs-submit"}),
            state=SimpleNamespace(tokenizer=tokenizer_impl),
            evaluation=False,
            sampling_params={"max_new_tokens": 4096, "temperature": 1.0},
        )
    )
    assert requests == reference_prompts
    assert len(result.samples) == len(expected) == 2
    for actual, wanted in zip(result.samples, expected, strict=True):
        if budget_stop and wanted is expected[-1]:
            # Harbor records feedback when the next decode consumes it. The reference
            # eagerly appends the final feedback even though the budget prevents that decode.
            trailing = len(wanted["tokens"]) - len(actual.tokens)
            assert trailing > 0
            assert not any(wanted["loss_mask"][-trailing:])
            wanted = dict(wanted, response_length=wanted["response_length"] - trailing)
            for field in ("tokens", "loss_mask", "rollout_log_probs"):
                wanted[field] = wanted[field][:-trailing]
        for field in ("tokens", "response_length", "loss_mask", "rollout_log_probs"):
            assert getattr(actual, field) == wanted[field]
        assert (actual.status == Sample.Status.TRUNCATED) == wanted["truncated"]
        assert actual.reward == reference.reward
        if budget_stop:
            assert actual.metadata["exit_status"] == "ContextOrTokenLimit"
    summary_start = len(reference_prompts[1])
    summary_end = summary_start + len(token_lists[1])
    assert result.samples[0].metadata["summary_span"] == (summary_start, summary_end)
    assert result.samples[0].tokens[summary_start:summary_end] == token_lists[1]
    assert result.samples[1].metadata["summary_span"] is None


@pytest.mark.asyncio
async def test_exact_soft_threshold_does_not_request_an_extra_summary():
    replies = ["```cpp\nfirst\n```", "```cpp\nsecond\n```"]
    baseline, prompts, _ = episode(replies, [80, 20], compaction="none")
    await baseline.run_async("statement")
    agent, actual, _ = episode(replies, [80, 20], compact_threshold=len(prompts[1]))
    trace = await agent.run_async("statement")
    assert actual == prompts and trace.n_compactions == 0
    assert agent._should_compact(len(prompts[1]) + 1, 1)
    assert agent._should_compact(agent.options.max_context_tokens - 100, 100)
