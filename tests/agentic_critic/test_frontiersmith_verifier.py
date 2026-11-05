import json

import pytest

from scripts import frontiersmith_verifier as verifier


@pytest.mark.parametrize("score, expected", [(0, 0.0), (37.5, 0.375), (100, 1.0)])
def test_reward_preserves_upstream_continuous_score(score, expected):
    assert verifier._reward({"status": "done", "score": score}) == expected


@pytest.mark.parametrize("score", [None, "50", True, float("nan"), float("inf"), -1, 101])
def test_invalid_scores_cannot_train(score):
    with pytest.raises(ValueError, match="Invalid"):
        verifier._reward({"status": "done", "score": score})


def test_contestant_compile_failure_scores_zero_but_judge_failure_does_not():
    assert verifier._reward({"status": "error", "error": "Error: compile failed: main.cpp: invalid C++"}) == 0.0
    for message in ("Error: checker compile failed: bad checker", "fetch failed", "Unknown problem"):
        with pytest.raises(RuntimeError, match="judge failure"):
            verifier._reward({"status": "error", "error": message})


def test_verifier_writes_numeric_reward_and_separate_diagnostics(monkeypatch, tmp_path):
    solution = tmp_path / "solution.cpp"
    solution.write_text("int main() {}")
    result = {"status": "done", "score": 42.5, "cases": [{"scoreRatio": 0.425}], "scoreUnbounded": 50}
    monkeypatch.setattr(verifier, "_judge", lambda *args, **kwargs: result)
    verifier._verify(
        solution=solution, logs=tmp_path, judge_url="http://judge/", problem_id="frontiersmith_0", timeout=30
    )
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 0.425}
    assert float((tmp_path / "reward.txt").read_text()) == 0.425
    assert json.loads((tmp_path / "judge-result.json").read_text()) == result


@pytest.mark.parametrize(
    "failure", [OSError("HTTP failed"), TimeoutError("judge timeout"), RuntimeError("bad service")]
)
def test_infrastructure_failures_leave_no_zero_or_stale_reward(monkeypatch, tmp_path, failure):
    solution = tmp_path / "solution.cpp"
    solution.write_text("int main() {}")
    (tmp_path / "reward.txt").write_text("1")
    (tmp_path / "reward.json").write_text('{"reward": 1}')

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(verifier, "_judge", fail)
    with pytest.raises(type(failure)):
        verifier._verify(
            solution=solution, logs=tmp_path, judge_url="http://judge", problem_id="frontiersmith_0", timeout=30
        )
    assert not (tmp_path / "reward.txt").exists() and not (tmp_path / "reward.json").exists()


def test_missing_solution_is_task_failure_without_contacting_judge(monkeypatch, tmp_path):
    monkeypatch.setattr(verifier, "_judge", lambda *args, **kwargs: pytest.fail("should not submit missing solution"))
    verifier._verify(
        solution=tmp_path / "missing.cpp",
        logs=tmp_path,
        judge_url="http://judge",
        problem_id="frontiersmith_0",
        timeout=30,
    )
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 0.0}


def test_judge_uses_upstream_json_contract_and_polls_until_done(monkeypatch):
    calls = []
    results = iter([{"sid": "123"}, {"status": "queued"}, {"status": "done", "score": 55}])

    def request(url, payload=None):
        calls.append((url, payload))
        return next(results)

    monkeypatch.setattr(verifier, "_wait_ready", lambda url: None)
    monkeypatch.setattr(verifier, "_request", request)
    monkeypatch.setattr(verifier.time, "sleep", lambda seconds: None)
    assert verifier._judge("http://judge", "frontiersmith_3", "code", timeout=30)["score"] == 55
    assert calls == [
        ("http://judge/submit", {"pid": "frontiersmith_3", "lang": "cpp", "code": "code"}),
        ("http://judge/result/123", None),
        ("http://judge/result/123", None),
    ]


def test_judge_timeout_is_an_error(monkeypatch):
    monkeypatch.setattr(verifier, "_wait_ready", lambda url: None)
    monkeypatch.setattr(verifier, "_request", lambda url, payload=None: {"sid": "123", "status": "queued"})
    timestamps = iter([0, 31])
    monkeypatch.setattr(verifier.time, "monotonic", lambda: next(timestamps))
    with pytest.raises(TimeoutError, match="timed out"):
        verifier._judge("http://judge", "frontiersmith_3", "code", timeout=30)
