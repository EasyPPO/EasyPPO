"""CPU contract tests for the Frontier judge adapter and bundled dataset IDs."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("frontiercs_reward", ROOT / "verl/utils/reward_score/frontiercs.py")
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


@pytest.mark.parametrize("pid", ["frontiersmith_0", "frontiersmith_199", "frontiercs_val_1"])
def test_submission_and_partial_score(monkeypatch, pid):
    post = Mock(return_value=response({"sid": "test"}))
    get = Mock(
        side_effect=[response({}, 404), response({"status": "running"}), response({"status": "done", "score": 42.5})]
    )
    monkeypatch.setattr(reward.requests, "post", post)
    monkeypatch.setattr(reward.requests, "get", get)
    monkeypatch.setattr(reward.time, "sleep", lambda _: None)
    result = reward.compute_score(
        "frontiercs", "<think>ignored</think>```cpp\nint main() {}\n```", pid, judge_url="http://judge/"
    )
    assert result == {"score": 0.425, "acc": 42.5, "raw_score": 42.5, "judge_raw_score": 42.5}
    assert post.call_args.kwargs["data"] == {"pid": pid, "lang": "cpp"}
    assert post.call_args.kwargs["files"] == {"code": ("sol.cpp", "int main() {}")}
    assert get.call_args.args == ("http://judge/result/test",)


@pytest.mark.parametrize("failure", ["empty", "missing_sid", "error", "connection", "timeout"])
def test_failures_score_zero(monkeypatch, failure):
    post = Mock(return_value=response({} if failure == "missing_sid" else {"sid": "test"}))
    if failure == "connection":
        post.side_effect = requests.ConnectionError("offline")
    monkeypatch.setattr(reward.requests, "post", post)
    monkeypatch.setattr(reward.requests, "get", Mock(return_value=response({"status": "error"})))
    if failure == "timeout":
        monkeypatch.setattr(reward, "MAX_WAIT", 0)
    result = reward.compute_score("frontiercs", "" if failure == "empty" else "int main() {}", "frontiersmith_0")
    assert result == {"score": 0.0, "acc": 0.0, "raw_score": 0.0, "judge_raw_score": 0.0}
    if failure == "empty":
        post.assert_not_called()


def test_wrong_data_source():
    with pytest.raises(ValueError, match="data_source"):
        reward.compute_score("aime", "code", "frontiersmith_0")


@pytest.mark.parametrize(
    "explicit,extra,expected",
    [
        ("http://explicit", "http://extra", "http://explicit"),
        (None, "http://extra", "http://extra"),
        (None, None, "http://environment"),
    ],
)
def test_judge_url_precedence(monkeypatch, explicit, extra, expected):
    post = Mock(return_value=response({}))
    monkeypatch.setattr(reward.requests, "post", post)
    monkeypatch.setenv("FRONTIERCS_JUDGE_URL", "http://environment")
    reward.compute_score("frontiercs", "code", "frontiersmith_0", extra_info={"judge_url": extra}, judge_url=explicit)
    assert post.call_args.args == (expected + "/submit",)


def test_bundled_splits_have_disjoint_judge_ids():
    problems = ROOT / "Frontier-CS/algorithmic/problems"
    splits = {}
    for name, count in [("train_synthetic_200", 200), ("frontiercs_val", 172)]:
        data = pd.read_parquet(ROOT / "data/frontiercs" / f"{name}.parquet")
        assert len(data) == count
        assert set(data.data_source) == {"frontiercs"}
        ids = {str(row["ground_truth"]) for row in data.reward_model}
        assert len(ids) == count
        for pid in ids:
            assert (problems / pid / "config.yaml").is_file()
            assert (problems / pid / "statement.txt").is_file()
            assert (problems / pid / "testdata").is_dir()
        splits[name] = ids
    assert splits["train_synthetic_200"] == {f"frontiersmith_{i}" for i in range(200)}
    assert not splits["train_synthetic_200"] & splits["frontiercs_val"]
    assert {p.name for p in problems.iterdir()} == splits["train_synthetic_200"] | splits["frontiercs_val"]


def test_smith_release_has_only_published_assets_and_matching_prompts():
    problems = ROOT / "Frontier-CS/algorithmic/problems"
    data = pd.read_parquet(ROOT / "data/frontiercs/train_synthetic_200.parquet")
    pair_count = 0
    for row in data.itertuples():
        problem = problems / row.reward_model["ground_truth"]
        assert {p.name for p in problem.iterdir()} == {"statement.txt", "chk.cc", "config.yaml", "testdata"}
        files = list((problem / "testdata").iterdir())
        assert all(p.is_file() and p.suffix in {".in", ".ans"} for p in files)
        inputs = {p.stem for p in files if p.suffix == ".in"}
        answers = {p.stem for p in files if p.suffix == ".ans"}
        assert inputs and inputs == answers
        pair_count += len(inputs)
        assert len(row.prompt) == 1
        assert row.prompt[0]["role"] == "user"
        assert row.prompt[0]["content"] == (
            "You are a competitive programmer. Solve the following problem in C++. "
            "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.\n\n"
            + (problem / "statement.txt").read_text(encoding="utf-8")
            + "\n\nGenerate solution code:"
        )
    assert pair_count == 1998
