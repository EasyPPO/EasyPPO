"""The Miles wrapper end to end with a fake inference endpoint and a fake judge. Needs Miles importable."""

import asyncio
from types import SimpleNamespace

import pytest

# not "miles" alone: with scripts/ on the import path (other tests put it there) scripts/miles/ answers to that name
pytest.importorskip("miles.utils.types")

from agentic_critic import miles_fcs  # noqa: E402
from tests.agentic_critic.test_fcs_env import EOS, CharTokenizer, FakeCase, FakeResult, code_reply  # noqa: E402


def test_generate_returns_valid_miles_samples(monkeypatch):
    from miles.utils.types import Sample

    replies = iter([code_reply("a"), f"</think>SUMMARY{EOS}", code_reply("b")])
    scores = iter([10.0, 40.0])
    prompts = []

    def fake_post(url, payload, headers, timeout=0):
        prompts.append(payload["input_ids"])
        assert payload["return_logprob"] and payload["sampling_params"]["max_new_tokens"] > 0
        text = next(replies)
        return {"text": text, "meta_info": {"output_token_logprobs": [(-0.25, ord(c), None) for c in text], "finish_reason": {"type": "stop"}}}

    monkeypatch.setattr(miles_fcs, "_post", fake_post)
    monkeypatch.setattr(miles_fcs.fcs_judge_http, "judge", lambda url, pid, code: FakeResult(next(scores), cases=[FakeCase("Accepted")]))
    args = SimpleNamespace(
        partial_rollout=False, sglang_router_ip="127.0.0.1", sglang_router_port=1, sglang_router_policy="round_robin",
        rollout_max_response_len=4096, rollout_max_context_len=None, use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False, rollout_top_logprobs_num=0, fcs_judge_url="http://judge", fcs_max_submissions=2,
        fcs_max_turns=8, fcs_compact_threshold=1,
    )  # fmt: skip
    state = SimpleNamespace(tokenizer=CharTokenizer(), args=args)
    base = Sample(index=7, prompt="S", metadata={"problem_id": "p1"})
    input = SimpleNamespace(args=args, state=state, sample=base, sampling_params={"temperature": 1.0}, evaluation=True)
    samples = asyncio.run(miles_fcs.generate(input)).samples

    assert len(samples) == 2 and [s.metadata["segment"] for s in samples] == [0, 1]
    for s in samples:
        s.validate()
        assert s.reward == 0.40 and s.rollout_id == 7 and s.status == Sample.Status.COMPLETED
        start, length = s.critic_span
        region = s.critic_tokens[len(s.critic_tokens) - s.critic_response_length :]
        assert region[start : start + length] == s.tokens[-s.response_length :]
        assert s.metadata["problem_id"] == "p1" and s.metadata["n_compactions"] == 1
    assert samples[1].critic_tokens == samples[0].tokens + samples[1].tokens
    # every model call saw exactly the actor's context of its segment so far
    assert prompts[0] == samples[0].tokens[: len(prompts[0])] and prompts[2] == samples[1].tokens[: len(prompts[2])]
