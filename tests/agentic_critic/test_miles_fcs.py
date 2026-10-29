from agentic_critic.fcs_env import ContextFull, TurnFormat, run_episode
from agentic_critic.miles_fcs import episode_samples
from tests.agentic_critic.test_fcs_env import EOS, CharTokenizer, FakeCase, FakeResult, code_reply


def run(replies, scores, full_after=None, **kw):
    fmt = TurnFormat(CharTokenizer())
    it, score_it, logprobs = iter(replies), iter(scores), []

    def generate(prompt_ids):
        if full_after is not None and len(logprobs) >= full_after:
            raise ContextFull
        ids = fmt.encode(next(it))
        logprobs.append([-0.5 - 0.01 * len(logprobs)] * len(ids))
        return ids, True

    ep = run_episode(statement="S", fmt=fmt, generate=generate, judge=lambda code: FakeResult(next(score_it), cases=[FakeCase("Accepted")]), **kw)
    return ep, logprobs, fmt


def test_one_sample_per_segment_with_aligned_fields():
    replies = [code_reply("a"), f"</think>SUMMARY-1{EOS}", code_reply("b"), code_reply("c")]
    ep, logprobs, fmt = run(replies, [10.0, 20.0, 30.0], max_submissions=3, compact_threshold_tokens=1, max_turns=2)
    samples = episode_samples(ep, logprobs)
    assert [s["segment"] for s in samples] == list(range(ep.n_compactions + 1))
    log = [t for e in ep.events for t in e.tokens]
    seen = 0
    for s in samples:
        n = s["response_length"]
        assert len(s["loss_mask"]) == len(s["rollout_log_probs"]) == n and s["loss_mask"][0] == 1
        # log-probabilities are non-zero exactly on actor tokens
        assert [lp != 0.0 for lp in s["rollout_log_probs"]] == [bool(m) for m in s["loss_mask"]]
        # the critic's sample is the whole append-only log; this segment's response region sits at its span
        assert s["critic_tokens"] == log and not s["truncated"]
        region = log[len(log) - s["critic_response_length"] :]
        start, length = s["critic_span"]
        assert length == n and region[start : start + length] == s["tokens"][-n:]
        assert s["critic_loss_mask"][start : start + length] == s["loss_mask"]
        seen += len(s["tokens"])
    # every actor token appears in exactly one sample
    marked = "".join(fmt.decode([t for t, m in zip(s["tokens"][-s["response_length"] :], s["loss_mask"]) if m]) for s in samples)
    assert marked == "".join(fmt.decode(e.tokens) for e in ep.events if e.actor)
    for sample in samples:
        summary_events = [e for e in ep.events if e.segment == sample["segment"] and e.kind == "summary"]
        span = sample["metadata"]["summary_span"]
        if summary_events:
            start, end = span
            assert sample["tokens"][start:end] == summary_events[0].tokens
        else:
            assert span is None


def test_no_compaction_gives_one_sample_whose_critic_tokens_equal_its_tokens():
    ep, logprobs, _ = run([code_reply("a"), code_reply("b")], [10.0, 20.0], max_submissions=2)
    (sample,) = episode_samples(ep, logprobs)
    assert sample["critic_tokens"] == sample["tokens"] and sample["critic_span"] == (0, sample["response_length"])
    assert sample["critic_response_length"] == sample["response_length"] and sample["critic_loss_mask"] == sample["loss_mask"]


def test_context_full_ends_the_episode_and_drops_a_segment_without_actor_tokens():
    replies = [code_reply("a"), f"</think>SUMMARY{EOS}", code_reply("b")]
    ep, logprobs, _ = run(replies, [10.0, 20.0], full_after=2, max_submissions=3, compact_threshold_tokens=1)
    assert ep.finish_reason == "context_full" and ep.n_compactions == 1 and ep.scores == [10.0]
    samples = episode_samples(ep, logprobs)
    assert [s["segment"] for s in samples] == [0]
