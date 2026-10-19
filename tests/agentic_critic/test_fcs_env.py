from dataclasses import dataclass, field

from agentic_critic.critic_input import build_full, build_shared, value_mask
from agentic_critic.fcs_env import COMPACT_PROMPT, TASK_PROMPT, TurnFormat, extract_code, run_episode

EOS = "<|im_end|>"


class CharTokenizer:
    """One token per character; enough to exercise the append-only bookkeeping."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt and len(messages) == 1
        return f"<|im_start|>user\n{messages[0]['content']}{EOS}\n<|im_start|>assistant\n<think>\n"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@dataclass
class FakeCase:
    status: str


@dataclass
class FakeResult:
    score: float
    status: str = "done"
    compile_msg: str = ""
    cases: list = field(default_factory=list)


def code_reply(tag: str) -> str:
    return f"plan for {tag}</think>\n\n```cpp\n// {tag}\nint main(){{}}\n```{EOS}"


def run(replies, scores, finished=None, **kw):
    fmt = TurnFormat(CharTokenizer())
    it, score_it, prompts = iter(replies), iter(scores), []
    fin = iter(finished or [True] * len(replies))

    def generate(prompt_ids):
        prompts.append(fmt.decode(prompt_ids))
        return fmt.encode(next(it)), next(fin)

    ep = run_episode(
        statement="S",
        fmt=fmt,
        generate=generate,
        judge=lambda code: FakeResult(next(score_it), cases=[FakeCase("Accepted"), FakeCase("Wrong Answer")]),
        **kw,
    )
    return ep, prompts, fmt


def text(fmt, ep, seg):
    return fmt.decode(ep.segment_tokens(seg))


def test_extract_code_ignores_the_thinking_block():
    assert extract_code("```cpp\nDRAFT\n```</think>\n```cpp\nFINAL\n```") == "FINAL"
    assert extract_code("```cpp\nONLY IN THINKING\n```</think>\nno code") is None
    assert extract_code("no code here") is None


def test_first_turn_is_the_single_turn_training_prompt():
    ep, prompts, fmt = run([code_reply("a")], [30.0], max_submissions=1)
    expected = fmt.tokenizer.apply_chat_template(
        [{"role": "user", "content": TASK_PROMPT.format(statement="S")}], tokenize=False, add_generation_prompt=True
    )
    assert prompts[0] == expected


def test_reward_is_best_submission_and_stops_at_limit():
    ep, _, _ = run([code_reply(str(i)) for i in range(3)], [10.0, 40.0, 25.0], max_submissions=3)
    assert ep.scores == [10.0, 40.0, 25.0]
    assert ep.reward == 0.40
    assert ep.finish_reason == "submissions_exhausted"
    assert ep.n_compactions == 0


def test_reply_without_code_is_not_a_submission():
    ep, _, _ = run([f"just talking{EOS}", code_reply("a")], [30.0], max_submissions=1)
    assert ep.scores == [30.0]
    assert [e.kind for e in ep.events] == ["task", "action", "no_code", "action"]


def test_context_is_append_only_and_every_prompt_is_the_log_so_far():
    ep, prompts, fmt = run([code_reply(str(i)) for i in range(3)], [1.0, 2.0, 3.0], max_submissions=3)
    for earlier, later in zip(prompts, prompts[1:]):
        assert later.startswith(earlier)
    log = text(fmt, ep, 0)
    assert all(log.startswith(p) for p in prompts)
    # Turn markers stay well formed: every assistant turn is opened once and closed once.
    assert log.count("<|im_start|>assistant\n") == 3 and log.count(EOS) == 3 + 3


def test_truncated_completion_gets_an_end_of_turn_mark_that_is_not_an_actor_token():
    ep, _, fmt = run(["cut off mid", code_reply("a")], [10.0], finished=[False, True], max_submissions=1)
    assert ep.truncated_turns == 1
    kinds = [e.kind for e in ep.events]
    assert kinds[:4] == ["task", "action", "truncation_mark", "no_code"]
    mark = ep.events[2]
    assert not mark.actor and fmt.decode(mark.tokens) == EOS


def test_compaction_starts_a_new_segment_from_task_and_summary():
    replies = [code_reply("a"), f"thinking</think>SUMMARY-1{EOS}", code_reply("b"), f"</think>SUMMARY-2{EOS}", code_reply("c")]
    ep, prompts, fmt = run(replies, [10.0, 20.0, 30.0], max_submissions=3, compact_threshold_tokens=1)
    assert ep.n_compactions == 2
    assert [e.kind for e in ep.events if e.segment == 0] == ["task", "action", "compact_request", "summary"]
    assert [e.kind for e in ep.events if e.segment == 1][:2] == ["resume", "action"]

    # The summary is written with the whole segment in view, and the request carries the judge result.
    assert "// a" in prompts[1] and COMPACT_PROMPT in prompts[1] and "score=10.00/100" in prompts[1]
    # After compaction the actor sees the task and the summary, not the old attempt or its thinking.
    assert "SUMMARY-1" in prompts[2] and "// a" not in prompts[2] and "thinking" not in prompts[2]
    assert prompts[2].startswith("<|im_start|>user\nYou are a competitive programmer.")
    # ... and the environment does not hand the score or the remaining budget back to it.
    assert "10.00" not in prompts[2] and "submission(s) left" not in prompts[2] and "best score" not in prompts[2].lower()
    # The full log still has everything, in time order.
    assert [e.segment for e in ep.events] == sorted(e.segment for e in ep.events)
    assert "".join(fmt.decode(e.tokens) for e in ep.events if e.actor) == "".join(replies)


def test_segments_feed_the_critic_input_code():
    replies = [code_reply("a"), f"</think>SUMMARY-1{EOS}", code_reply("b")]
    ep, prompts, fmt = run(replies, [10.0, 20.0], max_submissions=2, compact_threshold_tokens=1)
    segments = ep.to_segments()
    assert len(segments) == 2
    for s, seg in enumerate(segments):
        assert seg.tokens == ep.segment_tokens(s)
    # Actor inputs are slices of the Full input, and values sit exactly on actor tokens (summary included).
    full = build_full(segments)
    assert full.tokens == [t for e in ep.events for t in e.tokens]
    marked = [t for t, m in zip(full.tokens, value_mask(full, segments)) if m]
    assert fmt.decode(marked) == "".join(replies)
    shared = build_shared(segments)
    assert [ci.tokens for ci in shared] == [seg.tokens for seg in segments]


def test_no_compaction_gives_identical_shared_and_full_inputs():
    ep, _, _ = run([code_reply(str(i)) for i in range(3)], [1.0, 2.0, 3.0], max_submissions=3)
    segments = ep.to_segments()
    (shared,), full = build_shared(segments), build_full(segments)
    assert shared.tokens == full.tokens and shared.response_spans == full.response_spans


def test_running_out_of_turns_never_leaves_an_empty_segment():
    """The last turn must not start a compaction or a next turn that no model call will answer."""
    replies = [f"still thinking{EOS}"] * 3 + [f"</think>SUMMARY{EOS}"] * 3
    ep, _, _ = run(replies, [], max_submissions=2, max_turns=3, compact_threshold_tokens=1)
    assert ep.finish_reason == "max_turns"
    assert ep.events[-1].actor and ep.events[-1].kind == "action"
    assert all(sum(seg.loss_mask) > 0 for seg in ep.to_segments())


def test_the_summary_request_is_never_trained_on():
    """The instruction that asks for a summary is environment text: no policy gradient, no value target."""
    replies = [code_reply("a"), f"</think>SUMMARY-1{EOS}", code_reply("b")]
    ep, prompts, fmt = run(replies, [10.0, 20.0], max_submissions=2, compact_threshold_tokens=1)
    assert COMPACT_PROMPT == "Summarize the text above:" and COMPACT_PROMPT in prompts[1]
    request = next(e for e in ep.events if e.kind == "compact_request")
    assert not request.actor and COMPACT_PROMPT in fmt.decode(request.tokens)
    # in the actor's sample of that segment, every token of the request has loss mask 0 ...
    seg = ep.to_segments()[0]
    start = text(fmt, ep, 0).index(COMPACT_PROMPT)  # one token per character in this tokenizer
    in_response = start - seg.response_start
    assert in_response >= 0 and not any(seg.loss_mask[in_response : in_response + len(COMPACT_PROMPT)])
    # ... while the summary the actor wrote in reply is trained on
    summary = next(e for e in ep.events if e.kind == "summary")
    assert summary.actor
    # and the same holds in the critic's whole-episode sample
    from agentic_critic.miles_samples import critic_episode

    sample, _ = critic_episode(ep.to_segments())
    log = fmt.decode(sample.tokens)
    at = log.index(COMPACT_PROMPT) - (len(sample.tokens) - sample.response_length)
    assert not any(sample.loss_mask[at : at + len(COMPACT_PROMPT)])


def test_keep_last_turns_starts_a_new_segment_from_the_task_and_the_last_round_verbatim():
    replies = [code_reply("a"), code_reply("b"), code_reply("c")]
    ep, prompts, fmt = run(replies, [10.0, 20.0, 30.0], max_submissions=3, compact_threshold_tokens=1, compaction="keep_last_turns")
    assert ep.n_compactions == 2 and len(prompts) == 3  # no model call writes a summary
    assert [e.kind for e in ep.events] == ["task", "action", "resume", "action", "resume", "action"]
    # the second prompt is exactly what an uncompacted episode would show for its last round, after the task
    assert prompts[1] == prompts[0] + replies[0] + fmt.decode(fmt.next_user_turn(
        "[submit] submission 1/3: status=done score=10.00/100\nCases: 1 Accepted, 1 Wrong Answer\n"
        "Best score so far: 10.00/100. 2 submission(s) left. Your result is the best score among your submissions.\n"
        "Submit an improved solution. Output ONLY the C++ code wrapped in ```cpp and ```."))
    # the third keeps only the second round: the first attempt and its score are gone
    assert "// b" in prompts[2] and "// a" not in prompts[2] and "score=10.00" not in prompts[2]
    # the copies are not actor tokens; every reply is an actor token exactly once
    assert "".join(fmt.decode(e.tokens) for e in ep.events if e.actor) == "".join(replies)
    assert ep.reward == 0.30


def test_keep_last_turns_keeps_several_rounds_and_a_cut_reply_keeps_its_end_mark():
    replies = [code_reply("a"), code_reply("b").replace(EOS, ""), code_reply("c")]
    ep, prompts, fmt = run(replies, [10.0, 20.0, 30.0], finished=[True, False, True], max_submissions=3,
                           compact_threshold_tokens=1, compaction="keep_last_turns", keep_turns=2)  # fmt: skip
    assert "// a" in prompts[2] and "// b" in prompts[2]
    assert replies[1] + EOS in prompts[2]
    # with enough rounds kept, the actor sees what it would see without compaction
    no_compaction, plain, _ = run(replies, [10.0, 20.0, 30.0], finished=[True, False, True], max_submissions=3)
    assert prompts == plain and no_compaction.n_compactions == 0


def test_keep_last_turns_segments_feed_the_critic_input_code():
    replies = [code_reply("a"), code_reply("b"), code_reply("c")]
    ep, _, fmt = run(replies, [10.0, 20.0, 30.0], max_submissions=3, compact_threshold_tokens=1, compaction="keep_last_turns")
    segments = ep.to_segments()
    assert [seg.tokens for seg in segments] == [ep.segment_tokens(s) for s in range(3)]
    # values sit exactly on the replies, once each: the copied rounds are prompt text of their segment
    full = build_full(segments)
    assert full.tokens == [t for e in ep.events for t in e.tokens]
    assert fmt.decode([t for t, m in zip(full.tokens, value_mask(full, segments)) if m]) == "".join(replies)
    assert [ci.tokens for ci in build_shared(segments)] == [seg.tokens for seg in segments]
