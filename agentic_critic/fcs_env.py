"""Agentic Frontier-CS episode: decode -> submit (tool) -> ... -> compaction -> decode.

A lightweight in-process replica of the official Harbor adapter's iterative-submission setting
(adapters/frontier-cs-algorithm in FrontierCS/Frontier-CS): the agent writes a C++ solution, may
submit it to the graded judge several times and gets the score back, and the episode return is
the best score over its submissions. There is no shell; the only tool is submit, triggered by a
fenced ```cpp block in the reply.

The trajectory is kept at token level and is append-only within a segment: every model call sees
exactly the tokens already in the segment, and its completion is appended verbatim (thinking
included). Chat templates that rewrite earlier turns are therefore not used after the first
turn; later turns are appended with the same turn markers. This makes the actor's input a
contiguous slice of the full log, which is what the critic-input code relies on.

The first turn describes the submission budget, compiler and judge feedback.

When the segment exceeds `compact_threshold_tokens`, the actor is asked to write a summary, and
a new segment starts from the task plus that summary.

A second, rule-based kind of compaction (`compaction="keep_last_turns"`) writes no summary: the new
segment starts from the task and a verbatim copy of the last `keep_turns` rounds (the actor's reply
and the judge's answer to it); everything older is dropped. The copy is environment text, so its
tokens are not actor tokens: they were valued where they were first generated.

The model, the judge and the tokenizer are injected so the loop can be tested without a GPU.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_critic.critic_input import Segment

DEFAULT_MAX_SUBMISSIONS = 6
DEFAULT_MAX_TURNS = 16

FCS_REASONING_EFFORT = "medium"
FCS_SUMMARY_EFFORT = "low"

TASK_PROMPT = (
    "You are a competitive programmer. You may submit up to {max_submissions} C++ solutions. "
    "Your result is the highest score among your submissions. The judge compiles each submission "
    "with g++ -O2 -pipe -std=gnu++17. Feedback includes the submission status, score out of 100, "
    "best score so far, and remaining submissions. For interactive problems your program communicates "
    "with the judge using the protocol in the statement. Feedback reports case counts by status. "
    "Compilation errors include a compiler diagnostic excerpt.\n\n"
    "To submit a solution, output your complete program in a ```cpp code block and end your turn "
    "immediately after the closing ```. The harness then automatically submits the program to the judge "
    "and returns the judge's feedback.\n\n"
    "{statement}\n\nGenerate solution code:"
)

FEEDBACK_PROMPT = """[submit] submission {index}/{max_submissions}: status={status} score={score:.2f}/100
{detail}
Best score so far: {best:.2f}/100. {remaining} submission(s) left. Your result is the best score among your submissions."""

TRUNCATED_NO_SUBMISSION_PROMPT = "Your reply reached the output token limit. No solution was submitted this turn."
NO_CODE_PROMPT = "No ```cpp code block was found in your reply."

NEXT_PROMPT = "Submit an improved solution. Output ONLY the C++ code wrapped in ```cpp and ```."

COMPACT_PROMPT = (
    "Write a summary for yourself to use on the next turn. Your summary replaces the conversation history. "
    "You will receive the original task and your summary, then continue the same episode with your "
    "remaining submissions. Writing the summary does not submit anything to the judge or use a submission."
)

# After a compaction the actor restarts from the task and its own summary, and nothing else: the environment does
# not repeat the best score or the number of submissions left. What the actor knows about its
# standing is what its summary kept; learning what to keep is part of what RL trains.
RESUME_PROMPT = """{task}

Earlier progress on this task was summarized as follows.

{summary}

""" + NEXT_PROMPT.replace("{", "{{").replace("}", "}}")

_CODE_RE = re.compile(r"```(?:cpp|c\+\+|cc)\s*\n(.*?)```", re.S | re.I)
_SENTINEL = "␟USER-CONTENT␟"


class TurnFormat:
    """Turn markers of a chat template, applied append-only.

    `tokenizer` needs `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`,
    `encode(text) -> ids` without special tokens added, and `decode(ids) -> text`.
    """

    def __init__(self, tokenizer, eos_token: str = "<|im_end|>"):
        messages = [{"role": "user", "content": _SENTINEL}]
        options = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
        rendered = tokenizer.apply_chat_template(messages, reasoning_effort=FCS_REASONING_EFFORT, **options)
        self.pre, self.post = rendered.split(_SENTINEL)
        summary = tokenizer.apply_chat_template(messages, reasoning_effort=FCS_SUMMARY_EFFORT, **options)
        self.summary_pre, self.summary_post = summary.split(_SENTINEL)
        self.tokenizer = tokenizer
        self.eos_text = eos_token

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def first_user_turn(self, content: str) -> list[int]:
        return self.encode(self.pre + content + self.post)

    def next_user_turn(self, content: str, *, summary: bool = False) -> list[int]:
        # A completion ends with the end-of-turn token; the template puts a newline after it.
        prefix, suffix = (self.summary_pre, self.summary_post) if summary else (self.pre, self.post)
        return self.encode("\n" + prefix.lstrip() + content + suffix)

    def end_of_turn(self) -> list[int]:
        return self.encode(self.eos_text)


@dataclass
class Event:
    tokens: list[int]
    actor: bool  # generated by the actor (thinking, answers and summaries); everything else is masked
    segment: int
    kind: str  # task | action | feedback | no_code | truncated_no_submission | compact_request | summary | resume | truncation_mark
    # feedback / no_code turns end with NEXT_PROMPT; a compact_request turn carries the same result text
    # followed by COMPACT_PROMPT instead.


@dataclass
class Episode:
    events: list[Event] = field(default_factory=list)  # everything, in time order (the critic's Full view)
    scores: list[float] = field(default_factory=list)  # one per submission
    n_compactions: int = 0
    truncated_turns: int = 0  # completions that hit the per-turn token limit
    finish_reason: str = ""

    @property
    def reward(self) -> float:
        return max(self.scores, default=0.0) / 100.0

    def segment_tokens(self, index: int) -> list[int]:
        """The actor's context for one segment (the Shared view)."""
        return [t for e in self.events if e.segment == index for t in e.tokens]

    def to_segments(self) -> list[Segment]:
        """One training sample per segment: response region starts at the first actor token."""
        out = []
        for s in range(self.n_compactions + 1):
            events = [e for e in self.events if e.segment == s]
            first = next((i for i, e in enumerate(events) if e.actor), len(events))
            prompt = [t for e in events[:first] for t in e.tokens]
            response = [t for e in events[first:] for t in e.tokens]
            mask = [int(e.actor) for e in events[first:] for _ in e.tokens]
            out.append(Segment(tokens=prompt + response, response_length=len(response), loss_mask=mask))
        return out


def answer_text(reply: str) -> str:
    """The part of a completion after its thinking block."""
    return reply.rsplit("</think>", 1)[-1]


def summary_text(reply: str) -> str:
    """Return visible summary text, including when generation ended inside thinking."""
    _, closed, visible = reply.rpartition("</think>")
    return visible if closed else ""


def extract_code(reply: str) -> str | None:
    blocks = _CODE_RE.findall(answer_text(reply))
    return blocks[-1].strip() if blocks else None


def format_feedback(result, index: int, max_submissions: int, best: float) -> str:
    """Score plus aggregate case statuses; no test inputs or expected outputs are revealed."""
    if result.status == "Compile Error":
        detail = "Compile error:\n" + result.compile_msg[-500:]
    elif result.status == "Judge Error":
        detail = "The judge returned an error for this submission."
    else:
        counts = Counter(result.case_statuses)
        detail = "Cases: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    return FEEDBACK_PROMPT.format(
        index=index,
        max_submissions=max_submissions,
        status=result.status,
        score=result.score,
        detail=detail,
        best=best,
        remaining=max_submissions - index,
    )


class ContextFull(Exception):
    """Raised by a `generate` callback when the context has no room for another completion."""


def run_episode(
    *,
    statement: str,
    fmt: TurnFormat,
    generate: Callable[[list[int]], tuple[list[int], bool]],
    judge: Callable[[str], object],
    max_submissions: int = DEFAULT_MAX_SUBMISSIONS,
    max_turns: int = DEFAULT_MAX_TURNS,
    compact_threshold_tokens: int | None = None,
    compaction: str = "summary",
    keep_turns: int = 1,
) -> Episode:
    """Run one episode.

    generate(prompt_token_ids) -> (completion_token_ids, finished). `finished` is False when the
    completion hit the per-turn token limit before the end-of-turn token.
    It may raise ContextFull instead; the episode then ends there with what it has.
    judge(code) -> a JudgeResult (see fcs_judge).
    `compact_threshold_tokens=None` disables compaction (the zero-compaction sanity setting).
    `compaction`: "summary" (the actor writes one) or "keep_last_turns" (the last `keep_turns` rounds are kept
    verbatim and nothing is written).
    """
    assert compaction in ("summary", "keep_last_turns") and keep_turns >= 1
    ep = Episode()
    seg = 0
    task = TASK_PROMPT.format(statement=statement, max_submissions=max_submissions)

    def add(tokens: list[int], actor: bool, kind: str) -> None:
        ep.events.append(Event(list(tokens), actor, seg, kind))

    rounds: list[list[int]] = []  # per model turn of the episode: its completion (with an end-of-turn mark if cut)

    def call(kind: str) -> tuple[str, bool]:
        completion, finished = generate(ep.segment_tokens(seg))
        add(completion, True, kind)
        rounds.append(list(completion))
        if not finished:
            ep.truncated_turns += 1
            add(fmt.end_of_turn(), False, "truncation_mark")
            rounds[-1] += fmt.end_of_turn()
        return fmt.decode(completion), finished

    add(fmt.first_user_turn(task), False, "task")

    def turns() -> None:
        nonlocal seg
        for turn in range(max_turns):
            reply, finished = call("action")
            if not finished:
                message, kind = TRUNCATED_NO_SUBMISSION_PROMPT, "truncated_no_submission"
            elif (code := extract_code(reply)) is None:
                message, kind = NO_CODE_PROMPT, "no_code"
            else:
                result = judge(code)
                ep.scores.append(result.score)
                if len(ep.scores) >= max_submissions:
                    ep.finish_reason = "submissions_exhausted"
                    return
                message, kind = format_feedback(result, len(ep.scores), max_submissions, max(ep.scores)), "feedback"

            if turn == max_turns - 1:
                # No further model call will follow, so neither a next turn nor a compaction is started.
                ep.finish_reason = "max_turns"
                return
            turn = fmt.next_user_turn(message + "\n" + NEXT_PROMPT)
            rounds[-1] += turn  # a round is the reply and the environment's answer to it
            over = compact_threshold_tokens is not None and len(ep.segment_tokens(seg)) + len(turn) > compact_threshold_tokens
            if not over:
                add(turn, False, kind)
            elif compaction == "keep_last_turns":
                # No summary: the new segment is the task and the last rounds, copied as environment text.
                ep.n_compactions += 1
                seg += 1
                add(fmt.first_user_turn(task) + [t for r in rounds[-keep_turns:] for t in r], False, "resume")
            else:
                # The same result text, but the turn asks for a summary instead of the next solution.
                add(fmt.next_user_turn(message + "\n\n" + COMPACT_PROMPT, summary=True), False, "compact_request")
                summary_reply, _ = call("summary")
                summary = summary_text(summary_reply).replace(fmt.eos_text, "").strip()
                rounds.pop()  # the summary is not a round that could be kept
                ep.n_compactions += 1
                seg += 1
                resume = RESUME_PROMPT.format(task=task, summary=summary)
                add(fmt.first_user_turn(resume), False, "resume")

    try:
        turns()
    except ContextFull:
        ep.finish_reason = "context_full"
    return ep
