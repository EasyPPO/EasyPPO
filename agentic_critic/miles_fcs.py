"""Agentic Frontier-CS as a Miles generate function, with the critic's own input attached to every sample.

    --custom-generate-function-path agentic_critic.miles_fcs.generate

One dataset row is one problem (`prompt` is the statement, `metadata["problem_id"]` names it for the judge).
One call runs one episode (agentic_critic/fcs_env.py) against the SGLang router and the judge server, and
returns one Miles sample per actor segment: the samples of an episode share `rollout_id` and the episode reward.
Each sample also carries the episode-level critic sample (`critic_tokens`, `critic_response_length`,
`critic_loss_mask`) and its own `critic_span` in it, which `--critic-context-mode full_history` uses
: the Full critic reads the whole episode once.

`episode_samples` constructs framework-independent sample fields for the Miles wrapper.
"""

import asyncio
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from agentic_critic import fcs_judge_http
from agentic_critic.fcs_env import ContextFull, Episode, TurnFormat, run_episode
from agentic_critic.miles_samples import critic_episode


def episode_samples(ep: Episode, call_logprobs: list[list[float]]) -> list[dict]:
    """One dict per segment that has actor tokens, with the fields of a Miles sample.

    call_logprobs: the per-token log-probabilities of each model call, in call order (= order of actor events).
    `rollout_log_probs` is 0.0 on tokens the actor did not generate, as Miles' multi-turn examples do.
    Every dict also holds the episode-level critic sample and the segment's (start, length) inside its response region.
    """
    actor_events = [e for e in ep.events if e.actor]
    assert len(actor_events) == len(call_logprobs)
    logprob_of = {id(e): lp for e, lp in zip(actor_events, call_logprobs, strict=True)}
    segments = ep.to_segments()
    while segments and not any(segments[-1].loss_mask):
        segments.pop()  # the episode stopped before the actor produced anything in its last segment
    if not segments:
        return []
    critic, spans = critic_episode(segments)
    out = []
    for s, (seg, span) in enumerate(zip(segments, spans, strict=True)):
        events = [e for e in ep.events if e.segment == s]
        first = next(i for i, e in enumerate(events) if e.actor)
        logprobs = []
        for e in events[first:]:
            lp = logprob_of[id(e)] if e.actor else [0.0] * len(e.tokens)
            assert len(lp) == len(e.tokens)
            logprobs.extend(lp)
        out.append(
            {
                "tokens": seg.tokens,
                "response_length": seg.response_length,
                "loss_mask": seg.loss_mask,
                "rollout_log_probs": logprobs,
                "critic_tokens": critic.tokens,
                "critic_response_length": critic.response_length,
                "critic_loss_mask": critic.loss_mask,
                "critic_span": span,
                "truncated": any(e.kind == "truncation_mark" for e in events),
                "segment": s,
                "metadata": {"summary_span": seg.summary_span},
            }
        )
    return out


_POOL = ThreadPoolExecutor(max_workers=256, thread_name_prefix="fcs-episode")


def _post(url: str, payload: dict, headers: dict | None, timeout: float = 7200) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


async def generate(input):
    from copy import deepcopy

    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, compute_routing_headers
    from miles.utils.types import Sample

    args, base = input.args, input.sample
    assert not args.partial_rollout, "Partial rollout is not supported"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    fmt = TurnFormat(input.state.tokenizer)
    call_logprobs: list[list[float]] = []

    def model_call(prompt_ids: list[int]) -> tuple[list[int], bool]:
        payload, _ = compute_request_payload(args, prompt_ids, input.sampling_params, evaluation=input.evaluation)
        if payload is None:
            raise ContextFull
        output = _post(url, payload, compute_routing_headers(args, base))
        items = output["meta_info"].get("output_token_logprobs") or []
        call_logprobs.append([item[0] for item in items])
        return [item[1] for item in items], output["meta_info"]["finish_reason"]["type"] != "length"

    def judge(code: str):
        return fcs_judge_http.judge(args.fcs_judge_url, str(base.metadata["problem_id"]), code)

    def run() -> Episode:
        return run_episode(
            statement=base.prompt, fmt=fmt, generate=model_call, judge=judge, max_submissions=args.fcs_max_submissions,
            max_turns=args.fcs_max_turns, compact_threshold_tokens=args.fcs_compact_threshold,
        )  # fmt: skip

    ep = await asyncio.get_running_loop().run_in_executor(_POOL, run)
    samples = []
    for fields in episode_samples(ep, call_logprobs):
        sample = deepcopy(base)
        sample.rollout_id = base.rollout_id if base.rollout_id is not None else base.index
        sample.tokens = fields["tokens"]
        sample.response_length = fields["response_length"]
        sample.loss_mask = fields["loss_mask"]
        sample.rollout_log_probs = fields["rollout_log_probs"]
        sample.critic_tokens = fields["critic_tokens"]
        sample.critic_response_length = fields["critic_response_length"]
        sample.critic_loss_mask = fields["critic_loss_mask"]
        sample.critic_span = fields["critic_span"]
        sample.response = fmt.decode(fields["tokens"][-fields["response_length"] :])
        sample.reward = ep.reward
        sample.status = Sample.Status.TRUNCATED if fields["truncated"] else Sample.Status.COMPLETED
        sample.metadata = {
            **(base.metadata or {}), **fields["metadata"], "segment": fields["segment"], "n_compactions": ep.n_compactions,
            "scores": list(ep.scores), "finish_reason": ep.finish_reason, "critic_tokens_in_episode": len(fields["critic_tokens"]),
        }  # fmt: skip
        samples.append(sample)
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser) -> None:
    parser.add_argument("--fcs-judge-url", type=str, default="http://127.0.0.1:8081")
    parser.add_argument("--fcs-max-submissions", type=int, default=6)
    parser.add_argument("--fcs-max-turns", type=int, default=16)
    parser.add_argument("--fcs-compact-threshold", type=int, default=None, help="tokens; omit to disable compaction")


generate.add_arguments = _add_arguments
