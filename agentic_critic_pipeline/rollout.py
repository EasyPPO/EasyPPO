"""Append-only Qwen3.5 trajectories with actor-generated context compaction.

Generated IDs, including thinking and summaries, are retained exactly.
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import digest, verify_problem
from .environment import HTTPJudge, LocalJudge, request_json

TASK = (
    "You are a competitive programmer. Solve the following problem in C++. "
    "Output ONLY the C++ code wrapped in ```cpp and ```. No explanation."
    "\n\n{statement}\n\nGenerate solution code:"
)
NEXT = "Submit an improved solution. Output ONLY the C++ code wrapped in ```cpp and ```."
SUMMARY = "Summarize the text above:"


def format_feedback(result, index, max_submissions, best):
    """Prompt feedback contains only aggregate outcomes or bounded compiler text."""
    if result["status"] == "Compile Error":
        detail = "Compile error:\n" + str(result.get("detail", ""))[-500:]
    else:
        counts = Counter(case["status"] for case in result["cases"])
        detail = "Cases: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    return (
        f"[submit] submission {index}/{max_submissions}: status={result['status']} "
        f"score={result['score']:.2f}/100\n{detail}\n"
        f"Best score so far: {best:.2f}/100. {max_submissions - index} submission(s) left. "
        "Your result is the best score among your submissions."
    )


class ContextFull(Exception):
    pass


class TurnFormat:
    def __init__(self, tokenizer, enable_thinking=True):
        self.tokenizer = tokenizer
        self.thinking = enable_thinking
        self.eos = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(self.eos) != 1 or self.eos[0] == tokenizer.unk_token_id:
            raise ValueError("This runner requires the Qwen im_start/im_end chat format")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def first(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def follow(self, text):
        # Re-rendering prior messages can erase thinking; append only the new turn.
        suffix = "<think>\n" if self.thinking else "<think>\n\n</think>\n\n"
        return self.encode("\n<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n" + suffix)


def turn_format(tokenizer, enable_thinking=True):
    """Build tokenizer-derived turn markers with explicit thinking mode."""
    from agentic_critic_runtime.fcs_env import TurnFormat as RuntimeTurnFormat

    class TokenizerView:
        def apply_chat_template(self, *args, **kwargs):
            kwargs["enable_thinking"] = enable_thinking
            return tokenizer.apply_chat_template(*args, **kwargs)

        def encode(self, *args, **kwargs):
            return tokenizer.encode(*args, **kwargs)

        def decode(self, ids):
            return tokenizer.decode(ids, skip_special_tokens=False)

    fmt = RuntimeTurnFormat(TokenizerView())
    fmt.thinking = enable_thinking
    return fmt


class HTTPActorSGLang:
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer

    def __call__(self, prompt, max_tokens, seed):
        if len(prompt) + max_tokens > self.args.context_length:
            raise ContextFull()
        result = request_json(
            self.args.actor_url.rstrip("/") + "/generate",
            {
                "input_ids": prompt,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "sampling_seed": seed,
                    "stop_token_ids": [self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
                    "skip_special_tokens": False,
                    "no_stop_trim": True,
                },
                "return_prompt_token_ids": True,
                "stream": False,
            },
            timeout=self.args.request_timeout,
        )
        ids = result.get("output_ids")
        if not isinstance(ids, list) or not ids or not all(type(token) is int for token in ids):
            raise ValueError("SGLang must return generated output_ids; text retokenization is forbidden")
        if result.get("prompt_token_ids") != prompt:
            raise ValueError("Server changed or omitted supplied prompt token IDs")
        reason = result.get("meta_info", {}).get("finish_reason", {}).get("type")
        if reason not in ("stop", "length"):
            raise RuntimeError(f"Actor request did not finish normally: {reason}")
        return ids, reason


class TransformersActor:
    def __init__(self, args, tokenizer):
        import torch
        from transformers import AutoModelForImageTextToText

        from .kernels import configure_training_kernels

        self.kernel_config = configure_training_kernels()
        self.args, self.tokenizer = args, tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            revision=args.model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda:0",
        ).eval()

    def __call__(self, prompt, max_tokens, seed):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import set_seed

        if len(prompt) + max_tokens > self.args.context_length:
            raise ContextFull()
        set_seed(seed)
        ids = torch.tensor([prompt], device="cuda", dtype=torch.long)
        params = {
            "max_new_tokens": max_tokens,
            "do_sample": self.args.temperature > 0,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.args.temperature > 0:
            params.update(temperature=self.args.temperature, top_p=self.args.top_p)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **params)
        generated = output[0, len(prompt) :].tolist()
        return generated, "stop" if generated[-1] == params["eos_token_id"] else "length"


def to_segments(events):
    segments = []
    for segment in sorted({e["segment"] for e in events}):
        rows = [e for e in events if e["segment"] == segment]
        tokens = [t for e in rows for t in e["tokens"]]
        mask = [int(e["actor"]) for e in rows for _ in e["tokens"]]
        if 1 not in mask:
            continue
        start = mask.index(1)
        segments.append({"input_ids": tokens, "response_start": start, "loss_mask": mask[start:]})
    return segments


def run_episode(
    statement,
    fmt,
    actor,
    judge,
    problem,
    *,
    seed,
    max_turns=10,
    max_submissions=6,
    max_response_tokens=24576,
    max_summary_tokens=24576,
    max_episode_response_tokens=120000,
    compact_threshold_tokens=12000,
    max_full_tokens=2000000,
):
    """Run an episode through the actor and judge transports.

    Quotas and retry policy live in collect.py. Turn ordering, code extraction,
    score feedback, summary compaction and final returns live in fcs_env.
    """
    from dataclasses import asdict
    from types import SimpleNamespace

    from agentic_critic_runtime import fcs_env

    generated = 0
    generation_index = 0
    action_index = -1
    latest_action_text = ""
    budget_reason = None
    metadata = []
    submissions = []
    missing_stop = False
    pending_summary = False

    class FormatAdapter:
        eos_text = getattr(fmt, "eos_text", "<|im_end|>")

        def encode(self, text):
            return fmt.encode(text)

        def decode(self, ids):
            return fmt.decode(ids)

        def first_user_turn(self, text):
            nonlocal pending_summary, missing_stop
            pending_summary = False
            missing_stop = False
            fn = getattr(fmt, "first_user_turn", None) or fmt.first
            return fn(text)

        def next_user_turn(self, text):
            nonlocal missing_stop, pending_summary, budget_reason
            pending_summary = text.endswith(fcs_env.COMPACT_PROMPT)
            if pending_summary and max_episode_response_tokens - generated < 2:
                budget_reason = "episode_response_budget"
                raise fcs_env.ContextFull()
            fn = getattr(fmt, "next_user_turn", None) or fmt.follow
            tokens = fn(text)
            if missing_stop:
                tokens = self.end_of_turn() + tokens
            return tokens

        def end_of_turn(self):
            fn = getattr(fmt, "end_of_turn", None)
            return fn() if fn else fmt.eos

    adapted_fmt = FormatAdapter()

    def generate(prompt):
        nonlocal generated, generation_index, budget_reason, missing_stop, action_index, latest_action_text
        # The final user turn identifies an action or a summary request.
        summary = pending_summary
        configured = max_summary_tokens if summary else max_response_tokens
        limit = min(configured, max_episode_response_tokens - generated - int(summary))
        if limit < 1:
            budget_reason = "episode_response_budget"
            raise fcs_env.ContextFull()
        # Full-history size includes copied resume prompts, not only actor context.
        # Its exact eligibility is checked after the episode returns.
        if len(prompt) + limit + len(adapted_fmt.end_of_turn()) > max_full_tokens:
            budget_reason = "full_token_budget"
            raise fcs_env.ContextFull()
        before = generated
        try:
            ids, reason = actor(prompt, limit, (seed + generation_index) % 2147483647)
        except ContextFull:
            budget_reason = "context_full"
            raise fcs_env.ContextFull() from None
        if not ids or len(ids) > limit:
            raise ValueError("Actor exceeded the requested generated-token limit or returned no tokens")
        generation_index += 1
        if not summary:
            action_index += 1
            latest_action_text = fmt.decode(ids)
        generated += len(ids)
        missing_stop = reason == "stop" and ids[-len(adapted_fmt.end_of_turn()) :] != adapted_fmt.end_of_turn()
        metadata.append(
            {
                "finish_reason": reason,
                "configured_max_tokens": configured,
                "requested_max_tokens": limit,
                "episode_response_tokens_before": before,
                "response_budget_limited": limit < configured,
            }
        )
        return ids, reason == "stop"

    def submit(code):
        result = judge(problem, code)
        # Preserve the raw fenced-block hash as observational provenance; the
        # episode runner decides which stripped code is actually submitted.
        raw_blocks = fcs_env._CODE_RE.findall(fcs_env.answer_text(latest_action_text))
        raw_code = raw_blocks[-1] if raw_blocks else code
        submissions.append(
            {**result, "turn": action_index, "code_sha256": hashlib.sha256(raw_code.encode()).hexdigest()}
        )
        return SimpleNamespace(
            score=float(result["score"]),
            status=result["status"],
            compile_msg=str(result.get("detail", "")),
            cases=[SimpleNamespace(status=case["status"]) for case in result["cases"]],
        )

    ep = fcs_env.run_episode(
        statement=statement,
        fmt=adapted_fmt,
        generate=generate,
        judge=submit,
        max_submissions=max_submissions,
        max_turns=max_turns,
        compact_threshold_tokens=compact_threshold_tokens or None,
        compaction="summary",
    )
    events = [dict(asdict(event), text=fmt.decode(event.tokens)) for event in ep.events]
    for event, detail in zip((e for e in events if e["actor"]), metadata, strict=True):
        event.update(detail)
        if event["kind"] == "summary":
            raw = event["text"]
            event["thinking_complete"] = not getattr(fmt, "thinking", True) or "</think>" in raw
            event["summary_source"] = (
                "final_answer"
                if "</think>" in raw
                else "unclosed_thinking"
                if getattr(fmt, "thinking", True)
                else "no_thinking"
            )
            event["summary_empty"] = not fcs_env.answer_text(raw).replace(adapted_fmt.eos_text, "").strip()
    for submission, segment in zip(submissions, ep.submission_segments, strict=True):
        submission["segment"] = segment
    return {
        "events": events,
        "segments": to_segments(events),
        "scores": ep.scores,
        "reward": ep.reward,
        "submissions": submissions,
        "submission_segments": ep.submission_segments,
        "n_compactions": ep.n_compactions,
        "truncated_turns": ep.truncated_turns,
        "actor_tokens": generated,
        "max_episode_response_tokens": max_episode_response_tokens,
        "finish_reason": budget_reason or ep.finish_reason,
        "episode_runner": "agentic_critic_runtime.fcs_env.run_episode",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=("train", "validation", "test"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=("transformers", "sglang"), default="sglang")
    p.add_argument("--model", default="HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1")
    p.add_argument("--model-revision", default="016efe4f9b8d1f472b412a26bd43e9b3f690a223")
    p.add_argument("--served-model-name", default="qwen35-9b")
    p.add_argument("--actor-url", default="http://127.0.0.1:8000")
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--judge", choices=("local", "http"), required=True)
    p.add_argument("--judge-url")
    p.add_argument("--testlib-dir")
    p.add_argument("--judge-cache", default="artifacts/judge-cache")
    p.add_argument("--allow-local-execution", action="store_true")
    p.add_argument("--samples-per-problem", type=int)
    p.add_argument("--sample-start", type=int)
    p.add_argument("--problem-limit", type=int)
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--context-length", type=int, default=131072)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-submissions", type=int, default=6)
    p.add_argument("--max-response-tokens", type=int, default=24576)
    p.add_argument("--max-summary-tokens", type=int, default=24576)
    p.add_argument(
        "--max-episode-response-tokens",
        type=int,
        default=120000,
        help="Cumulative generated action and summary tokens, including thinking; prompts excluded",
    )
    p.add_argument("--compact-threshold-tokens", type=int, default=12000)
    p.add_argument("--max-full-tokens", type=int, default=2000000)
    p.add_argument("--disable-thinking", action="store_true")
    args = p.parse_args()
    if any(
        getattr(args, key) < 1
        for key in (
            "context_length",
            "max_turns",
            "max_submissions",
            "max_response_tokens",
            "max_summary_tokens",
            "max_episode_response_tokens",
            "max_full_tokens",
        )
    ):
        p.error("Token, context and turn budgets must be positive")
    if args.compact_threshold_tokens < 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        p.error("Invalid compaction or sampling parameters")
    if not 0 <= args.shard_index < args.num_shards or args.workers < 1:
        p.error("Invalid shard/workers")
    if args.workers > 1 and (args.backend == "transformers" or args.judge == "local"):
        p.error("Concurrency requires sglang + HTTP judge; local backends are sequential")
    manifest = json.loads(Path(args.manifest).read_text())
    declared_hash = manifest.pop("manifest_hash")
    if digest(manifest) != declared_hash:
        raise ValueError("Manifest checksum mismatch")
    sampling = manifest["sampling"][args.split]
    samples = args.samples_per_problem if args.samples_per_problem is not None else sampling["samples_per_problem"]
    start = args.sample_start if args.sample_start is not None else sampling["sample_start"]
    if (
        samples < 1
        or start < 0
        or (args.split == "train" and start + samples > 64)
        or (args.split == "validation" and start < 64)
    ):
        p.error("Train sample indices must be <64; validation indices >=64; counts positive")
    problems = manifest["splits"][args.split][: args.problem_limit]
    for problem in problems:
        verify_problem(problem)
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "output",
            "workers",
            "manifest",
            "shard_index",
            "num_shards",
            "max_episodes",
            "problem_limit",
            "judge_cache",
            "actor_url",
            "judge_url",
        )
    }
    config.update(manifest_hash=declared_hash, samples_per_problem=samples, sample_start=start)
    config_hash = digest(config)
    jobs = []
    for problem in problems:
        for sample in range(start, start + samples):
            identity = f"{problem['problem_id']}:{args.split}:{sample}"
            seed = int(hashlib.sha256((str(manifest["seed"]) + identity).encode()).hexdigest()[:8], 16) % 2147483647
            jobs.append((problem, sample, identity, seed))
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_index]
    if args.max_episodes is not None:
        jobs = jobs[: args.max_episodes]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        completed = set()
        for line in stream:
            record = json.loads(line)  # partial/corrupt JSON must fail loudly
            if record["config_hash"] != config_hash:
                raise ValueError("Resume configuration differs; use a new output path")
            if record["episode_id"] in completed:
                raise ValueError("Duplicate episode in output")
            completed.add(record["episode_id"])
        jobs = [job for job in jobs if job[2] not in completed]
        if not jobs:
            print(json.dumps({"status": "already_complete", "episodes": len(completed)}))
            return
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
        fmt = turn_format(tokenizer, not args.disable_thinking)
        actor = TransformersActor(args, tokenizer) if args.backend == "transformers" else HTTPActorSGLang(args, tokenizer)
        if args.judge == "local":
            if not args.allow_local_execution or not args.testlib_dir:
                p.error("Local judge requires --allow-local-execution and --testlib-dir")
            judge = LocalJudge(args.judge_cache, args.testlib_dir)
        else:
            if not args.judge_url:
                p.error("HTTP judge requires --judge-url")
            judge = HTTPJudge(args.judge_url)

        def work(job):
            problem, sample, identity, seed = job
            started = time.monotonic()
            episode = run_episode(
                (Path(problem["path"]) / "statement.txt").read_text(),
                fmt,
                actor,
                judge,
                problem,
                seed=seed,
                **{
                    key: getattr(args, key)
                    for key in (
                        "max_turns",
                        "max_submissions",
                        "max_response_tokens",
                        "max_summary_tokens",
                        "max_episode_response_tokens",
                        "compact_threshold_tokens",
                        "max_full_tokens",
                    )
                },
            )
            if not episode["segments"]:
                raise ValueError(f"No actor tokens for {identity}; initial prompt exceeds configured token budget")
            return {
                "schema_version": 1,
                "episode_id": identity,
                "problem_id": problem["problem_id"],
                "split": args.split,
                "sample_index": sample,
                "seed": seed,
                "config": config,
                "config_hash": config_hash,
                "source_revision": problem["source_revision"],
                "model": args.model,
                "tokenizer": args.model,
                "elapsed_seconds": time.monotonic() - started,
                **episode,
            }

        stream.seek(0, os.SEEK_END)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Local uses the main thread so preexec_fn is never used in a worker thread.
            results = map(work, jobs) if args.workers == 1 else executor.map(work, jobs)
            for episode in results:
                stream.write(json.dumps(episode, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(
                    json.dumps(
                        {
                            "episode_id": episode["episode_id"],
                            "reward": episode["reward"],
                            "segments": len(episode["segments"]),
                            "finish_reason": episode["finish_reason"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
