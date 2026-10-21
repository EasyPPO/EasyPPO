"""Collect an exact per-problem quota of naturally finished, length-qualified episodes."""

import argparse
import fcntl
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import collection_quotas, digest, verify_problem
from .environment import HTTPJudge, LocalJudge
from .records import validate_episode
from .rollout import HTTPActorSGLang, TransformersActor, run_episode, to_segments, turn_format

NATURAL_FINISHES = {"max_turns", "submissions_exhausted"}
SPLIT_INDEX = {"train": 0, "validation": 1, "test": 2}


def qualification(record, limit):
    length = sum(len(segment["input_ids"]) for segment in record["segments"])
    if record["finish_reason"] not in NATURAL_FINISHES:
        return False, "non_natural_finish:" + record["finish_reason"], length
    if length >= limit:
        return False, "full_length_at_or_above_limit", length
    return True, None, length


def candidate(manifest, split, problem_id, attempt):
    # Disjoint 64-seed blocks separate every action/summary call across up to
    # 4,096 attempts per split and 2,047 problems (32 action turns maximum).
    identifiers = sorted({p["problem_id"] for rows in manifest["splits"].values() for p in rows})
    if len(identifiers) > 2047 or not 0 <= attempt < 4096:
        raise ValueError("Candidate seed namespace exhausted")
    sample = 3 * attempt + SPLIT_INDEX[split]
    seed = (
        manifest["seed"] + (identifiers.index(problem_id) * 3 + SPLIT_INDEX[split]) * 4096 * 64 + attempt * 64
    ) % 2147483647
    return sample, seed, f"{problem_id}:{split}:{sample}"


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def collect(manifest, settings, output_root, produce):
    """`produce(problem, attempt, sample, seed, identity)` returns a complete episode.

    Only completed real attempts enter raw.jsonl. An infrastructure failure is a
    stop, and a later resume retries that unfinished attempt with the same seed.
    """
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _collect_locked(manifest, settings, output, produce)


def _collect_locked(manifest, settings, output, produce):
    split = settings["split"]
    config_hash = digest(
        {k: v for k, v in settings.items() if k not in ("workers", "shard_index", "num_shards", "problem_limit")}
    )
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Collection configuration changed; use a new output directory")
    else:
        atomic_json(settings_path, settings)
    problems = manifest["splits"][split][: settings.get("problem_limit")]
    problems = [p for i, p in enumerate(problems) if i % settings["num_shards"] == settings["shard_index"]]
    if not problems:
        raise ValueError("No problems in this collection shard")
    quotas = settings.get("quota_by_problem")
    if quotas is None:
        override = settings.get("target_per_problem")
        quotas = (
            {problem["problem_id"]: override for problem in manifest["splits"][split]}
            if override is not None
            else collection_quotas(manifest, split)
        )
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ValueError("Collection quotas must be positive integers")
    if set(quotas) != {problem["problem_id"] for problem in manifest["splits"][split]}:
        raise ValueError("Collection settings quota map must cover the full split")
    limit = settings["max_full_tokens_exclusive"]
    pilot = settings.get("pilot_only", False)
    state = {p["problem_id"]: {"attempted": 0, "accepted": [], "rejected": [], "reasons": {}} for p in problems}
    raw_path = output / "raw.jsonl"
    raw_path.touch(exist_ok=True)
    previous_path = output / "collection.json"
    previous_raw = json.loads(previous_path.read_text())["raw"] if previous_path.exists() else None
    seen = set()

    def validate_record(record):
        validate_episode(record)
        if to_segments(record["events"]) != record["segments"]:
            raise ValueError("Saved segments or masks differ from the original event tokens")
        if not 0 <= record["reward"] <= 1 or abs(record["reward"] - max(record["scores"], default=0) / 100) > 1e-8:
            raise ValueError("Reward differs from the maximum submission score")

    def register(record, offset, checksum):
        validate_record(record)
        pid = record["problem_id"]
        if pid not in state or record["split"] != split or record["config_hash"] != config_hash:
            raise ValueError("Raw episode belongs to a different collection")
        row = state[pid]
        sample, seed, identity = candidate(manifest, split, pid, row["attempted"])
        if (
            record["episode_id"] != identity
            or record["sample_index"] != sample
            or record["seed"] != seed
            or identity in seen
        ):
            raise ValueError("Raw candidates are duplicated, out of order, or have changed seeds")
        if len(row["accepted"]) >= quotas[pid] and not pilot:
            raise ValueError("Raw data exceeds its per-problem qualified quota")
        accepted, reason, length = qualification(record, limit)
        item = {"episode_id": identity, "full_tokens": length, "offset": offset, "sha256": checksum}
        seen.add(identity)
        row["attempted"] += 1
        row["accepted" if accepted else "rejected"].append(item)
        if reason:
            row["reasons"][reason] = row["reasons"].get(reason, 0) + 1

    previous_prefix = hashlib.sha256()
    prefix_count = 0
    with raw_path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            register(json.loads(line), offset, hashlib.sha256(line).hexdigest())
            if previous_raw and prefix_count < previous_raw["episodes"]:
                previous_prefix.update(line)
                prefix_count += 1
    if previous_raw and (
        prefix_count != previous_raw["episodes"] or previous_prefix.hexdigest() != previous_raw["sha256"]
    ):
        raise ValueError("Previously recorded raw attempts changed or were removed")
    mutex = threading.Lock()
    stop = threading.Event()
    errors = []

    def stop_reason(row):
        n = row["attempted"]
        if n >= settings["pilot_attempts"] and len(row["accepted"]) / n < settings["min_acceptance_rate"]:
            return "low_acceptance_rate: measure and admit a larger full-input limit, then start a new collection"
        if n >= settings["max_attempts_per_problem"]:
            return "maximum_attempts_reached: review acceptance and validate a larger limit before restarting"
        return None

    with raw_path.open("ab") as stream:

        def work(problem):
            pid = problem["problem_id"]
            row = state[pid]
            while not stop.is_set():
                if pilot:
                    if row["attempted"] >= settings["pilot_per_problem"]:
                        return
                elif len(row["accepted"]) >= quotas[pid]:
                    return
                elif reason := stop_reason(row):
                    with mutex:
                        errors.append({"problem_id": pid, "reason": reason})
                        stop.set()
                    return
                attempt = row["attempted"]
                sample, seed, identity = candidate(manifest, split, pid, attempt)
                try:
                    record = produce(problem, attempt, sample, seed, identity)
                    record.update(
                        episode_id=identity,
                        problem_id=pid,
                        split=split,
                        sample_index=sample,
                        attempt_index=attempt,
                        seed=seed,
                        config=settings,
                        config_hash=config_hash,
                    )
                    validate_record(record)
                    line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                    with mutex:
                        offset = stream.tell()
                        stream.write(line)
                        stream.flush()
                        os.fsync(stream.fileno())
                        register(record, offset, hashlib.sha256(line).hexdigest())
                    accepted, reason, length = qualification(record, limit)
                    print(
                        json.dumps(
                            {
                                "episode_id": identity,
                                "qualified": accepted,
                                "full_tokens": length,
                                "reason": reason,
                                "accepted": len(row["accepted"]),
                                "target": quotas[pid],
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    with mutex:
                        errors.append(
                            {"problem_id": pid, "reason": "infrastructure_or_generation_error", "error": repr(error)}
                        )
                        stop.set()
                    return

        if settings["workers"] == 1:
            list(map(work, problems))
        else:
            with ThreadPoolExecutor(settings["workers"]) as executor:
                list(executor.map(work, problems))
    accepted_items = sorted(
        (item for row in state.values() for item in row["accepted"]), key=lambda item: item["episode_id"]
    )
    accepted_path = output / "accepted.jsonl"
    temp = output / "accepted.jsonl.tmp"
    with raw_path.open("rb") as source, temp.open("wb") as destination:
        for item in accepted_items:
            source.seek(item["offset"])
            line = source.readline()
            if hashlib.sha256(line).hexdigest() != item["sha256"]:
                raise ValueError("Raw episode changed during accepted export")
            destination.write(line)
        destination.flush()
        os.fsync(destination.fileno())
    temp.replace(accepted_path)
    complete = not errors and all(
        row["attempted"] == settings["pilot_per_problem"] if pilot else len(row["accepted"]) == quotas[pid]
        for pid, row in state.items()
    )
    ids = [item["episode_id"] for item in accepted_items]
    report = {
        "schema_version": 1,
        "status": ("pilot_complete" if pilot else "complete") if complete else "stopped",
        "manifest_hash": manifest["manifest_hash"],
        "config_hash": config_hash,
        "config": settings,
        "split": split,
        "shard_index": settings["shard_index"],
        "num_shards": settings["num_shards"],
        "target_per_problem": settings.get("target_per_problem"),
        "quota_by_problem": quotas,
        "max_full_tokens_exclusive": limit,
        "raw": {"path": str(raw_path), "sha256": file_hash(raw_path), "episodes": len(seen)},
        "accepted": {
            "path": str(accepted_path),
            "sha256": file_hash(accepted_path),
            "episodes": len(ids),
            "episode_ids": ids,
            "episode_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        },
        "problems": {
            pid: {
                "attempted": row["attempted"],
                "accepted": len(row["accepted"]),
                "rejected": len(row["rejected"]),
                "acceptance_rate": len(row["accepted"]) / row["attempted"] if row["attempted"] else None,
                "accepted_ids": [item["episode_id"] for item in row["accepted"]],
                "rejected_ids": [item["episode_id"] for item in row["rejected"]],
                "rejection_reasons": row["reasons"],
                "lengths": [
                    {"episode_id": item["episode_id"], "full_tokens": item["full_tokens"]}
                    for item in row["accepted"] + row["rejected"]
                ],
            }
            for pid, row in state.items()
        },
        "stop_reason": errors,
        "acceptance_rate": len(ids) / len(seen) if seen else None,
    }
    atomic_json(output / "collection.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=list(SPLIT_INDEX), required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--model", default="HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1")
    p.add_argument("--model-revision", default="016efe4f9b8d1f472b412a26bd43e9b3f690a223")
    p.add_argument("--backend", choices=["sglang", "transformers"], default="sglang")
    p.add_argument("--actor-url", default="http://127.0.0.1:8000")
    p.add_argument("--served-model-name", default="qwen35-9b")
    p.add_argument("--request-timeout", type=int, default=3600)
    p.add_argument("--judge", choices=["http", "local"], required=True)
    p.add_argument("--judge-url")
    p.add_argument("--testlib-dir")
    p.add_argument("--judge-cache", default="artifacts/judge-cache")
    p.add_argument("--allow-local-execution", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--target-per-problem", type=int)
    p.add_argument("--problem-limit", type=int)
    p.add_argument("--max-full-tokens-exclusive", type=int, default=120000)
    p.add_argument("--min-acceptance-rate", type=float, default=0.25)
    p.add_argument("--pilot-attempts", type=int, default=32)
    p.add_argument("--max-attempts-per-problem", type=int, default=1024)
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--pilot-per-problem", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--context-length", type=int, default=131072)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-submissions", type=int, default=6)
    p.add_argument("--max-response-tokens", type=int, default=24576)
    p.add_argument("--max-summary-tokens", type=int, default=24576)
    p.add_argument("--max-episode-response-tokens", type=int)
    p.add_argument("--max-full-tokens", type=int, default=2000000)
    p.add_argument("--compact-threshold-tokens", type=int, default=12000)
    p.add_argument("--disable-thinking", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("Invalid workers or shard selection")
    if args.workers > 1 and (args.judge == "local" or args.backend == "transformers"):
        p.error("Local judging and transformers require workers=1")
    if not 0 < args.min_acceptance_rate <= 1 or not 1 <= args.pilot_attempts <= args.max_attempts_per_problem <= 4096:
        p.error("Invalid acceptance-rate or attempt limits")
    if any(
        getattr(args, key) < 1
        for key in (
            "max_full_tokens_exclusive",
            "context_length",
            "max_turns",
            "max_submissions",
            "max_response_tokens",
            "max_summary_tokens",
            "max_full_tokens",
            "pilot_per_problem",
        )
    ):
        p.error("Token and turn limits must be positive")
    if args.max_turns > 32 or args.pilot_per_problem > 4096:
        p.error("Candidate seed blocks support up to 32 action turns and 4096 attempts")
    if args.problem_limit is not None and args.problem_limit < 1:
        p.error("Problem limit must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.compact_threshold_tokens < 0:
        p.error("Invalid sampling or compaction parameters")
    args.max_episode_response_tokens = args.max_episode_response_tokens or args.max_full_tokens_exclusive
    if (
        args.max_episode_response_tokens < args.max_full_tokens_exclusive
        or args.max_full_tokens <= args.max_full_tokens_exclusive
    ):
        p.error("Response and raw safety budgets must not prematurely censor potentially eligible episodes")
    if args.max_full_tokens_exclusive > 120000:
        p.error("This production recipe supports complete inputs below 120000 tokens")
    manifest = json.loads(Path(args.manifest).read_text())
    unhashed = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(unhashed) != manifest["manifest_hash"]:
        raise ValueError("Manifest hash mismatch")
    if args.target_per_problem is not None and args.target_per_problem < 1:
        p.error("Target quota must be positive")
    quotas = collection_quotas(manifest, args.split)
    if args.split == "train" and args.target_per_problem is None:
        if sum(quotas.values()) != 5120 or any(value % 16 for value in quotas.values()):
            p.error(
                "The training manifest must plan exactly 5120 trajectories in 16-episode groups"
            )
    if args.target_per_problem is not None:
        quotas = {problem: args.target_per_problem for problem in quotas}
    elif len(set(quotas.values())) == 1:
        args.target_per_problem = next(iter(quotas.values()))
    for problem in manifest["splits"][args.split][: args.problem_limit]:
        verify_problem(problem)
    settings = {
        k: v
        for k, v in vars(args).items()
        if k not in ("output_root", "manifest", "actor_url", "judge_url", "judge_cache")
    }
    from agentic_critic_runtime import fcs_env

    settings["manifest_hash"] = manifest["manifest_hash"]
    settings["quota_by_problem"] = quotas
    settings["episode_runner"] = "agentic_critic_runtime.fcs_env.run_episode"
    settings["episode_runner_sha256"] = file_hash(fcs_env.__file__)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    fmt = turn_format(tokenizer, not args.disable_thinking)
    actor = HTTPActorSGLang(args, tokenizer) if args.backend == "sglang" else TransformersActor(args, tokenizer)
    if args.judge == "local":
        if not args.allow_local_execution or not args.testlib_dir:
            p.error("Local judge requires --allow-local-execution and --testlib-dir")
        judge = LocalJudge(args.judge_cache, args.testlib_dir)
    else:
        if not args.judge_url:
            p.error("HTTP judge requires --judge-url")
        judge = HTTPJudge(args.judge_url)

    def produce(problem, attempt, sample, seed, identity):
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
            raise ValueError("Actor produced no trainable tokens")
        return {
            "schema_version": 1,
            "source_revision": problem["source_revision"],
            "model": args.model,
            "tokenizer": args.model,
            "elapsed_seconds": time.monotonic() - started,
            **episode,
        }

    report = collect(manifest, settings, args.output_root, produce)
    print(
        json.dumps(
            {
                "status": report["status"],
                "accepted": report["accepted"]["episodes"],
                "attempted": report["raw"]["episodes"],
                "acceptance_rate": report["acceptance_rate"],
            }
        )
    )
    if report["status"] == "stopped":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
