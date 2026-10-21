"""Admit a complete rollout split: never hide failed jobs or duplicate attempts."""

import argparse
import hashlib
import json
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from .collect import candidate, qualification
from .data import collection_quotas, digest
from .records import validate_episode
from .rollout import to_segments


def validate_trajectory_record(record):
    """Check reward provenance and recorded-token alignment, including rejected attempts."""
    validate_episode(record)
    if to_segments(record["events"]) != record["segments"]:
        raise ValueError("Trajectory segments or masks differ from original event tokens")
    scores = record["scores"]
    if any(
        not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100 for score in scores
    ):
        raise ValueError("Submission scores must be finite and within [0,100]")
    if scores != [submission["score"] for submission in record["submissions"]]:
        raise ValueError("Scores differ from recorded judge submissions")
    if not 0 <= record["reward"] <= 1 or abs(record["reward"] - max(scores, default=0) / 100) > 1e-8:
        raise ValueError("Reward does not equal maximum observed submission score / 100")


def verify_collection_ledgers(manifest, manifest_hash, split, ledger_paths):
    """Verify every raw attempt and qualified ID before admitting a complete split."""
    expected_problems = [p["problem_id"] for p in manifest["splits"][split]]
    quotas = collection_quotas(manifest, split)
    ledgers = [json.loads(Path(path).read_text()) for path in ledger_paths]
    if not ledgers:
        raise ValueError("No collection ledgers")
    shard_counts = {ledger["num_shards"] for ledger in ledgers}
    if len(shard_counts) != 1:
        raise ValueError("Collection shard counts differ")
    shard_count = shard_counts.pop()
    if not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("Collection shard count must be positive")
    if len(ledgers) != shard_count or {r["shard_index"] for r in ledgers} != set(range(shard_count)):
        raise ValueError("Missing or duplicate collection shards")
    all_ids, configs, limits, accepted_paths = set(), set(), set(), []
    attempted = rejected = 0
    accepted_file_hashes = {}
    per_problem = {}
    for ledger in ledgers:
        if ledger["status"] != "complete" or ledger["split"] != split:
            raise ValueError("Only complete production collections can be merged")
        ledger_quotas = ledger.get("quota_by_problem") or {
            problem: ledger["target_per_problem"] for problem in expected_problems
        }
        uniform = next(iter(quotas.values())) if len(set(quotas.values())) == 1 else None
        if (
            ledger["manifest_hash"] != manifest_hash
            or ledger_quotas != quotas
            or ledger["target_per_problem"] != uniform
        ):
            raise ValueError("Collection manifest/quota mismatch")
        owned = {p for i, p in enumerate(expected_problems) if i % shard_count == ledger["shard_index"]}
        if set(ledger["problems"]) != owned:
            raise ValueError("Collection problem coverage mismatch")
        config = ledger["config"]
        if config.get("pilot_only") or config.get("manifest_hash") != manifest_hash:
            raise ValueError("Collection is a pilot or belongs to a different manifest")
        for field in ("split", "shard_index", "num_shards", "target_per_problem", "max_full_tokens_exclusive"):
            if config.get(field) != ledger[field]:
                raise ValueError(f"Collection configuration disagrees with ledger {field}")
        if "quota_by_problem" in config and config["quota_by_problem"] != quotas:
            raise ValueError("Collection configuration quota map disagrees with manifest")
        computed_config = digest(
            {k: v for k, v in config.items() if k not in ("workers", "shard_index", "num_shards", "problem_limit")}
        )
        if computed_config != ledger["config_hash"]:
            raise ValueError("Collection configuration hash mismatch")
        configs.add(ledger["config_hash"])
        limit = ledger["max_full_tokens_exclusive"]
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Collection exclusive length limit must be positive")
        limits.add(limit)
        accepted_paths.append(ledger["accepted"]["path"])
        seen, selected = set(), set()
        counts = {problem: Counter() for problem in owned}
        accepted_ids = {problem: [] for problem in owned}
        rejected_ids = {problem: [] for problem in owned}
        reasons = {problem: Counter() for problem in owned}
        lengths = {problem: {} for problem in owned}
        selected_line_hashes = {}
        raw_hash = hashlib.sha256()
        with Path(ledger["raw"]["path"]).open("rb") as stream:
            for line in stream:
                raw_hash.update(line)
                episode = json.loads(line)
                validate_trajectory_record(episode)
                identity, problem = episode["episode_id"], episode["problem_id"]
                if problem not in owned or identity in seen or identity in all_ids or episode["split"] != split:
                    raise ValueError("Invalid raw collection identity or problem")
                if episode["config_hash"] != ledger["config_hash"] or episode["config"] != config:
                    raise ValueError("Raw collection configuration mismatch")
                if counts[problem]["accepted"] >= quotas[problem]:
                    raise ValueError("Raw collection continued after reaching its qualified quota")
                sample = episode["sample_index"]
                ordinal = {"train": 0, "validation": 1, "test": 2}[split]
                if (
                    sample % 3 != ordinal
                    or sample // 3 != counts[problem]["attempted"]
                    or episode["attempt_index"] != counts[problem]["attempted"]
                ):
                    raise ValueError("Collection attempt sequence or split namespace mismatch")
                expected_sample, seed, expected_identity = candidate(
                    manifest, split, problem, counts[problem]["attempted"]
                )
                if sample != expected_sample or identity != expected_identity or episode["seed"] != seed:
                    raise ValueError("Collection identity/seed mismatch")
                seen.add(identity)
                counts[problem]["attempted"] += 1
                eligible, reason, length = qualification(episode, limit)
                counts[problem]["accepted" if eligible else "rejected"] += 1
                lengths[problem][identity] = length
                (accepted_ids if eligible else rejected_ids)[problem].append(identity)
                if reason:
                    reasons[problem][reason] += 1
                if eligible:
                    selected.add(identity)
                    selected_line_hashes[identity] = hashlib.sha256(line).digest()
        if raw_hash.hexdigest() != ledger["raw"]["sha256"]:
            raise ValueError("Collection raw checksum mismatch")
        accepted_hash, observed_accepted = hashlib.sha256(), set()
        with Path(ledger["accepted"]["path"]).open("rb") as stream:
            for line in stream:
                accepted_hash.update(line)
                record = json.loads(line)
                identity = record["episode_id"]
                if identity not in selected or identity in observed_accepted:
                    raise ValueError("Accepted collection contains an unexpected or duplicate episode")
                if hashlib.sha256(line).digest() != selected_line_hashes[identity]:
                    raise ValueError("Accepted episode differs from its original raw record")
                observed_accepted.add(identity)
        if observed_accepted != selected:
            raise ValueError("Accepted file does not contain every raw-qualified episode")
        if accepted_hash.hexdigest() != ledger["accepted"]["sha256"]:
            raise ValueError("Collection accepted checksum mismatch")
        accepted_file_hashes[ledger["accepted"]["path"]] = accepted_hash.hexdigest()
        if len(seen) != ledger["raw"]["episodes"] or len(selected) != ledger["accepted"]["episodes"]:
            raise ValueError("Collection raw/accepted count mismatch")
        if sorted(selected) != ledger["accepted"]["episode_ids"]:
            raise ValueError("Collection accepted IDs disagree with raw eligibility")
        ids_hash = hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()
        if ids_hash != ledger["accepted"]["episode_ids_sha256"]:
            raise ValueError("Collection accepted ID digest mismatch")
        for problem, actual in counts.items():
            declared = ledger["problems"][problem]
            if actual["accepted"] != quotas[problem]:
                raise ValueError("Insufficient or excess qualified trajectories for a problem")
            if any(actual[k] != declared[k] for k in ("attempted", "accepted", "rejected")):
                raise ValueError("Per-problem collection count mismatch")
            if declared["accepted_ids"] != accepted_ids[problem] or declared["rejected_ids"] != rejected_ids[problem]:
                raise ValueError("Per-problem accepted/rejected IDs disagree with raw attempts")
            declared_lengths = {row["episode_id"]: row["full_tokens"] for row in declared["lengths"]}
            if (
                declared_lengths != lengths[problem]
                or len(declared["lengths"]) != actual["attempted"]
                or declared["rejection_reasons"] != dict(reasons[problem])
            ):
                raise ValueError("Per-problem rejected lengths or reasons disagree with raw attempts")
            rate = actual["accepted"] / actual["attempted"]
            if abs(declared["acceptance_rate"] - rate) > 1e-12:
                raise ValueError("Per-problem acceptance rate mismatch")
            per_problem[problem] = dict(actual, acceptance_rate=rate)
        if ledger["stop_reason"]:
            raise ValueError("Complete collection still declares stopped attempts")
        if not seen or not math.isclose(ledger["acceptance_rate"], len(selected) / len(seen), abs_tol=1e-12):
            raise ValueError("Collection acceptance rate mismatch")
        all_ids.update(selected)
        attempted += len(seen)
        rejected += len(seen) - len(selected)
    if len(configs) != 1 or len(limits) != 1:
        raise ValueError("Collection configurations or length limits differ")
    return (
        accepted_paths,
        all_ids,
        dict(
            raw_attempts=attempted,
            rejected=rejected,
            qualified=len(all_ids),
            acceptance_rate=len(all_ids) / attempted,
            max_full_tokens_exclusive=limits.pop(),
            per_problem=per_problem,
            ledger_paths=[str(p) for p in ledger_paths],
            accepted_file_sha256=accepted_file_hashes,
        ),
    )


def merge(manifest_path, split, shards, output, allow_partial=False, collection_ledgers=None):
    manifest = json.loads(Path(manifest_path).read_text())
    manifest_hash = manifest.pop("manifest_hash")
    if digest(manifest) != manifest_hash:
        raise ValueError("Manifest hash mismatch")
    sampling = manifest["sampling"][split]
    start = sampling["sample_start"]
    quotas = collection_quotas(manifest, split)
    expected = {
        f"{p['problem_id']}:{split}:{i}"
        for p in manifest["splits"][split]
        for i in range(start, start + quotas[p["problem_id"]])
    }
    if "quota_by_problem" in sampling and not collection_ledgers:
        raise ValueError("Variable-quota collection requires complete collection ledgers")
    collection_summary = None
    if collection_ledgers:
        if shards or allow_partial:
            raise ValueError("Collection-ledger merge cannot combine legacy shards or partial smoke mode")
        shards, expected, collection_summary = verify_collection_ledgers(
            manifest, manifest_hash, split, collection_ledgers
        )
    records, config_hashes = {}, set()
    lengths, problems, rewards = [], set(), []
    compactions, finish_reasons = Counter(), Counter()
    truncations = 0
    call_lengths = {"action": [], "summary": []}
    call_truncations = Counter()
    unclosed_summaries = submission_episodes = 0
    # Retain offsets, never every raw event/token list for a production split.
    for path in shards:
        shard_hash = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                shard_hash.update(line)
                r = json.loads(line)
                validate_trajectory_record(r)
                identity = r["episode_id"]
                if identity not in expected or identity in records or r["split"] != split:
                    raise ValueError(f"Unexpected or duplicated episode: {identity}")
                if r["config"]["manifest_hash"] != manifest_hash:
                    raise ValueError("Rollout belongs to a different frozen manifest")
                semantic_config = (
                    r["config"]
                    if collection_summary is None
                    else {
                        key: value
                        for key, value in r["config"].items()
                        if key not in ("workers", "shard_index", "num_shards", "problem_limit")
                    }
                )
                if digest(semantic_config) != r["config_hash"]:
                    raise ValueError("Rollout configuration hash mismatch")
                records[identity] = (str(path), offset, hashlib.sha256(line).digest())
                config_hashes.add(r["config_hash"])
                lengths.append(sum(len(s["input_ids"]) for s in r["segments"]))
                problems.add(r["problem_id"])
                rewards.append(r["reward"])
                compactions[r["n_compactions"]] += 1
                finish_reasons[r["finish_reason"]] += 1
                truncations += r["truncated_turns"]
                submission_episodes += bool(r["scores"])
                for event in r["events"]:
                    kind = event["kind"]
                    if event["actor"] and kind in call_lengths:
                        call_lengths[kind].append(len(event["tokens"]))
                        call_truncations[kind] += event["finish_reason"] == "length"
                        if kind == "summary":
                            unclosed_summaries += event.get("thinking_complete") is False
        if (
            collection_summary is not None
            and shard_hash.hexdigest() != collection_summary["accepted_file_sha256"][str(path)]
        ):
            raise ValueError("Accepted collection changed after ledger verification")
    if len(config_hashes) != 1:
        raise ValueError("All shards must have the same semantic rollout configuration")
    missing = sorted(expected - records.keys())
    if missing and not allow_partial:
        raise ValueError(f"Incomplete split: {len(missing)} missing episodes; first missing={missing[:5]}")
    lengths.sort()
    call_stats = {}
    for kind, values in call_lengths.items():
        values.sort()
        call_stats[kind] = {
            "count": len(values),
            "length_limited": call_truncations[kind],
            "length_limited_fraction": call_truncations[kind] / len(values) if values else None,
            "length": {
                name: values[round(q * (len(values) - 1))]
                for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1))
            }
            if values
            else {},
        }
    summary = {
        "split": split,
        "expected": len(expected),
        "actual": len(records),
        "missing": len(missing),
        "partial_smoke_only": allow_partial,
        "config_hash": next(iter(config_hashes)),
        "problem_count": len(problems),
        "full_length": {
            name: lengths[round(q * (len(lengths) - 1))]
            for name, q in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1)]
        },
        "reward_mean": sum(rewards) / len(rewards),
        "compactions": dict(compactions),
        "finish_reasons": dict(finish_reasons),
        "response_length_truncations": truncations,
        "calls": call_stats,
        "summaries_without_closed_thinking": unclosed_summaries,
        "episodes_with_submission": submission_episodes,
        "submission_episode_fraction": submission_episodes / len(records),
        "episode_ids_sha256": hashlib.sha256("\n".join(sorted(records)).encode()).hexdigest(),
    }
    if collection_summary is not None:
        summary["collection"] = collection_summary
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Merged output already exists: {out}; choose a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    with ExitStack() as stack:
        inputs = {str(p): stack.enter_context(Path(p).open("rb")) for p in shards}
        stream = stack.enter_context(temp.open("wb"))
        for identity in sorted(records):
            path, offset, checksum = records[identity]
            inputs[path].seek(offset)
            line = inputs[path].readline()
            if hashlib.sha256(line).digest() != checksum:
                raise ValueError("Rollout shard changed during merge")
            stream.write(line.rstrip(b"\n") + b"\n")
    temp.replace(out)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], required=True)
    p.add_argument("--shards", nargs="+", default=[])
    p.add_argument("--collection-ledgers", nargs="+", help="Complete quota-collection ledgers")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-partial-smoke", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            merge(a.manifest, a.split, a.shards, a.output, a.allow_partial_smoke, a.collection_ledgers), indent=2
        )
    )


if __name__ == "__main__":
    main()
