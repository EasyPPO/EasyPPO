"""Lossless conversion from collected episodes to Miles replay records."""

import argparse
import hashlib
import json
from pathlib import Path


def export_episodes(source, destination):
    from .records import validate_episode

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    identities = []
    seen = set()
    with Path(source).open() as records, temporary.open("w") as output:
        for line in records:
            episode = json.loads(line)
            validate_episode(episode)
            identity = episode["episode_id"]
            if identity in seen:
                raise ValueError(f"Duplicate episode: {identity}")
            seen.add(identity)
            events = [
                {key: event[key] for key in ("tokens", "actor", "segment", "kind")} for event in episode["events"]
            ]
            scores = episode["scores"]
            if abs(max(scores, default=0.0) / 100.0 - episode["reward"]) > 1e-8:
                raise ValueError(f"Final-return rule disagrees for {identity}")
            submission_segments = episode.get("submission_segments")
            if submission_segments is None:
                submission_segments = [submission["segment"] for submission in episode["submissions"]]
            if len(submission_segments) != len(scores):
                raise ValueError(f"Submission segment identities disagree for {identity}")
            from agentic_critic_runtime.fcs_env import Episode, Event

            original = Episode(
                events=[Event(**event) for event in events],
                scores=scores,
                submission_segments=submission_segments,
                n_compactions=episode["n_compactions"],
            )
            reconstructed = [
                {"input_ids": segment.tokens, "response_start": segment.response_start, "loss_mask": segment.loss_mask}
                for segment in original.to_segments()
                if any(segment.loss_mask)
            ]
            if reconstructed != episode["segments"]:
                raise ValueError(f"Event reconstruction disagrees with recorded segments: {identity}")
            full_length = sum(len(event["tokens"]) for event in events)
            if episode["finish_reason"] not in {"max_turns", "submissions_exhausted"}:
                raise ValueError(f"Replay requires a naturally completed episode: {identity}")
            if full_length >= 120000:
                raise ValueError(f"Replay episode reaches the exclusive 120k token limit: {identity}")
            # The converter preserves sample verbatim in each Miles Sample.
            # Use the exact stable identity, not a potentially colliding split-local index.
            record = {
                "problem": episode["problem_id"],
                "sample": identity,
                "events": events,
                "scores": scores,
                "submission_segments": submission_segments,
                "n_compactions": episode["n_compactions"],
                "truncated_turns": episode["truncated_turns"],
                "finish_reason": episode["finish_reason"],
                "full_tokens": sum(len(event["tokens"]) for event in events),
                "reward": episode["reward"],
                "episode_id": identity,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            identities.append({"problem": record["problem"], "sample": identity, "episode_id": identity})
    if not identities:
        raise ValueError("No episodes to replay")
    temporary.replace(destination)
    source_hash = hashlib.sha256()
    with Path(source).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            source_hash.update(block)
    metadata = {"source_sha256": source_hash.hexdigest(), "episodes": identities}
    destination.with_suffix(".identities.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def export_prompts(manifest_path, identities, destination):
    """Write the Miles raw-statement dataset for the exact replay cohort."""
    from .data import digest

    manifest = json.loads(Path(manifest_path).read_text())
    declared = manifest.pop("manifest_hash")
    if digest(manifest) != declared:
        raise ValueError("Manifest checksum mismatch")
    requested = {row["problem"] for row in identities}
    problems = {row["problem_id"]: row for row in manifest["splits"]["train"]}
    if requested - problems.keys():
        raise ValueError("Replay problems are missing from the training manifest")
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as output:
        for identity in sorted(requested):
            problem = problems[identity]
            content = (Path(problem["path"]) / "statement.txt").read_bytes()
            if hashlib.sha256(content).hexdigest() != problem["statement_sha256"]:
                raise ValueError(f"Problem statement changed: {identity}")
            output.write(
                json.dumps(
                    {
                        "prompt": content.decode(),
                        "metadata": {"problem_id": identity, "judge_id": problem["judge_id"]},
                    }
                )
                + "\n"
            )
    temporary.replace(destination)
    return len(requested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--prompts-output")
    args = parser.parse_args()
    result = export_episodes(args.source, args.output)
    if bool(args.manifest) != bool(args.prompts_output):
        parser.error("--manifest and --prompts-output must be supplied together")
    if args.manifest:
        export_prompts(args.manifest, result["episodes"], args.prompts_output)
    print(json.dumps({"episodes": len(result["episodes"]), "output": args.output}))


if __name__ == "__main__":
    main()
