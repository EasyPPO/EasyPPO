"""Turn stored agentic Frontier-CS episodes into rollout files that Miles can train from without generating.

During critic-only training the actor is frozen, so rollouts do not depend on the critic and one stored pool can
be replayed for every critic setting (`--load-debug-rollout-data DIR/rollout_{rollout_id}.pt --debug-train-only`).

Each rollout file is one training step of exactly problems-per-step x episodes-per-problem episodes (Miles counts a
step in episodes), made of whole problem groups of at least two episodes, so that the noise-normalised critic
weights have a group to normalise over. Problems are drawn in a seeded random order; running out of problems
is an error unless --allow-reuse is given (then they are reused in a new order). The first `--heldout-problems` problems of a seeded shuffle never appear
in a rollout file; their episodes are written to heldout.jsonl for evaluation.

Every sample is one actor segment, exactly as agentic_critic.miles_fcs.generate produces it during a live
rollout: the episode's samples share `rollout_id` and the reward and carry the episode-level critic sample.

Needs Miles importable: run with the Miles environment and PYTHONPATH=<this repo>:<miles>.
Usage: python scripts/miles/episodes_to_rollouts.py --episodes FILE.jsonl [...] --out DIR --steps 30
"""

import argparse
import json
import random
from pathlib import Path

import torch

from agentic_critic.fcs_env import Episode, Event
from agentic_critic.miles_fcs import episode_samples
from miles.utils.types import Sample

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--problems-per-step", type=int, default=8)
ap.add_argument("--episodes-per-problem", type=int, default=4)
ap.add_argument("--heldout-problems", type=int, default=5)
ap.add_argument("--max-episode-tokens", type=int, default=None, help="leave out longer episodes (a whole episode is one critic sample in the Full setting)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--first-step", type=int, default=0, help="number of the first rollout file; for a second batch that continues a run")
ap.add_argument("--allow-reuse", action="store_true", help="let a problem's episodes be used in more than one step")
ap.add_argument("--pack", action="store_true", help="fill a step with the next problem (in the shuffled order) that still fits; "
                "when none fits, cut a group and use the rest of it in a later step instead of dropping it")
ap.add_argument("--steps-max", action="store_true", help="write as many steps as the episodes allow, at most --steps")
ap.add_argument("--rounds", action="store_true", help="many episodes per problem: use each problem several times, --episodes-per-problem "
                "episodes at a time, going through all problems once before any problem comes again")
args = ap.parse_args()

by_problem: dict[str, list[dict]] = {}
seen = too_long = 0
for path in args.episodes:
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        seen += 1
        if args.max_episode_tokens and r["full_tokens"] > args.max_episode_tokens:
            too_long += 1
            continue
        by_problem.setdefault(r["problem"], []).append(r)
# the critic's weights are computed per problem group, so a problem needs at least two episodes
complete = sorted(p for p, eps in by_problem.items() if len(eps) >= 2)
rng = random.Random(args.seed)
rng.shuffle(complete)
heldout, train = complete[: args.heldout_problems], complete[args.heldout_problems :]
if args.rounds:
    # A pool with many episodes per problem: cut each problem's episodes into groups of --episodes-per-problem and
    # go through the problems round by round (every problem's first group, in a fresh random order; then every
    # second group; ...). In the first round each step holds problems the critic has not seen; an episode is used once.
    rounds: dict[int, list[str]] = {}
    for problem in train:
        eps = by_problem.pop(problem)
        for k in range(0, len(eps), args.episodes_per_problem):
            chunk = eps[k : k + args.episodes_per_problem]
            if len(chunk) >= 2:
                by_problem[f"{problem}@{k // args.episodes_per_problem}"] = chunk
                rounds.setdefault(k // args.episodes_per_problem, []).append(f"{problem}@{k // args.episodes_per_problem}")
    train = []
    for k in sorted(rounds):
        rng.shuffle(rounds[k])
        train += rounds[k]
    round_sizes = [len(rounds[k]) for k in sorted(rounds)]
assert len(train) >= args.problems_per_step, f"only {len(train)} training problems with at least two episodes"

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "heldout.jsonl", "w") as f:
    for p in heldout:
        for r in by_problem[p][: args.episodes_per_problem]:
            f.write(json.dumps(r) + "\n")


def samples_of(record: dict, group_index: int, rollout_id: int, first_index: int) -> list[Sample]:
    ep = Episode(events=[Event(**e) for e in record["events"]], scores=record["scores"],
                 submission_segments=record["submission_segments"], n_compactions=record["n_compactions"])  # fmt: skip
    n_calls = sum(e.actor for e in ep.events)
    fields = episode_samples(ep, [[0.0] * len(e.tokens) for e in ep.events if e.actor])
    assert n_calls and fields, "an episode without actor tokens cannot be trained on"
    out_samples = []
    for k, f in enumerate(fields):
        s = Sample(
            index=first_index + k, group_index=group_index, rollout_id=rollout_id, prompt="", tokens=f["tokens"],
            response_length=f["response_length"], loss_mask=f["loss_mask"], reward=ep.reward,
            status=Sample.Status.TRUNCATED if f["truncated"] else Sample.Status.COMPLETED,
            metadata={"problem": record["problem"], "sample": record["sample"], "segment": f["segment"], "n_compactions": ep.n_compactions},
        )  # fmt: skip
        # log-probabilities of the stored rollouts were not recorded; the critic does not use them
        s.rollout_log_probs = None
        for key in ("critic_tokens", "critic_response_length", "critic_loss_mask", "critic_span"):
            setattr(s, key, f[key])
        s.validate()
        out_samples.append(s)
    return out_samples


target = args.problems_per_step * args.episodes_per_problem  # episodes per step; Miles counts a step in episodes
# ids of a second batch continue after the first one's
order, manifest, rollout_counter, sample_counter = [], [], args.first_step * target, args.first_step * 1_000_000
for step in range(args.steps):
    samples, chosen, n_episodes, group = [], [], 0, 0
    while n_episodes < target:
        if not order and args.steps_max and step > 0 and not args.allow_reuse:
            break  # the episodes are used up: the step that was being filled is dropped
        if not order:
            # explained variance is read on each step before the critic trains on it, so an episode is used once
            assert args.allow_reuse or step == 0, f"training problems ran out at step {step}; fewer steps, or --allow-reuse"
            order = train[:]
            if not args.rounds:  # with rounds the order is already random inside each round and must stay round by round
                rng.shuffle(order)
        room = target - n_episodes
        fits = next((i for i, q in enumerate(order) if len(by_problem[q][: args.episodes_per_problem]) <= room), 0) if args.pack else 0
        problem = order.pop(fits)
        records = by_problem[problem][: args.episodes_per_problem]
        if len(records) > room:  # the step must hold exactly `target` episodes
            if args.pack:  # the rest of the group is used in a later step, as a smaller group of the same problem
                by_problem[problem + "#rest"] = records[room:]
                order.append(problem + "#rest")
            records = records[:room]  # (a lone last episode forms a group of one; its weight falls back to the floor)
        chosen.append(problem)
        for record in records:
            new = samples_of(record, group_index=(args.first_step + step) * target + group, rollout_id=rollout_counter, first_index=sample_counter)
            samples += new
            rollout_counter += 1
            sample_counter += len(new)
        n_episodes += len(records)
        group += 1
    if n_episodes < target:
        break
    torch.save({"rollout_id": args.first_step + step, "metadata": {}, "samples": [s.to_dict() for s in samples]}, out / f"rollout_{args.first_step + step}.pt")
    manifest.append({"step": args.first_step + step, "problems": chosen, "episodes": len({s.rollout_id for s in samples}), "samples": len(samples),
                     "actor_tokens": sum(len(s.tokens) for s in samples),
                     "critic_tokens_full": sum(len(s.critic_tokens) for s in samples if s.critic_span[0] == 0)})  # fmt: skip
(out / "manifest.json").write_text(json.dumps({"rounds": round_sizes if args.rounds else None, "args": vars(args), "episodes_read": seen, "episodes_too_long": too_long, "train_problems": train, "heldout_problems": heldout, "steps": manifest}, indent=1))
print(f"{seen} episodes read, {too_long} left out as longer than --max-episode-tokens")
print(f"{len(train)} training problems, {len(heldout)} held out; {len(manifest)} steps written to {out}")
print("first step:", {k: v for k, v in manifest[0].items() if k != "problems"})
