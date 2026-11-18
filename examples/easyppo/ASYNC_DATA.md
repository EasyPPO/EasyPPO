# Terminal actor-critic datasets

The asynchronous and synchronous recipes use the same bash/mini-swe-agent harness and dataset pairs:

| Recipe | Training corpus | Evaluation corpus | Prepared inputs |
| --- | --- | --- | --- |
| FrontierSmith | All 200 released synthetic problems | All 180 Frontier-CS algorithmic problems | `frontiersmith.train.jsonl`, `fcs.eval.jsonl` |
| Terminal | Eligible TMAX-15K Harbor tasks | All 89 Terminal-Bench 2.1 tasks | `tmax.train.jsonl`, `tb21.eval.jsonl` |

The source release explicitly identifies 10 synthetic parity problems as
aliases of Frontier-CS IDs 306–315. The preparation script retains these
10 problems in training and records the overlap in `known_train_eval_aliases`.
The training and evaluation corpora are therefore not completely disjoint.
[Alias manifest](https://github.com/FrontierCS/FrontierSmith/blob/ac3046cce7a270e3177cbb41376f36e03de1d518/data/sample_lists/frontiersmith_parity_10.json).

The fixed Frontier-CS version contains 180 algorithmic directories, all included
in `fcs.eval.jsonl`. Problem 61 is included for newly generated solutions. The
upstream instruction to skip problem 61 concerns rescoring old solutions after
an output-format change; it does not exclude the current problem from evaluation.
[Pinned evaluation source](https://github.com/FrontierCS/Frontier-CS/tree/59687e5428266a22a49cf51f981b912103bfa91d).

Use Python 3.12+, Docker Engine, Compose v2, and this checkout's terminal extra:

```bash
python -m pip install -e '.[terminal]'
python scripts/make_miles_frontiersmith_data.py \
  --judge-url http://10.0.0.20:8081 \
  --tasks-dir /data/harbor --out /data/prepared
python scripts/make_miles_terminal_data.py \
  --dataset tmax --tasks-dir /data/harbor --out /data/prepared
```

The embedded `Frontier-CS/` directory combines files from these fixed commits:

| Source | Commit |
| --- | --- |
| FrontierSmith training data and Harbor adapter | `ac3046cce7a270e3177cbb41376f36e03de1d518` |
| Frontier-CS evaluation data, checkers and judge | `59687e5428266a22a49cf51f981b912103bfa91d` |

The adapter is bundled under `Frontier-CS/harbor/`; no source download or separate
adapter installation is needed. `--source-dir` can select another copy of this
combined directory; the 27B launcher exposes it as `--frontier-source-dir`. Base requirements supply PyYAML, and the
terminal extra pins Harbor 0.24.0 and mini-swe-agent 1.17.1.

The FrontierSmith command writes the two recipe inputs and
`frontiersmith.manifest.json`. The manifest records both source revisions,
training/evaluation IDs and the ten retained cross-corpus aliases. Rows include `task_path` relative to `/data/harbor`,
dataset, canonical `problem_id`, split, source URL/revision, and reward key.
Duplicate IDs and incomplete task collections cause preparation to fail.
`--eval-count N --seed 42` optionally reserves an additional synthetic holdout
in `frontiersmith.eval.jsonl`; the default is zero because Frontier-CS is the
recipe's evaluation corpus. `--overwrite` regenerates upstream task artifacts.

The TMAX command uses the existing fixed Harbor dataset references, removes
exact prompt/fixture overlaps with TB2.1 and duplicate executable fixtures,
and filters tasks whose initial instruction cannot fit the action budget.
Thus the eligible training count can be below the source's nominal 15K.
The complete 89-task TB2.1 file remains evaluation input.

The Frontier agent reads the prepared statement and submits C++ solutions to
the shared judge. The synchronous recipe sends requests from each rollout
worker. The asynchronous recipe defaults to a Harbor submission container.
Both use the highest bounded submission score divided by 100 as the episode
reward. Judge error verdicts, HTTP errors, polling timeouts and invalid scores
count as zero for the affected submission. `judge-result.json` records
the verdicts and `reward.json` records the verified numeric reward.

Set `--frontier-judge-timeout` during preparation to cover judging and queue
wait. Reprepare when changing that timeout. Docker-based trials also use the
judge URL embedded during preparation. Direct submissions use the launch-time
`--frontier-judge-url`. See the [judge deployment instructions](../../README.md#4-run-the-task-environments-and-judges)
for endpoint and reachability settings. TMAX and Terminal-Bench use their task
verifiers and rewards.
