# Terminal actor-critic datasets

The FrontierSmith and TMAX recipes use these training and evaluation pairs:

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

## Runbook AgentPick release

Prepare the 802-task release from the TorchTitan runbook and pass its task index to the terminal launcher:

```bash
python scripts/make_miles_agentpick_data.py --published-release --tasks-dir /data/harbor
# Launcher arguments
# --tasks-dir /data/harbor --prompt-data /data/harbor/agentpick-802/train.jsonl
```

The importer verifies the pinned HF archive and retains the original release under
`agentpick-802/release/`. Harbor uses the published Dockerfiles, task resources and
verifiers. The agent inherits each task’s network policy; grading uses
`[verifier].network_mode = "public"` on the Modal single-container backend.
To import an already downloaded archive, add `--archive-dir` pointing to the
directory containing `release.tar`.

## csk4gby2 terminal training subset

List the 543 historical task IDs, corpora and splits in shuffled order:

```bash
python scripts/make_miles_agentpick_data.py --list-tasks
```

Both listing and preparation select tasks from the pinned upstream
[split_assignment.tsv](https://huggingface.co/datasets/Fzz1/tb-training-mix-agentpick/blob/6e3e6c8a01d84f06b9f18f61782170f4801d5330/metadata/split_assignment.tsv).
They use the `clean` and `calibforge_clean` splits, excluding corpus `SMITH`
and task `tw_14495`. Preparation writes all 543 tasks in filtered TSV order.
`source_index` is the zero-based position in that order. Pass `--seed N`
to reproduce a shuffled listing. The selection includes TerminalWorld
benchmark tasks, and upstream used TB2.1 for retrieval.

From the repository root, with the terminal extra installed:

```bash
python scripts/make_miles_agentpick_data.py --tasks-dir /data/harbor
```

Preparation downloads two checksum-verified archives from the pinned release.
TMAX resource limits come from the same release’s `data/clean.parquet`.
To reuse local archives, pass `--archive-dir` pointing to a directory containing
`tasks-clean.tar` and `tasks-calibforge_clean.tar`. The command creates
`agentpick-csk4gby2/` under the tasks directory and requires that output
subdirectory to be new. Each task builds its Dockerfile with the Terminus tmux layer.
Workers need Docker access to the source images.

Pass these arguments to the existing 9B terminal launcher:

```text
--tasks-dir /data/harbor --prompt-data /data/harbor/agentpick-csk4gby2/train.jsonl
```

The launcher's preparation step supplies TB2.1 evaluation data.
[Source release](https://huggingface.co/datasets/Fzz1/tb-training-mix-agentpick/tree/6e3e6c8a01d84f06b9f18f61782170f4801d5330)


For Qwen3.5-9B Terminus-2 training checks and the TB2.1 critic comparison,
follow the [Terminal Bench runbook](../../Claude.md#terminal-bench).
