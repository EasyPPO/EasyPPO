# Terminal actor-critic datasets

The asynchronous and synchronous recipes use the same bash/mini-swe-agent harness and dataset pairs:

| Recipe | Training corpus | Evaluation corpus | Prepared inputs |
| --- | --- | --- | --- |
| FrontierSmith | 190 of the 200 released synthetic problems | 179 Frontier-CS algorithmic problems | `frontiersmith.train.jsonl`, `fcs.eval.jsonl` |
| Terminal | Eligible TMAX-15K Harbor tasks | All 89 Terminal-Bench 2.1 tasks | `tmax.train.jsonl`, `tb21.eval.jsonl` |

The source release explicitly identifies 10 synthetic parity problems as
aliases of Frontier-CS IDs 306–315. The preparation script removes these
10 problems **from training**, preserving the full selected Frontier-CS
evaluation set.
[Alias manifest](https://github.com/FrontierCS/FrontierSmith/blob/ac3046cce7a270e3177cbb41376f36e03de1d518/data/sample_lists/frontiersmith_parity_10.json).

The fixed Frontier-CS version contains 180 algorithmic directories. Its
published scoring comparison uses 179 problems, excluding ID 61 after an
output-format change. `fcs.eval.jsonl` follows that 179-problem set, rather
than the historical 172 count in older FrontierSmith documentation.
[Pinned evaluation source](https://github.com/FrontierCS/Frontier-CS/tree/59687e5428266a22a49cf51f981b912103bfa91d).

Use Python 3.12+, Docker Engine, Compose v2, and this checkout's terminal extra:

```bash
python -m pip install -e '.[terminal]'
python scripts/make_miles_frontiersmith_data.py \
  --source-dir /data/FrontierSmith \
  --frontier-cs-source-dir /data/Frontier-CS-eval \
  --judge-url http://10.0.0.20:8081 \
  --tasks-dir /data/harbor --out /data/prepared
python scripts/make_miles_terminal_data.py \
  --dataset tmax --tasks-dir /data/harbor --out /data/prepared
```

Missing source checkouts are cloned at these fixed commits:

| Source | Commit |
| --- | --- |
| FrontierSmith training data and Harbor adapter | `ac3046cce7a270e3177cbb41376f36e03de1d518` |
| Frontier-CS evaluation data, checkers and judge | `59687e5428266a22a49cf51f981b912103bfa91d` |

Existing checkouts must have the specified HEAD and unmodified tracked files.
The adapter is imported from the validated FrontierSmith source; no separate
adapter installation is needed. Base requirements supply PyYAML, and the
terminal extra pins Harbor 0.24.0 and mini-swe-agent 1.17.1.

The FrontierSmith command writes the two recipe inputs and
`frontiersmith.manifest.json`. The manifest records both source revisions,
training/evaluation IDs, excluded aliases and reasons, and evaluation's
problem-61 exclusion. Rows include `task_path` relative to `/data/harbor`,
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

The actor reads FrontierSmith/Frontier-CS statements from `/app/statement.txt`,
writes `/app/solution.cpp`, and can compile and self-test through bash. The
same terminal harness handles both corpora; it does not invoke the separate
Frontier-CS `submit`-only wrapper. The final verifier calls one shared judge HTTP service.

Task directories must exist at the same absolute paths on every Docker worker
host. Both complete source checkouts must exist at their prepared paths on the
judge host. Preparation writes `frontier-judge.compose.yaml` beside the JSONL
files, with one read-only mount per selected problem and persistent data/submission
volumes. It uses the pinned Frontier-CS judge code for both corpora. Start it once
as described in the [README](../../README.md#4-run-the-task-environments-and-judges).
Each Harbor trial contains only `main`; its teardown cannot remove the shared service.

`--judge-url` must be reachable from every task container. The single-host default
is `http://host.docker.internal:8081`; multi-host training must supply the judge
host's private address. `--judge-timeout` defaults to 3600 seconds including queue
wait; Harbor's verifier timeout is 90 seconds longer. Rerunning preparation updates
these settings in existing tasks and removes their old sidecars without downloading
the sources again. The 27B launchers expose these preparation settings as
`--frontier-judge-url` and `--frontier-judge-timeout`.
Data preparation itself requires no Docker or GPU.

FrontierSmith and Frontier-CS preserve their pinned checkers' bounded mean
partial score divided by 100, in `[0, 1]`. This remains a continuous score;
it is not converted into a count of passed assertions. Missing/empty C++
solutions and explicit contestant compile failures score zero. Other judge
errors, HTTP failures and poll timeouts fail the trial without a reward file.
Only numeric values enter `reward.json`; full diagnostics go to
`judge-result.json`. TMAX and Terminal-Bench retain their supplied verifier
reward. These adapters do not silently replace dataset rewards with the T1
paper's dense test-count reward.
