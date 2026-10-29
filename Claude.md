# Experiment progress and execution order

All documentation, reports and figure text must be in English. Follow the order
below. **Data collection and full-history/shared critic training are already
DONE**, as confirmed by the user. They are context and inputs for the remaining
work, not instructions to repeat completed experiments.

| Order | Experiment or artifact | Status | Action |
| --- | --- | --- | --- |
| 1 | Original trajectory collection and frozen train/validation cohorts | DONE | Reuse existing files |
| 2 | Full-history and actor-summary/shared critic training | DONE | Reuse existing checkpoints and baseline results |
| 3 | Baseline validation protocol, saved predictions and visualization | Existing baseline experiment | Locate and reuse its artifacts; do not rerun baseline training |
| 4 | Development checks and compact-token gradient probe | Checks completed; probe execution unconfirmed | TODO: locate the probe report and establish whether any probe run remains |
| 5 | Train an AM critic using the same offline replay | TODO — production run not performed here | Change the memory algorithm and rerun the existing trajectories; AM only |
| 6 | Evaluate existing full-history checkpoints with full/shared/AM inputs | TODO — production run not performed here | Frozen weights; no training |

Keep both AM workflows. Do not overwrite the completed experiments, generate new
trajectories, or retokenize the stored data. Deployment paths remain blank for
Claude to fill on the machine containing the existing artifacts.

## 1. DONE — original data collection and frozen cohorts

The following records the completed baseline data preparation and collection.
The commands are retained for provenance/reference only. **Do not execute them
for the current AM work.**

### Frozen sources and sampling

FrontierSmith `main` was checked on 2026-10-24. The recipe now pins
`ac3046cce7a270e3177cbb41376f36e03de1d518` (2026-10-17, checker/reference-answer
repairs), replacing `166c8be14a62b013ef78e6868e4355421c8c6456`.
The completed collection used fresh directories and a manifest for this pinned
revision. Reuse that existing collection for AM; do not create a new cohort. The manifest records source commits,
problem hashes, independent split namespaces and exact collection quotas.

| Split | Source | Problems | Eligible trajectories per problem | Total |
|---|---|---:|---|---:|
| Training | FrontierSmith | 128 | 64 problems × 208; 64 problems × 192 | 25,600 |
| Validation | The same FrontierSmith problems, independent attempts | 128 | 32 | 4,096 |

The seeded quota assignment gives 64 problems 13 groups and 64 problems 12 groups,
with 16 trajectories per group. Twelve full rounds of 128 prompts plus one round
of 64 prompts fit exactly into 50 batches of 32 prompts. The original scheduler
and replay converter consume every accepted training trajectory once. No
padding, repetition, training-loop change or dropped remainder is needed.

The optional Frontier-CS test split is disabled by default (`TEST_PROBLEMS=0`).
Its checkout is still used for official judge/testlib infrastructure. Setting
`TEST_PROBLEMS` creates a separate test split; it is never used for the periodic
FrontierSmith validation report.

Both critics start from `HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1`, revision
`016efe4f9b8d1f472b412a26bd43e9b3f690a223`, with seed 1234. `full_history` reads the
complete concatenated input with gradients through the entire sequence;
`actor_summary` reads each actor-visible segment independently. Keep the original
Adam settings: critic LR 2e-6, weight decay .01, value clip .2, value-loss coefficient
.5, population return-standard-deviation floor .075 and gradient clipping 1.
Training and native validation use the same BF16 model; Adam moments are BF16,
with FP32 gradient accumulation. Validation values are stored as FP32.

Only naturally complete trajectories with full input **strictly below 120,000
tokens** are eligible. This counts all segment inputs, including copied summaries
and resumed prompts. There is no minimum length and no truncation to fit a batch.
Preserve rejected raw attempts and reasons; infrastructure failures stop collection.

### Historical collection commands — reference only

The completed recipe used the repository's pinned Python 3.12/CUDA 13 native environment on an eight-GPU
node. Set paths for the deployment; preparation and training use the same runtime.
The enlarged cohort and ten complete validation passes require substantially more
storage than the former 5,120-trajectory recipe. Retain original collections,
replay, native dumps and checkpoints.

```bash
bash scripts/agentic_critic/setup.sh train
source scripts/agentic_critic/activate_miles_native.sh
export MODEL_REVISION=016efe4f9b8d1f472b412a26bd43e9b3f690a223
bash scripts/agentic_critic/download_model.sh
export MODEL=/shared/models/Qwen3.5-9B-FrontierCS-SFT-DSv3.1
export DATA_ROOT="$PWD/artifacts/critic_512/sources"
export MANIFEST="$PWD/artifacts/critic_512/manifest.json"
export MAX_BATCHES=50 PROBLEMS_PER_BATCH=32 EPISODES_PER_PROBLEM=16
export VALIDATION_GROUP_SIZE=32 VALIDATION_INTERVAL=5
QUALITY_WORKERS=16 bash scripts/agentic_critic/prepare_data.sh
```

Start the FrontierSmith judge and actor in separate terminals:

```bash
SPLIT=train JUDGE_PORT=8081 bash scripts/agentic_critic/serve_judge.sh
bash scripts/agentic_critic/serve_actor.sh
```

Keep the existing actor sampling and compaction settings: thinking enabled,
temperature/top-p 1/1, 24,576 tokens per action and summary, 120,000 cumulative
generated-token budget, context 131,072, compaction threshold 12,000, maximum 10
actions and 6 submissions. The summary request is `Summarize the text above:`.
Prompts, feedback, copied summaries and environment markers are masked; generated
summary tokens are supervised in the preceding segment.

```bash
export COLLECTION_ROOT="$PWD/artifacts/critic_512/collections"
export JUDGE_URL=http://127.0.0.1:8081
export MAX_FULL_TOKENS_EXCLUSIVE=120000
export MIN_ACCEPTANCE_RATE=.25 PILOT_ATTEMPTS=32 MAX_ATTEMPTS_PER_PROBLEM=1024
SPLIT=train OUTPUT_ROOT="$COLLECTION_ROOT/train/shard-0" bash scripts/agentic_critic/generate.sh
SPLIT=validation OUTPUT_ROOT="$COLLECTION_ROOT/validation/shard-0" bash scripts/agentic_critic/generate.sh

export MERGED_ROOT="$PWD/artifacts/critic_512/merged"
mkdir -p "$MERGED_ROOT"
for split in train validation; do
  python -m agentic_critic_pipeline.merge --manifest "$MANIFEST" --split "$split" \
    --collection-ledgers "$COLLECTION_ROOT/$split"/shard-*/collection.json \
    --output "$MERGED_ROOT/$split.jsonl"
done
RAW_TRAIN_DATA="$MERGED_ROOT/train.jsonl" RAW_VALIDATION_DATA="$MERGED_ROOT/validation.jsonl" \
  OUTPUT_ROOT="$PWD/artifacts/critic_512/cohort" bash scripts/agentic_critic/freeze_cohort.sh
```

Collection supports deterministic resume. For parallel collectors, freeze
`NUM_SHARDS` and give each worker its own `SHARD_INDEX` and output directory.
Do not count incomplete attempts toward the quota or change sharding on resume.

## 2. DONE — full-history and shared critic training

The production recipe trains each critic for **50 logical batches of 512 trajectories**:
32 distinct prompts per batch × 16 trajectories per prompt. It keeps the existing
Miles batching, loss, optimizer and four-minibatch schedule. Each minibatch now
contains 128 trajectories, for 200 optimizer updates per critic. Validation runs
after completed batches **5, 10, …, 50**, with a pure forward pass through the
resident, updated Miles critic. It never advances the optimizer or scheduler.

### Historical baseline training commands — reference only

Stop actor servers sharing the training GPUs. Launch once to train the critics
sequentially on identical replay with validation every five batches. The launcher
retains native checkpoints, all per-rank validation prediction dumps and raw validation
statistics. The updated launcher also exports full-history checkpoints and runs the frozen
AM comparison when invoked for a new experiment. For this existing completed
baseline, do not invoke it again; run the standalone TODO steps below instead. Rendering
the original two-trained-critic report remains a separate, manually invoked step.

```bash
export TRAIN_DATA="$PWD/artifacts/critic_512/cohort/train/train_ready.jsonl"
export VALIDATION_DATA="$PWD/artifacts/critic_512/cohort/validation/length_included.jsonl"
export OUTPUT_ROOT=/shared/agentic/full_vs_actor-summary-512
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export MAX_SEQUENCE_LENGTH=120000
bash scripts/agentic_critic/run_comparison.sh
```

Preparation freezes `validation_replay/validation_plan.json`, including the exact
validation source/replay hashes and **20 uniformly random trajectory IDs selected
before either critic trains** (seed `ANALYSIS_SEED`, default 1234). The same IDs are
used for both models at all ten validation points. Validation requires Megatron,
synchronous `train.py`, PP=1 and CP=1; group size must divide across DP ranks.
The one-prompt validation replay chunks have 32 trajectories each. They do not
replace the 512-trajectory training batches or mutate training arguments.

Native checkpoints are under `<mode>/train/critic/checkpoints_critic/`. Per-token
training prediction dumps are disabled; training input trajectories and replay
are retained. These weights-only
checkpoints do not restore optimizer state; `RESUME=1` is rejected. Use a fresh
output root after interrupted training. An existing fixed Docker deployment can
use `MILES_RUNTIME=docker` and a pinned `MILES_IMAGE`/`MEGATRON_PATH`.

## 3. Existing baseline validation, metrics and figures

For each mode and completed batch N, `<mode>/train/validation/batch_NNNN/` contains:

| File | Content |
|---|---|
| `chunk_*/N_rank.critic.pt` | Per-rank 100-token value means, token counts, response-position ranges, rollout identities and exact scalar statistics |
| `values.jsonl.gz` | 100-token mean predictions for every validation trajectory, segment boundaries, counts, position ranges and return targets |
| `selected_trajectories.jsonl.gz` | 100-token mean predictions and observed-return targets for the fixed 20 IDs |
| `metrics.json` | EV, token/event counts, scalar jump distributions and frozen plan hash; profiles are reconstructed offline |

Existing validation dumps can be analyzed again entirely on CPU. Keep all ranks'
`chunk_*/*.critic.pt` files together with `validation_replay/validation_plan.json`
and the corresponding `validation_replay/rollout_*.pt` files. The replay supplies
episode/segment identities and offsets; the plan supplies frozen targets and the
20 preselected IDs. These files are sufficient to reconstruct EV, compaction
profiles and selected value trajectories without rerunning model inference or
exporting an HF checkpoint. The dumps are predictions, not model/optimizer
checkpoints, and cannot resume training.

EV is `1 − Var(G − V) / Var(G)`, accumulated over all supervised validation tokens,
not averaged over prompt-level EVs. Zero target variance is `null`. Results include
both exact **prior compaction count at each token** and **total compaction count
of each trajectory**. The observed target G is the final best submission score /100
at every generated-token position; it is a Monte Carlo return, not an oracle
conditional expected value.

Saved predictions use **one mean per 100 generated tokens within each segment**.
A final partial bin keeps its actual token count; bins never cross compaction boundaries.
A 50,000-token trajectory therefore has about 500 saved values, plus at most one
partial bin per segment. Masked feedback/context tokens do not enter the means.
The validation result directory uses this resolution. Full token-level values and
masks are kept separately under `$ROLLOUT_DIR/validation_values/<mode>/batch_NNNN/chunk_*/`
for local use; deliver the compact results, leaving rollout data out.

EV and compaction jump statistics are calculated from the full predictions in memory,
before binning. Each segment retains six sufficient statistics for exact token-weighted
EV/MSE/MAE. Each compaction retains only scalar eligibility, baseline and jump measurements.
The baseline is the exact mean of the preceding **1024 generated actor tokens**;
events with fewer preceding tokens are excluded. Jump summaries include the first
post-compaction token's absolute deviation and the absolute mean shift over a complete
1024-token post window, grouped by compaction ordinal and total trajectory compactions.

Curves are reconstructed from the saved means over **3076 tokens on each side** of a
boundary. They show `abs(mean(V_bin) - baseline)` at approximately 100-token resolution,
not the unavailable per-token L1 curve `mean(abs(V_token - baseline))`. Each relative
100-token window averages overlapping bin deviations by token count, then events
receive equal weight. Clipped windows retain actual coverage; eligible event counts
are shown alongside each curve. Neither per-event L1 arrays nor aggregate profile
arrays are stored during training; rendering computes the aggregate curves locally.

If manually rendered, `validation_report/` contains these PNG (220 dpi) and PDF figures, plus
`plot_data.json` with every plotted numeric value:

- `validation_ev`: paired EV curves at batches 5 through 50.
- `ev_by_prior_compaction_count` and `ev_by_total_compaction_count`: exact-count panels.
- `compaction_l1_profiles`: final-batch comparison, one panel per boundary ordinal,
  shaded baseline and an adjacent eligible-event-count panel.
- `compaction_jump_by_count`: final-batch jump means and event IQR, explicitly not a confidence interval.
- `trajectories/trajectory_01` … `trajectory_20`: one full-size panel per preselected
  trajectory, both critics, observed return and every compaction boundary.

The legacy visualization entrypoint and its dedicated analysis, figures,
diagnostics, robustness and report-upload modules have been removed. Midpoint
controls, retention sweeps, legacy input-swap reports, feature regressions, gradient
dashboards and duplicate overlaid figures from the retired visualization stack are no longer generated. This does not
remove the separate AM frozen-checkpoint workflow below. The optional
`visualize.sh` entrypoint renders only the focused validation figures listed above.
All ten validation checkpoints retain binned values and selected traces. Trajectory and boundary
figures focus on the final checkpoint; its L1 profiles are reconstructed in a streaming
pass over `values.jsonl.gz` and written into the report's `plot_data.json`.

```bash
# Render saved validation data locally on CPU.
bash scripts/agentic_critic/visualize.sh
```

Visualization reads the saved `metrics.json`, `values.jsonl.gz`, `selected_trajectories.jsonl.gz`,
`validation_plan.json` and `miles_run.json`. It writes only local PNG/PDF figures
and `plot_data.json`; it does not import W&B, create tracking runs, upload artifacts
or rerun model inference. No W&B flag, account or API key is needed. Native dumps
and all original data remain unchanged. Training-time tracking settings are separate.

## 4. Existing development verification and gradient analysis

The earlier implementation passed 117 targeted tests covering synthetic tiny Qwen3.5 checkpoints, exact
summary-token reconstruction, two independent CPU evaluation workers, checkpoint
immutability, paired cohort checks, AM cache/gradient contracts, and PNG/PDF
rendering for both the new and original reports. It does not establish production
accuracy, eight-GPU throughput, or native checkpoint-conversion success. The
broader Miles/launcher integration tests could not complete locally: the CPU
environment lacks Ray, and the separate Miles environment has an SGLang build
without `encoding_dsv4`. Run those checks in the pinned deployment environment.

The AM training addition passed 94 targeted regression tests, including an
end-to-end two-worker CPU run for all three modes and an AM single-worker
comparison. These checks verify parameter updates, global token-normalized
gradient equivalence with unequal trajectory lengths, unchanged offline inputs,
checkpoint exports, validation and figures. They use tiny random models and
synthetic trajectories; production AM training remains TODO.

The whole-prefix recompression update passed 97 targeted regression tests.
Coverage includes six consecutive compactions in the reference, value and training
readers; current-summary preservation; unchanged logical positions and detached
memory; BF16 forward/backward; and repeated biased-cache mass/output matching
against an uncompressed reference. These remain development checks, not production results.

### Existing compact-token gradient probe

See [the complete operating guide](docs/examples/compact-token-grad-norm.md) for the
frozen-model gradient probe. The production configuration randomly selects exactly
100 post-compaction segments with at least 1,024 generated actor tokens and uses
eight independent GPU workers for critic caching and actor measurements. Compare
the first 64 generated tokens with random 64-token windows; direction diagnostics
use `g_rest = g_all - g_window` for each window. Actor and critic weights never
update.

TODO: locate the existing gradient-probe result and record its completion status.
The user confirmed collection and full/shared training are complete; completion
of this separate probe has not been confirmed here. Run it only if still pending.

## 5. TODO — train the AM critic on the existing offline trajectories

**Reuse the existing offline trajectories and replay. Do not collect new
trajectories, invoke the actor or judge, or re-tokenize stored text.** The existing
rollouts and events contain the original IDs and masks. Old replay files without
summary offsets are supported: restore offsets from their original actor events,
verify exact token/mask equality, and leave the stored inputs untouched.

Claude must fill these deployment paths. Real production training and validation
for this new script have not been run in the development environment.

```bash
export MODEL=""
export TRAIN_REPLAY_DIR=""
export VALIDATION_PLAN=""
export VALIDATION_JSONL=""
export TRAINED_AM_OUTPUT=""
```

`MODEL` is the original shared SFT initialization. `TRAIN_REPLAY_DIR` is the
previously prepared replay directory containing `rollout_*.pt`, `manifest.json`,
`episodes.jsonl` and `episodes.identities.json`. Reuse the original validation
plan, validation JSONL, and its selected trajectory IDs. `TRAINED_AM_OUTPUT`
must be a fresh directory separate from both prior experiments.

```bash
: "${MODEL:?Fill MODEL}"
: "${TRAIN_REPLAY_DIR:?Fill TRAIN_REPLAY_DIR}"
: "${VALIDATION_PLAN:?Fill VALIDATION_PLAN}"
: "${VALIDATION_JSONL:?Fill VALIDATION_JSONL}"
: "${TRAINED_AM_OUTPUT:?Fill TRAINED_AM_OUTPUT}"
git submodule update --init third_party/compaction
python -m agentic_critic_pipeline.train_am_comparison \
  --model "$MODEL" --rollouts "$TRAIN_REPLAY_DIR" \
  --validation-plan "$VALIDATION_PLAN" --validation-episodes "$VALIDATION_JSONL" \
  --output "$TRAINED_AM_OUTPUT" --modes attention_matching --workers 8 --mini-batches 4 \
  --validation-interval 5 --save-interval 50 --kv-ratio 0.1
```

Full-history and shared training are already DONE. The next task is to rerun
the same existing replay with the AM memory algorithm and the original SFT
initialization, objective and schedule. The default and command above train AM
only; do not rerun the completed baselines. The script retains an explicit
`--modes full_history actor_summary attention_matching` option for a separately
requested, controlled HF-backend comparison. It is not part of the current TODO.
Existing baseline results remain available for reference, but record their
native Miles backend versus the AM runner's HF backend when interpreting differences.

The new runner uses FP32 model parameters/gradients, BF16 autocast and BF16 Adam
moments. Each worker holds a complete critic and optimizer state; eight GPUs are
data parallel, not model sharding. Verify per-device capacity before launching.
The common HF implementation differs from the existing native Miles BF16 model
implementation; document this when reporting results.

Validate the full frozen cohort every five completed replay batches and at the
end, under each trained critic's own input mode. Save checkpoints, pre-update
value summaries, loss/EV/MSE/MAE, clipping fractions, global gradient norms,
phase timings and peak CUDA memory. Render `training_report/` with the same
validation and compaction metrics plus AM training and cost curves. This is actual
AM critic optimization, not a frozen input swap. Summary queries still use all
original summary tokens; retention remains 0.1 and cross-segment KV is detached.
Each compaction refits the entire physical prefix before the current summary,
including earlier compressed KV and earlier summaries. Only the current summary
suffix is protected. The harness owns compaction triggers; this code adds no
cache-size threshold. Existing attention biases enter selection and both fits.

TODO acceptance criteria: before marking this experiment complete, verify unchanged input file hashes,
all replay batches and mini-batch updates, paired validation cohorts/checkpoints,
and the generated figures. Record actual paths, hardware, model/checkpoint IDs,
metrics and the output location here. Synthetic CPU tests are not production
results. The earlier frozen experiment must remain independently runnable.

## 6. TODO — frozen full-history checkpoints with AM inputs

The implementation is ready for deployment validation. **Production validation
has not been run in the development environment.** The user explicitly requested
that deployment paths remain blank here and that Claude fill them in on the
machine holding the existing checkpoints and validation data. Do not substitute
an untrained model, synthetic episodes, training data, or newly collected data
and present that as the requested validation result. All documentation, reports,
figure titles and captions must be in English.

Fill in these values before running:

```bash
export MODEL=""
export FULL_HISTORY_TRAINING_OUTPUT=""
export VALIDATION_PLAN=""
export VALIDATION_JSONL=""
export FROZEN_AM_OUTPUT=""
```

- `MODEL`: the exact local HF base used by the previous full-history training.
- `FULL_HISTORY_TRAINING_OUTPUT`: the existing `full_history/train` directory,
  containing `miles_run.json`. Its `checkpoints` field must locate the native
  `checkpoints_critic` directory and its completed-iteration tracker.
- `VALIDATION_PLAN`: the original `validation_replay/validation_plan.json`.
- `VALIDATION_JSONL`: the original validation trajectory JSONL, including exact
  segment tokens and original actor events. A relocated copy is allowed only
  when its SHA-256 matches the plan. Prediction dumps cannot replace these inputs.
- `FROZEN_AM_OUTPUT`: a new output directory, outside the existing training results.

Use the pinned repository environment (`transformers==5.12.1`, compatible PyTorch,
CUDA/FLA for GPU inference, and Matplotlib). Initialize the pinned upstream code:

```bash
git submodule update --init third_party/compaction
: "${MODEL:?Fill MODEL}"
: "${FULL_HISTORY_TRAINING_OUTPUT:?Fill FULL_HISTORY_TRAINING_OUTPUT}"
: "${VALIDATION_PLAN:?Fill VALIDATION_PLAN}"
: "${VALIDATION_JSONL:?Fill VALIDATION_JSONL}"
: "${FROZEN_AM_OUTPUT:?Fill FROZEN_AM_OUTPUT}"
python -m agentic_critic_pipeline.frozen_am_validation \
  --base-model "$MODEL" --training-output "$FULL_HISTORY_TRAINING_OUTPUT" \
  --plan "$VALIDATION_PLAN" --episodes "$VALIDATION_JSONL" \
  --out "$FROZEN_AM_OUTPUT" --workers 8 --dtype bfloat16 \
  --kv-ratio 0.1 --max-tokens 120000
```

The runner converts each available completed full-history checkpoint, evaluates
the **same frozen weights** using `full_history`, `actor_summary` and
`attention_matching` inputs, and renders the report. It does not train an AM
critic or update any weights. Native checkpoint iteration is the zero-based
rollout ID: `iter_0000004` is completed training batch 5. The runner uses every
available checkpoint up to the completed tracker, without inventing missing
training checkpoints. Each GPU worker loads one complete critic; there is no
tensor parallelism in this offline phase.

If native checkpoints were already converted, use repeated `--checkpoint BATCH=DIR`
instead of `--training-output`, for example `--checkpoint "5=$CRITIC_BATCH_5"`.
Verify that every supplied export came from the full-history-trained run. Each
directory must contain `critic.safetensors`, `value_head.pt` and `meta.json`.

AM uses **every original actor summary token**, with no 400/N query budget and no
decode/re-tokenize cycle. During the critic's ordinary segment forward, it
extracts the summary queries and recompresses all KV before the current summary,
including previously compressed memory and earlier summaries. The default
retention is 0.1 of this entire physical prefix; only the current summary suffix
remains intact. This follows the repeated whole-cache mechanism in
[the AM paper, Appendix F.3](https://arxiv.org/html/2602.16284v2#A6.SS3), replacing
its protected recent 20 tokens with the current actor summary. Trigger timing is
owned by the harness, with no new threshold here. Existing biases are used in
key selection, mass fitting and value fitting; fitted bias increments are added
to the selected keys' old biases before the value fit and cache read.
Cross-segment memory is detached. Frozen evaluation adds no second
model forward for probing and no training cache-rebuild pass. The optional
standalone training reader still rebuilds detached memory with its existing
no-grad forward on nonfinal segments; do not describe evaluation timing as
training overhead. Q extraction repeats the summary Q projection, and AM fitting
and biased-cache reads remain additional work.

Before declaring the production experiment complete, verify:

1. The full original frozen cohort and the same preselected trace IDs are present
   for all three inputs at each checkpoint. No token truncation or text fallback.
2. `frozen_run.json` reports `complete`; checkpoint hashes and the plan hash agree
   across all conditions. Keep the existing training outputs untouched.
3. `validation_report/` contains EV curves, exact compaction-count panels,
   compaction L1 profiles, jump distributions, the selected trajectory curves,
   `evaluation_cost.png/.pdf`, and `plot_data.json`. Inspect the figures.
4. Report real validation EV/MSE/MAE and measured inference time/memory. Timing
   excludes checkpoint loading, conversion and plotting; it does not measure
   full critic-training overhead. Explain undefined EV or ineligible boundaries.
5. Record the filled paths, checkpoint batches, hardware, software versions and
   result location here after the run. Do not mark synthetic smoke tests as
   production results or claim an AM quality improvement without measurements.

Re-render existing results without running inference:

```bash
python -m agentic_critic_pipeline.frozen_am_figures --output-root "$FROZEN_AM_OUTPUT"
```

For new experiments only, `scripts/agentic_critic/run_comparison.sh` invokes this
frozen AM comparison automatically **after both existing critic training runs**.
For previously trained checkpoints, use the command above; do not rerun training.
