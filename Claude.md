# Pending execution: frozen full-history checkpoints with attention matching

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
extracts the summary queries and fits only the preceding segment KV. The default
retention is 0.1 for that pre-summary block; summary KV and previous memory blocks
remain intact. Cross-segment memory is detached. Frozen evaluation adds no second
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

For new experiments, `scripts/agentic_critic/run_comparison.sh` now invokes this
frozen AM comparison automatically **after both existing critic training runs**.
For previously trained checkpoints, use the command above; do not rerun training.

Development verification passed 117 targeted tests covering synthetic tiny Qwen3.5 checkpoints, exact
summary-token reconstruction, two independent CPU evaluation workers, checkpoint
immutability, paired cohort checks, AM cache/gradient contracts, and PNG/PDF
rendering for both the new and original reports. It does not establish production
accuracy, eight-GPU throughput, or native checkpoint-conversion success. The
broader Miles/launcher integration tests could not complete locally: the CPU
environment lacks Ray, and the separate Miles environment has an SGLang build
without `encoding_dsv4`. Run those checks in the pinned deployment environment.

# Compact-token gradient experiment

See [the complete operating guide](docs/examples/compact-token-grad-norm.md) for the
frozen-model gradient probe. The production configuration randomly selects exactly
100 post-compaction segments with at least 1,024 generated actor tokens and uses
eight independent GPU workers for critic caching and actor measurements. Compare
the first 64 generated tokens with random 64-token windows; direction diagnostics
use `g_rest = g_all - g_window` for each window. Actor and critic weights never
update.

# Full-history versus actor-summary critic comparison

The production recipe trains each critic for **50 logical batches of 512 trajectories**:
32 distinct prompts per batch × 16 trajectories per prompt. It keeps the existing
Miles batching, loss, optimizer and four-minibatch schedule. Each minibatch now
contains 128 trajectories, for 200 optimizer updates per critic. Validation runs
after completed batches **5, 10, …, 50**, with a pure forward pass through the
resident, updated Miles critic. It never advances the optimizer or scheduler.

## Frozen sources and sampling

FrontierSmith `main` was checked on 2026-10-24. The recipe now pins
`ac3046cce7a270e3177cbb41376f36e03de1d518` (2026-10-17, checker/reference-answer
repairs), replacing `166c8be14a62b013ef78e6868e4355421c8c6456`.
Use a fresh `DATA_ROOT`, manifest and output directory; do not reuse old revision
quality reports or collected cohorts. The manifest records source commits,
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

## Prepare and collect

Use the repository's pinned Python 3.12/CUDA 13 native environment on an eight-GPU
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

## Train both critics and retain validation dumps

Stop actor servers sharing the training GPUs. Launch once to train the critics
sequentially on identical replay with validation every five batches. The launcher
retains native checkpoints, all per-rank validation prediction dumps and raw validation
statistics. After both training runs, the launcher exports the saved full-history
checkpoints and runs the separate frozen AM comparison described above. Rendering
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

## Metrics, figures and raw data

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
controls, retention sweeps, input-swap reports, feature regressions, gradient
dashboards and duplicate overlaid figures are no longer generated. The optional
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
