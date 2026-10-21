# Offline agentic critic production runbook

Follow the repository contribution instructions. Run all commands from the root of this repository. Collect trajectories, train full-history and actor-summary critics with Miles, and compare their predictions with the visualizations below.

## Training contract

Initialize the actor and both critics from `HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1`, revision `016efe4f9b8d1f472b412a26bd43e9b3f690a223`. Use the Miles scalar-head initialization, seed 1234, replay files, supervised tokens, and optimizer configuration in both conditions. `full_history` reads the concatenated complete segment inputs; `actor_summary` reads each actor-visible segment independently. Full-history gradients traverse the whole input. Do not replace this with detached caches, truncation, LoRA, or sliding windows.

Collect complete trajectories until every problem reaches its eligible quota. Eligibility requires natural completion and **complete full-history input strictly below 120,000 tokens**. Include task text, feedback, all generated tokens, and copied task/summary text in that count. There is **no minimum length**. Never pad generation to reach a length or count a truncated episode toward the quota. Preserve raw rejected attempts and their reasons. The collector enforces this complete-input limit automatically.

| Split | Source | Problems | Eligible episodes per problem | Total |
|---|---|---:|---:|---:|
| train | FrontierSmith | 120 | 48 for 80 problems; 32 for 40 problems | 5,120 |
| validation | Same FrontierSmith problems, independent attempts | 120 | 8 | 960 |
| test | Frontier-CS | 20 | 8 | 160 |

Train each critic for 40 logical batches of 8 prompts/problems × 16 episodes = 128 trajectories, with four minibatches of 32 trajectories per logical batch (160 optimizer updates per critic). The manifest assigns exactly 320 groups of 16 episodes across the 120 problems: two groups for every problem and a third group for 80 problems chosen deterministically. Collect only these 5,120 eligible training episodes. The replay converter consumes every accepted training episode exactly once across 40 batches; both critics use the same replay files. Validation and test episodes are independent and are used only for evaluation. Use the Miles defaults selected by `train_b200.sh`: Adam with weight decay, LR 2e-6, weight decay .01, value clip .2, loss coefficient .5, population return-standard-deviation floor .075, gradient norm limit 1. Miles performs the old-value forward pass, return construction, clipped value regression, per-problem weighting, token-loss reduction, and four critic updates. Targets are final best submission score / 100 at the position preceding each generated actor token, including generated summaries. Mask prompts, feedback, copied summaries, and environment-inserted markers.

Use the Miles precision-aware optimizer configuration: BF16 model computation and BF16 first/second Adam moments, with FP32 gradient accumulation. The training entry point delegates model construction, training losses, distributed scheduling, and checkpoints to Miles; there is no separate critic training loop. Preserve generated token IDs and thinking. Infrastructure failures must stop the run; never turn them into zero rewards. Use the same accepted episode IDs for both critics and all input swaps. A different length limit, model, or manifest defines a new experiment and requires a separate output root.

## Prepare environments and sources

Use one pinned Python 3.12 environment for SGLang actor serving and native Miles training. The pinned wheels require a CUDA 13 compatible driver. The production launcher starts one Miles job on the eight-GPU node. Collection and evaluation use the same environment. Reserve at least 500 GB for converted model checkpoints, critic checkpoints, replay files, and value logs. The default runbook uses the pinned native Miles environment for preparation, training, checkpoint conversion, evaluation, and plotting.

```bash
bash scripts/agentic_critic/setup.sh train
source scripts/agentic_critic/activate_miles_native.sh
export MODEL_REVISION=016efe4f9b8d1f472b412a26bd43e9b3f690a223
bash scripts/agentic_critic/download_model.sh
# Set MODEL to the absolute snapshot directory printed by the download command.
export MODEL=/shared/models/Qwen3.5-9B-FrontierCS-SFT-DSv3.1
export DATA_ROOT="$PWD/artifacts/agentic_critic/sources"
export MANIFEST="$PWD/artifacts/agentic_critic/manifest.json"
QUALITY_WORKERS=16 bash scripts/agentic_critic/prepare_data.sh
```

`setup.sh train` installs PyTorch 2.13/CUDA 13, Transformer Engine 2.17, Ray 2.58.1, Transformers 5.12.1, SGLang 0.5.21, the matching official binary wheels, and a CUDA 13.0.88 compiler. Wheel URLs and hashes, Python packages, and Megatron source commits are pinned in the native environment files. The Miles training runtime is included in this repository. SGLang is included in the same pinned environment and serves the actor with exact input and output token IDs. Necessary dependency checks happen automatically.

The preparation script freezes source revisions, selected problem order, and file hashes in the manifest. Keep that manifest unchanged throughout collection and training. Necessary data/configuration checks run inside the official scripts; no separate manual checks or performance preflight are required.

## Start judging and actor serving

On a host with Docker and NVIDIA GPU support, start one official HTTP judge per dataset. The serving script distinguishes infrastructure failures from submission failures. Keep the services running in separate terminals:

```bash
SPLIT=train JUDGE_PORT=8081 bash scripts/agentic_critic/serve_judge.sh
SPLIT=test JUDGE_PORT=8082 bash scripts/agentic_critic/serve_judge.sh
bash scripts/agentic_critic/serve_actor.sh
```

Judges bind to loopback. For remote workers set `JUDGE_BIND_ADDR` explicitly and use that host's reachable address.

Keep actor defaults: thinking enabled; temperature/top-p 1/1; 24,576 tokens per action and per summary; cumulative generated-action-plus-summary budget 120,000; context 131,072; compaction threshold 12,000; maximum 10 actions and 6 submissions. The raw-log guard is 2,000,000 tokens. The summary request is `Summarize the text above:`; resume with task and parsed summary, preserving every generated summary token in the previous segment. Do not inject hidden score features into the resumed actor context.

For an eight-B200 node, one GPU normally suffices for this actor; start additional replicas on distinct GPUs/ports and assign collector shards to their URLs if useful. Keep each server at four concurrent requests initially. Stop all actor servers sharing training GPUs before training.

## Collect trajectories

Collect directly with the fixed 120,000-token complete-input ceiling:

```bash
export COLLECTION_ROOT="$PWD/artifacts/agentic_critic/collections"
export JUDGE_URL=http://127.0.0.1:8081
export MAX_FULL_TOKENS_EXCLUSIVE=120000
export MIN_ACCEPTANCE_RATE=.25 PILOT_ATTEMPTS=32 MAX_ATTEMPTS_PER_PROBLEM=1024
SPLIT=train OUTPUT_ROOT="$COLLECTION_ROOT/train/shard-0" \
  bash scripts/agentic_critic/generate.sh
SPLIT=validation OUTPUT_ROOT="$COLLECTION_ROOT/validation/shard-0" \
  bash scripts/agentic_critic/generate.sh
SPLIT=test JUDGE_URL=http://127.0.0.1:8082 OUTPUT_ROOT="$COLLECTION_ROOT/test/shard-0" \
  bash scripts/agentic_critic/generate.sh
```

For parallel collection set a fixed `NUM_SHARDS` and unique `SHARD_INDEX` per process, with disjoint output directories. Each shard owns disjoint problems and refills their manifest quotas. Collection stops at those quotas; no unused training reserve is requested. Resume the identical command in the identical directory; completed attempts are verified and reused, and unfinished attempts retain deterministic seeds. Do not change shard count or configuration during resume. Low acceptance or the per-problem attempt ceiling must stop collection for review, never produce an undersized training set.

Merge only complete production ledgers, then freeze one common cohort:

```bash
export MERGED_ROOT="$PWD/artifacts/agentic_critic/merged"
mkdir -p "$MERGED_ROOT"
for split in train validation test; do
  python -m agentic_critic_pipeline.merge --manifest "$MANIFEST" --split "$split" \
    --collection-ledgers "$COLLECTION_ROOT/$split"/shard-*/collection.json \
    --output "$MERGED_ROOT/$split.jsonl"
done
RAW_TRAIN_DATA="$MERGED_ROOT/train.jsonl" RAW_VALIDATION_DATA="$MERGED_ROOT/validation.jsonl" \
RAW_TEST_DATA="$MERGED_ROOT/test.jsonl" OUTPUT_ROOT="$PWD/artifacts/agentic_critic/cohort" \
  bash scripts/agentic_critic/freeze_cohort.sh
```

The merge automatically checks raw/accepted hashes, masks, rewards, IDs, and exact per-problem manifest quotas. Cohort freezing rejects incomplete batches or surplus training episodes. Keep test and same-problem validation separate throughout.

## Run production training

Use eight B200 GPUs on one node. Set `NPROC_PER_NODE` to the actual GPU count and launch the comparison once on that node:

```bash
export TRAIN_DATA="$PWD/artifacts/agentic_critic/cohort/train/train_ready.jsonl"
export EVAL_DATA="$PWD/artifacts/agentic_critic/cohort/test/length_included.jsonl"
export VALIDATION_DATA="$PWD/artifacts/agentic_critic/cohort/validation/length_included.jsonl"
export OUTPUT_ROOT=/shared/agentic/full_vs_actor-summary
export NNODES=1 NODE_RANK=0 NPROC_PER_NODE=8
export MAX_SEQUENCE_LENGTH=120000
export MAX_BATCHES=40
mkdir -p "$OUTPUT_ROOT"
bash scripts/agentic_critic/run_comparison.sh
```

The default production batch is 8 problems × 16 episodes, split into four minibatches of 32 episodes; both critics train for 40 batches. `run_comparison.sh` converts the frozen records into training episodes and prepares one shared replay directory. It invokes Miles in `shared` mode for `actor_summary`, and `full_history` mode for `full_history`. Tensor parallelism is 1 and data parallelism is 8 by default. The distributed sampler keeps all segments of an episode on the same rank and in the same minibatch, and aligns actor segments with their full-history critic sample. Each of the four minibatches contains 32 complete trajectories globally. Full-history episodes remain whole; the 120k input ceiling is not divided across GPUs.

The comparison trains the critics sequentially, converts their native Miles checkpoints, evaluates each checkpoint under both input modes on test and validation, and generates every analysis below. Evaluation uses FP32 with bounded-memory chunking to keep fixed-checkpoint comparisons stable; training retains BF16 computation. Stop actor servers on these GPUs before launching. Do not start both critics simultaneously.

Native Miles checkpoints are under `<mode>/train/critic/checkpoints_critic/`; pre-update predictions are under `<mode>/train/critic/dump/`. Converted checkpoints are under `<mode>/critic_hf/`. The analysis entry point exports the stored predictions into the visualization format; it does not recompute them with another trainer.

Checkpoints contain weights without optimizer state. Do not treat them as exact optimizer/RNG resume points. After interrupted training, use a fresh `OUTPUT_ROOT` and replay the same frozen data from the start; `RESUME=1` is rejected rather than silently restarting optimizer moments. Collection resume remains supported. Keep the native environment active for the full comparison; its activation script sets `MILES_RUNTIME=native` and the matching compiler and library paths. To use an existing Miles Docker deployment instead, set `MILES_RUNTIME=docker` and `MILES_IMAGE` to that deployment's fixed image digest, and provide its `MEGATRON_PATH` (usually `/root/Megatron-LM`). Use the same image for preparation, training, and conversion.

## Visualizations and analysis outputs

`run_comparison.sh` automatically evaluates both critics, runs all analyses, and renders the following figures. To regenerate them from completed value logs with the same exported data/output paths:

```bash
bash scripts/agentic_critic/visualize.sh
```

### Main comparison and output layout

The main comparison uses the separately trained `actor_summary` and `full_history` checkpoints on identical episodes and supervised tokens. Outputs are relative to `OUTPUT_ROOT`:

| Directory | Contents |
|---|---|
| `analysis/` | `analysis.json`, `report.md`, and a three-panel `<phase>.png`: EV over training batches, EV by prior compactions, and prediction deviation around compaction boundaries. Numeric results also include MSE, MAE, bias, paired confidence intervals, jump statistics, retention, and variance share. |
| `figures/<phase>/` | The detailed PNG figures listed below. `figures/figures.json` stores their numeric values, eligible event counts, and uncertainty estimates. |
| `diagnostics/` | `diagnostics.json`, `diagnostics-report.md`, and the eight-panel `<phase>-diagnostics.png` described below. |
| `input_swap_actor_summary/` | Analysis, figures, and diagnostics comparing both input modes with the same actor-summary-trained checkpoint. |
| `input_swap_full_history/` | Analysis, figures, and diagnostics comparing both input modes with the same full-history-trained checkpoint. |
| `validation/` | The same main comparison and both input-swap suites on independent attempts at training problems, kept separate from cross-dataset test. The main subdirectories are `analyze/`, `figures/`, and `diagnostics/`. |

`<phase>` is `pre_update` for recorded training predictions or `heldout` for fixed-checkpoint evaluation. Input swaps and the separate validation suite contain held-out results. Chronological training controls apply only when pre-update records are available.

### Detailed figures

Every filename below is generated in `figures/<phase>/`; the same figure suite is also generated inside each input-swap and validation output directory when the required records exist. Blue denotes `full_history`; orange denotes `actor_summary`.

| PNG filename | What it displays |
|---|---|
| `ev_training_comparison.png` | Both critics' explained variance (EV) over logical batches. |
| `ev_training_by_compaction.png` | Six EV panels for all tokens and tokens with 0, 1, 2, 3, or 4+ prior compactions; the training curves use moving batch windows. |
| `boundary_profile.png` | Absolute deviation from the pre-event mean versus actor-token offset, with separate panels for actual compaction boundaries and ordinary segment midpoints. |
| `paired_boundary_profile.png` | The same boundary and midpoint profiles overlaid on one axis to compare both critics against the midpoint control. |
| `jump_by_compaction_index.png` | Mean absolute value jump by compaction index, alongside corresponding midpoint jumps. |
| `retention_by_compaction_index.png` | Change in correlation between predicted value and final return across successive compactions, with midpoint controls. |
| `retention_by_offset.png` | Return-correlation change at post-event actor-token offsets 0, 128, 512, and 2048; all offsets use a common eligible event cohort. |
| `jump_by_removed_tokens.png` | Mean absolute jump across quintiles of tokens removed at an individual compaction, with midpoint reference lines. |
| `jump_histogram.png` | Absolute-jump distributions for each critic at compaction boundaries and midpoints. |
| `jump_direction.png` | Signed value jump versus final return minus pre-event value, grouped into quintiles, for boundaries and midpoints. |
| `jump_training.png` | Boundary and midpoint jump magnitudes over moving training-batch windows, with a separate panel per critic. |
| `paired_jump_training.png` | Both critics' boundary and midpoint jump curves overlaid on one axis. |
| `window_sensitivity.png` | Correlation retention by compaction index using pre/post windows of 8, 32, 128, and 512 actor tokens. |
| `training_halves.png` | Correlation retention by compaction index for the chronological early and late halves of training; unavailable for a fixed checkpoint alone. |
| `same_episodes.png` | Retention at the first, second, and third-or-later compactions using the same eligible episodes with at least three actual compaction boundaries, plus the matched midpoint control in numeric output. |
| `retention_by_removed.png` | Correlation retention across thirds of individually removed-token counts, comparing all compactions with first-compaction-only events. |
| `jump_by_cumulative_removed.png` | Mean absolute jump across quintiles of cumulative tokens removed before and including the current compaction. |
| `episode_traces.png` | Up to four individual trajectories: both value curves over cumulative actor tokens, vertical compaction markers, and a horizontal final-return reference. |

The default boundary window is 32 actor tokens, profile span is 256, and profile bin width is 16. `WINDOW`, `SPAN`, and `BIN_SIZE` override these analysis settings. Confidence intervals use paired problem-cluster bootstrap resampling, with `BOOTSTRAP=2000` and `ANALYSIS_SEED=0` by default.

### Diagnostic dashboard

Each `<phase>-diagnostics.png` contains these eight panels:

1. History-feature residual R² from regression with problems held out.
2. Squared-error fraction explained by those history features.
3. Value-head gradient norm by prior-compaction bucket.
4. Value-head gradient signal-to-noise ratio (GSNR) by prior-compaction bucket.
5. Gradient cosine similarity between distinct episodes of the same problem, by prior-compaction bucket.
6. Critic EV across thirds of cumulative removed-token counts.
7. Critic EV by post-compaction actor-token position.
8. EV of token-time score-only reference predictors by prior-compaction bucket.

Head-gradient panels use the per-segment head gradients recorded automatically by held-out evaluation. The recorded gradient is `mean((V-G)*h)` for scalar-head weight parameters only. It is not a full-backbone gradient or a PPO minibatch gradient. Training-only dashboards may therefore have undefined head-gradient panels. Score-only references use only information visible at the prediction token; extracting a numeric score from a summary is a documented heuristic in the diagnostic report.

### Interpretation

Use the last 30 logical batches for the primary training readout. Keep the independent test and same-problem validation results separate. Input-swap plots compare readings of a single fixed checkpoint and complement the main comparison of two separately trained critics. Zero-compaction input equality is an exact-input control only within a fixed checkpoint.

Undefined EV or correlations remain null. Figures label insufficient eligible data rather than replacing missing statistics with zeros. Intervals crossing zero are inconclusive. Numeric companion files retain event counts and eligibility information for interpreting each panel.
