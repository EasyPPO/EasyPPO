# Train an attention-matching critic on existing trajectories

Data collection and full/shared baseline training are already complete. The
remaining training task is AM on the same stored trajectories. Both workflows
remain available:

| Workflow | Weights | Entry point |
| --- | --- | --- |
| Frozen input comparison | Reuse previously trained full-history checkpoints; no updates | `python -m agentic_critic_pipeline.frozen_am_validation` |
| AM training | Train AM on the existing replay, from the original base | `python -m agentic_critic_pipeline.train_am_comparison` |

The original Miles training recipe and compact-token gradient probe are unchanged.
The new trained comparison uses one common Hugging Face backend for all conditions.
It does not add AM to Megatron or substitute AM results into the native Miles
comparison. Do not mix the two backends' curves as a controlled experiment.

## Reuse the existing offline data

**Do not generate new trajectories. Do not start an actor server or a judge. Do
not tokenize the stored text again.** Reuse the original training replay and
frozen validation cohort from the earlier experiment:

- `replay/rollout_0.pt`, `rollout_1.pt`, ... and `manifest.json`.
- `replay/episodes.jsonl`, containing the original actor events.
- `replay/episodes.identities.json`, linking training identities to the original source hash.
- The existing `validation_replay/validation_plan.json` and original validation JSONL.

These files contain original token IDs, actor loss masks, rewards and summary
events. The reader verifies token/mask/return equality between events and replay.
For older replay files without `metadata.summary_span`, it restores offsets from
the existing actor events in memory, without rewriting replay or re-encoding text.
Missing events or token mismatches are errors; there is no text fallback.

Training and validation IDs must be disjoint. Every configured full replay is
consumed exactly once, in its original batch and episode order. No resampling,
new rollout collection, prompt rewriting, padding or truncation is performed.
A byte-offset index lets workers read the original event pool without retaining
all trajectory tokens in RAM. `--steps` can explicitly select the first N existing
replay batches for a short run; the default consumes all batches.

## Run

Use the pinned Qwen3.5 environment and initialize the compaction submodule. Fill
in deployment paths in `Claude.md`; they are intentionally not supplied here.
Choose a fresh output directory, separate from the earlier training and frozen
comparison results.

```bash
git submodule update --init third_party/compaction
python -m agentic_critic_pipeline.train_am_comparison \
  --model "$MODEL" \
  --rollouts "$TRAIN_REPLAY_DIR" \
  --validation-plan "$VALIDATION_PLAN" \
  --validation-episodes "$VALIDATION_JSONL" \
  --output "$TRAINED_AM_OUTPUT" \
  --modes attention_matching --workers 8 --mini-batches 4 \
  --validation-interval 5 --save-interval 50 \
  --kv-ratio 0.1
```

The default is `attention_matching` only. Do not rerun the completed full/shared
baselines for the current task. The script also supports an explicitly requested
`--modes full_history actor_summary attention_matching` comparison, training the
three conditions sequentially from the same HF base and head initialization.
That optional controlled HF-backend comparison is separate from the current TODO.
Keep earlier Miles results as reference and disclose the backend/precision
difference when comparing them with the new AM run.
The script writes local statistics and figures; it does not
start W&B runs or upload data.

Each GPU holds a complete critic, gradients and optimizer state. This is
replicated data parallelism, **not tensor parallelism or sharded model state**.
Model and gradient storage are FP32, default autocast is BF16, and Adam moments
are BF16. Unlike the native Miles recipe, this backend does not store model
parameters in BF16. Verify per-GPU capacity before a production run; eight GPUs
do not pool their memory to fit one model. Use one worker and `--autocast none`
for small CPU checks. Production GPU throughput has not been measured locally.

## Training and validation

Defaults preserve the existing standalone critic objective: LR `2e-6`, weight
decay `0.01`, value clip `0.2`, value-loss coefficient `0.5`, return-standard-
deviation floor `0.075`, global gradient clipping at `1`, and four mini-batches.
Prompt weights use the whole replay batch's population return variance. Losses
are normalized by the global mini-batch's generated-token count. Replicas SUM
these already-normalized gradients once per mini-batch, including contributions
from ranks with unequal token counts, before clipping and updating.

Each replay batch first receives a no-gradient value pass; its predictions stay
fixed as the clipping reference for all mini-batches. Training updates the entire
critic backbone and value head. AM memory uses every original actor summary token
and 0.1 pre-summary KV retention. Actor summary generation is unchanged. KV fits
are detached and rebuilt at current weights. Summary probing has no additional
model pass; the checkpointed training reader retains its existing no-grad cache
reconstruction pass on nonfinal segments. This work is included in training time.

After each validation interval and at the final batch, evaluate the resident,
updated critic on the exact original validation cohort under its own input mode.
No validation optimizer/scheduler update occurs. Reuse the original selected
trajectory IDs. Validation and save intervals use completed replay batches, not
optimizer mini-steps. Save a checkpoint at every validation point, every requested
save interval, and the final batch. Checkpoints contain FP32 backbone weights,
value head and provenance; optimizer state is not saved and training resume is
not supported. Stored checkpoints can be large.

## Outputs

For each mode, the new output root contains:

- `training_log.json`: pre-update EV/MSE/MAE, every mini-batch's loss, value clipping
  fraction and pre-clip global gradient norm, value-pass time, training/update
  time and peak CUDA allocated memory.
- `pre_update/batch_NNNN/values.rankR.jsonl.gz`: segment-local 100-token value means
  and exact sufficient statistics from the training value pass.
- `checkpoints/batch_NNNN/`: `critic.safetensors`, `value_head.pt`, and `meta.json`.
- `validation/batch_NNNN/`: paired validation means, selected traces, exact
  EV/MSE/MAE, compaction-jump statistics, checkpoint hashes and inference timings.

`training_report/` includes PNG/PDF figures and `plot_data.json`:

- Training loss, clipping fraction, global gradient norm and pre-update EV.
- Value-pass time, training time and peak memory across replay batches.
- Validation EV versus completed training batches, grouped by prior/total compactions.
- Final-checkpoint compaction L1 profiles and jump distributions.
- The same preselected validation trajectories, with the selected trained modes
  (AM only by default) and return targets.

Training timings use the slowest rank for each phase, include cache construction,
AM fitting, backward and gradient synchronization as applicable, and exclude
validation and checkpoint export. They are not total job wall-clock time. The
existing per-token compact-boundary gradient experiment remains a separate probe;
these logs report the usual whole-update gradient norm.

Re-render without training or model inference:

```bash
python -m agentic_critic_pipeline.trained_am_figures --output-root "$TRAINED_AM_OUTPUT"
```

Development tests use tiny random Qwen3.5 models and synthetic offline data to
check optimization, unchanged inputs, distributed gradient equivalence, checkpoint
saving and plotting. They do not establish production model quality or overhead.
