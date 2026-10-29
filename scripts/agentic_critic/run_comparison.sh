#!/usr/bin/env bash
# Train both critic conditions with periodic validation and retain all native dumps.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${MODEL:?Set MODEL to the frozen SFT snapshot}"
: "${TRAIN_DATA:?Set TRAIN_DATA}"
: "${VALIDATION_DATA:?Set VALIDATION_DATA}"
: "${OUTPUT_ROOT:?Choose OUTPUT_ROOT for this Miles run}"
if [[ ${NNODES:-1} != 1 || ${NODE_RANK:-0} != 0 ]]; then
  echo 'Launch once on the single eight-GPU node' >&2; exit 2
fi
if [[ ${RESUME:-0} != 0 ]]; then
  echo 'Miles weights-only checkpoints do not restore optimizer state; use a fresh OUTPUT_ROOT after interrupted training' >&2
  exit 2
fi
export MODEL=$(realpath -- "$MODEL")
export TRAIN_DATA=$(realpath -- "$TRAIN_DATA")
export VALIDATION_DATA=$(realpath -- "$VALIDATION_DATA")
mkdir -p "$OUTPUT_ROOT"
export OUTPUT_ROOT=$(realpath -- "$OUTPUT_ROOT")
export NPROC_PER_NODE=${NPROC_PER_NODE:-8} MAX_BATCHES=${MAX_BATCHES:-50}
export MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-120000}
export ROLLOUT_DIR=${ROLLOUT_DIR:-"$OUTPUT_ROOT/replay"}
export VALIDATION_PLAN="$OUTPUT_ROOT/validation_replay/validation_plan.json"
export MILES_MODEL_DIR=${MILES_MODEL_DIR:-"$OUTPUT_ROOT/miles_models"}
# One wandb group per comparison; each trained critic is a distinct run in it.
# Keep the group available for optional offline analysis later.
if [[ -z ${WANDB_GROUP:-} && -s "$OUTPUT_ROOT/wandb_group.txt" ]]; then
  WANDB_GROUP=$(<"$OUTPUT_ROOT/wandb_group.txt")
fi
export WANDB_GROUP=${WANDB_GROUP:-"$(basename -- "$OUTPUT_ROOT")-$(date +%Y%m%d-%H%M%S)"}
printf '%s\n' "$WANDB_GROUP" > "$OUTPUT_ROOT/wandb_group.txt"
# Preparing once ensures both critics consume the identical stored Miles batches.
bash "$ROOT/scripts/agentic_critic/miles_prepare.sh"
# Freeze the held-out replay and randomly selected trace IDs before either critic trains.
bash "$ROOT/scripts/agentic_critic/miles_runtime.sh" python -m agentic_critic_pipeline.validation \
  --episodes "$VALIDATION_DATA" --training-episodes "$ROLLOUT_DIR/cohort/train_ready.jsonl" \
  --out "$(dirname -- "$VALIDATION_PLAN")" --group-size "${VALIDATION_GROUP_SIZE:-32}" \
  --trace-count 20 --seed "${ANALYSIS_SEED:-1234}"
for trained in full_history actor_summary; do
  train_output="$OUTPUT_ROOT/$trained/train"
  OUTPUT="$train_output" MODE="$trained" WANDB_RUN_NAME="$WANDB_GROUP-train-$trained" \
    bash "$ROOT/scripts/agentic_critic/train_b200.sh" "$@"
done
# Final frozen-weight ablation: reuse every saved full-history checkpoint and
# the original validation cohort. This stage performs no additional training.
bash "$ROOT/scripts/agentic_critic/miles_runtime.sh" python -m agentic_critic_pipeline.frozen_am_validation \
  --base-model "$MODEL" --plan "$VALIDATION_PLAN" \
  --training-output "$OUTPUT_ROOT/full_history/train" --out "$OUTPUT_ROOT/frozen_am" \
  --workers "$NPROC_PER_NODE" --max-tokens "$MAX_SEQUENCE_LENGTH"
