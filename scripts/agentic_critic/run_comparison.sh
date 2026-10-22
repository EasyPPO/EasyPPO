#!/usr/bin/env bash
# Train both critic conditions and run their evaluation and visualization suites.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${MODEL:?Set MODEL to the frozen SFT snapshot}"
: "${TRAIN_DATA:?Set TRAIN_DATA}"
: "${EVAL_DATA:?Set EVAL_DATA}"
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
export EVAL_DATA=$(realpath -- "$EVAL_DATA")
export VALIDATION_DATA=$(realpath -- "$VALIDATION_DATA")
mkdir -p "$OUTPUT_ROOT"
export OUTPUT_ROOT=$(realpath -- "$OUTPUT_ROOT")
export NPROC_PER_NODE=${NPROC_PER_NODE:-8} MAX_BATCHES=${MAX_BATCHES:-40}
export MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-120000}
export ROLLOUT_DIR=${ROLLOUT_DIR:-"$OUTPUT_ROOT/replay"}
export MILES_MODEL_DIR=${MILES_MODEL_DIR:-"$OUTPUT_ROOT/miles_models"}
# One wandb group per comparison; each trained critic and the analysis dump are distinct runs in it.
# The group is stored so visualize.sh (and reruns of it) report into the same group.
if [[ -z ${WANDB_GROUP:-} && -s "$OUTPUT_ROOT/wandb_group.txt" ]]; then
  WANDB_GROUP=$(<"$OUTPUT_ROOT/wandb_group.txt")
fi
export WANDB_GROUP=${WANDB_GROUP:-"$(basename -- "$OUTPUT_ROOT")-$(date +%Y%m%d-%H%M%S)"}
printf '%s\n' "$WANDB_GROUP" > "$OUTPUT_ROOT/wandb_group.txt"
# Preparing once ensures both critics consume the identical stored Miles batches.
bash "$ROOT/scripts/agentic_critic/miles_prepare.sh"
for trained in full_history actor_summary; do
  train_output="$OUTPUT_ROOT/$trained/train"
  OUTPUT="$train_output" MODE="$trained" WANDB_RUN_NAME="$WANDB_GROUP-train-$trained" \
    bash "$ROOT/scripts/agentic_critic/train_b200.sh" "$@"
  python -m agentic_critic_pipeline.miles_analysis training \
    --episodes "$TRAIN_DATA" --rollouts "$ROLLOUT_DIR" --dump "$train_output/critic/dump" \
    --mode "$trained" --out "$train_output"
  converted="$OUTPUT_ROOT/$trained/critic_hf"
  OUTPUT="$train_output" CONVERTED_OUTPUT="$converted" \
    bash "$ROOT/scripts/agentic_critic/miles_convert.sh"
  for input_mode in full_history actor_summary; do
    DATA="$EVAL_DATA" OUTPUT="$OUTPUT_ROOT/$trained/eval_$input_mode" MODE="$input_mode" \
      TRAINED_MODE="$trained" CHECKPOINT="$converted" \
      bash "$ROOT/scripts/agentic_critic/evaluate_b200.sh"
    DATA="$VALIDATION_DATA" OUTPUT="$OUTPUT_ROOT/$trained/validation_$input_mode" MODE="$input_mode" \
      TRAINED_MODE="$trained" CHECKPOINT="$converted" \
      bash "$ROOT/scripts/agentic_critic/evaluate_b200.sh"
  done
done
bash "$ROOT/scripts/agentic_critic/visualize.sh"
