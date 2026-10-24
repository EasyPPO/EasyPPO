#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${MODEL:?Set MODEL to the local SFT checkpoint}"
: "${ROLLOUT_DIR:?Set ROLLOUT_DIR to the prepared Miles rollout directory}"
: "${OUTPUT:?Set OUTPUT}"
export MODEL=$(realpath -- "$MODEL")
export ROLLOUT_DIR=$(realpath -- "$ROLLOUT_DIR")
mkdir -p "$OUTPUT"
export OUTPUT=$(realpath -- "$OUTPUT")
export MILES_MODEL_DIR=${MILES_MODEL_DIR:-"$OUTPUT/miles_models"}
mkdir -p "$MILES_MODEL_DIR"
export MILES_MODEL_DIR=$(realpath -- "$MILES_MODEL_DIR")
export PROMPT_DATA=${PROMPT_DATA:-"$ROLLOUT_DIR/prompts.jsonl"}
export PROMPT_DATA=$(realpath -- "$PROMPT_DATA")
exec bash "$ROOT/scripts/agentic_critic/miles_runtime.sh" python \
  "$ROOT/scripts/agentic_critic/miles_train.py" \
  --model "$MODEL" --rollouts "$ROLLOUT_DIR" --output "$OUTPUT" \
  --model-dir "$MILES_MODEL_DIR" --prompt-data "$PROMPT_DATA" \
  --mode "${MODE:-full_history}" --steps "${MAX_BATCHES:-${MAX_STEPS:-50}}" \
  --gpus "${NPROC_PER_NODE:-8}" --tensor-parallel "${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
  --max-tokens "${MAX_SEQUENCE_LENGTH:-120000}" \
  --save-interval "${SAVE_INTERVAL:-50}" \
  --problems-per-batch "${PROBLEMS_PER_BATCH:-32}" --episodes-per-problem "${EPISODES_PER_PROBLEM:-16}" \
  --mini-batches "${MINI_BATCHES:-4}" \
  --validation-plan "${VALIDATION_PLAN:?Prepare frozen validation replay first}" \
  --validation-interval "${VALIDATION_INTERVAL:-5}" "$@"
