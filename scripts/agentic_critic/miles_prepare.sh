#!/usr/bin/env bash
# Convert collected records into shared replay batches.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
: "${TRAIN_DATA:?Set TRAIN_DATA to the collected training JSONL}"
: "${MANIFEST:?Set MANIFEST to the collection problem manifest}"
: "${ROLLOUT_DIR:?Set ROLLOUT_DIR to the shared replay directory}"
TRAIN_DATA=$(realpath -- "$TRAIN_DATA")
mkdir -p "$ROLLOUT_DIR"
ROLLOUT_DIR=$(realpath -- "$ROLLOUT_DIR")
export TRAIN_DATA ROLLOUT_DIR
python -m agentic_critic_pipeline.cohort \
  --input "$TRAIN_DATA" --output "$ROLLOUT_DIR/cohort" --manifest "$MANIFEST" --split train \
  --max-full-tokens "${MAX_FULL_TOKENS:-120000}" --balance-training-batches \
  --problems-per-batch "${PROBLEMS_PER_BATCH:-32}" --episodes-per-problem "${EPISODES_PER_PROBLEM:-16}" \
  --seed "${SCHEDULE_SEED:-1234}"
TRAIN_DATA="$ROLLOUT_DIR/cohort/train_ready.jsonl"
export TRAIN_DATA
# These are data/shape checks in the actual preparation path, not smoke runs.
python -m agentic_critic_pipeline.miles_episodes --source "$TRAIN_DATA" --output "$ROLLOUT_DIR/episodes.jsonl" \
  --manifest "$MANIFEST" --prompts-output "$ROLLOUT_DIR/prompts.jsonl"
exec bash "$ROOT/scripts/agentic_critic/miles_runtime.sh" python \
  "$ROOT/scripts/agentic_critic/episodes_to_rollouts.py" \
  --episodes "$ROLLOUT_DIR/episodes.jsonl" --out "$ROLLOUT_DIR" \
  --steps "${MAX_BATCHES:-50}" --problems-per-step "${PROBLEMS_PER_BATCH:-32}" \
  --episodes-per-problem "${EPISODES_PER_PROBLEM:-16}" --heldout-problems 0 \
  --seed "${SCHEDULE_SEED:-1234}" --rounds --require-exhaustion "$@"
