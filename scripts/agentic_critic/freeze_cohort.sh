#!/usr/bin/env bash
# Freeze exact eligible quotas after collection; never shorten or silently discard training samples.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RAW_TRAIN_DATA:?Set RAW_TRAIN_DATA to the complete merged raw training split}"
: "${RAW_VALIDATION_DATA:?Set RAW_VALIDATION_DATA to the complete merged raw validation split}"
: "${MANIFEST:?Set MANIFEST to the frozen collection manifest}"
: "${OUTPUT_ROOT:?Choose a new directory for the immutable common cohort}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output already exists: $OUTPUT_ROOT" >&2; exit 1; }
PYTHON=${PYTHON:-python}
MAX_FULL_TOKENS=${MAX_FULL_TOKENS:-120000}
"$PYTHON" -m agentic_critic_pipeline.cohort \
  --input "$RAW_TRAIN_DATA" --output "$OUTPUT_ROOT/train" --max-full-tokens "$MAX_FULL_TOKENS" \
  --manifest "$MANIFEST" --split train \
  --problems-per-batch "${PROBLEMS_PER_BATCH:-32}" --episodes-per-problem "${EPISODES_PER_PROBLEM:-16}" \
  --balance-training-batches --seed "${SCHEDULE_SEED:-1234}" --epochs "${EPOCHS:-1}"
"$PYTHON" -m agentic_critic_pipeline.cohort \
  --input "$RAW_VALIDATION_DATA" --output "$OUTPUT_ROOT/validation" --max-full-tokens "$MAX_FULL_TOKENS" \
  --manifest "$MANIFEST" --split validation
if [[ -n "${RAW_TEST_DATA:-}" ]]; then
"$PYTHON" -m agentic_critic_pipeline.cohort \
  --input "$RAW_TEST_DATA" --output "$OUTPUT_ROOT/test" --max-full-tokens "$MAX_FULL_TOKENS" \
  --manifest "$MANIFEST" --split test
fi
echo "Train both critics on $OUTPUT_ROOT/train/train_ready.jsonl with the same seed and epochs."
echo "Validate both critics on $OUTPUT_ROOT/validation/length_included.jsonl."
