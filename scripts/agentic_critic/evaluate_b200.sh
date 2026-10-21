#!/usr/bin/env bash
# Evaluate a converted Miles critic on complete episodes.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL:?Set MODEL}"
: "${DATA:?Set DATA to the held-out JSONL}"
: "${OUTPUT:?Set OUTPUT}"
: "${MODE:?Set MODE}"
: "${TRAINED_MODE:?Set TRAINED_MODE}"
: "${CHECKPOINT:?Set CHECKPOINT to the converted Miles critic directory}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if [[ ${NNODES:-1} != 1 ]]; then echo 'This production launcher supports one node' >&2; exit 2; fi
# Rerunning evaluation regenerates every shard; stale shards must not be mixed.
mkdir -p "$OUTPUT"
rm -f -- "$OUTPUT"/values.rank*.jsonl
exec torchrun --standalone --nproc-per-node "${NPROC_PER_NODE:-8}" \
  -m agentic_critic_pipeline.miles_analysis heldout \
  --episodes "$DATA" --base-model "$MODEL" --critic "$CHECKPOINT" --out "$OUTPUT" \
  --input-mode "$MODE" --trained-mode "$TRAINED_MODE" \
  --checkpoint-run-id "$CHECKPOINT" --checkpoint-batch "${MAX_BATCHES:-40}" \
  --max-tokens "${MAX_SEQUENCE_LENGTH:-120000}" "$@"
