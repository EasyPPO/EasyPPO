#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${MODEL:?Set MODEL to the original SFT checkpoint}"
: "${OUTPUT:?Set OUTPUT to the Miles training output}"
: "${CONVERTED_OUTPUT:?Set CONVERTED_OUTPUT}"
export MODEL=$(realpath -- "$MODEL")
export OUTPUT=$(realpath -- "$OUTPUT")
mkdir -p "$CONVERTED_OUTPUT"
export OUTPUT_ROOT=$(realpath -- "$CONVERTED_OUTPUT")
exec bash "$ROOT/scripts/agentic_critic/miles_runtime.sh" python \
  "$ROOT/scripts/agentic_critic/miles_convert.py" \
  --model "$MODEL" --training-output "$OUTPUT" --output "$OUTPUT_ROOT"
