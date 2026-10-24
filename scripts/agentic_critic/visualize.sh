#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the two-critic experiment}"
exec python -m agentic_critic_pipeline.validation_figures --output-root "$OUTPUT_ROOT" "$@"
