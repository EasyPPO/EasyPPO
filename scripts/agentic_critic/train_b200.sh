#!/usr/bin/env bash
# Train a critic with the production Miles configuration.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export MAX_STEPS=${MAX_BATCHES:-50}
exec bash "$ROOT/scripts/agentic_critic/miles_train.sh" "$@"
