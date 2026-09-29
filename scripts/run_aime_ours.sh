#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Safe preview uses Hydra composition only: no Ray, tokenizer, model or GPU.
if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$PYTHON_BIN" scripts/check_easyppo_config.py "$@"
fi

"$PYTHON_BIN" scripts/check_easyppo_config.py "$@" >/dev/null
"$PYTHON_BIN" scripts/prepare_aime_data.py \
  --train-output data/aime/dapo-math-17k.parquet \
  --validation-output data/aime/aime-2024.parquet --verify-only
exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$ROOT/configs" --config-name easyppo_aime "$@"
