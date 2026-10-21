#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/agentic_critic/activate_miles_native.sh
exec python -m sglang.launch_server \
  --model-path "${MODEL:-HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1}" \
  --revision "${MODEL_REVISION:-016efe4f9b8d1f472b412a26bd43e9b3f690a223}" \
  --served-model-name qwen35-9b --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --tp-size "${TP_SIZE:-1}" --dtype bfloat16 --attention-backend flashinfer --mm-attention-backend sdpa \
  --context-length "${CONTEXT_LENGTH:-131072}" --max-running-requests "${MAX_NUM_SEQS:-4}" \
  --mem-fraction-static "${GPU_MEMORY_UTILIZATION:-0.85}" "$@"
