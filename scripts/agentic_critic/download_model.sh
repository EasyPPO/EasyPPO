#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export MODEL=${MODEL:-HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1}
export MODEL_REVISION=${MODEL_REVISION:-016efe4f9b8d1f472b412a26bd43e9b3f690a223}
"$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['MODEL'], revision=os.environ['MODEL_REVISION'],
                        ignore_patterns=['*.md', '*.png', '*.jpg', '*.mp4']))
PY
