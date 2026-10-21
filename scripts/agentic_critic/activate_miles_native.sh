#!/usr/bin/env bash
# Source this file before native collection-to-Miles preparation or training.
_miles_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export AGENTIC_MILES_VENV=${AGENTIC_MILES_VENV:-"$_miles_root/.venv-agentic-miles"}
export MILES_DIR="$_miles_root"
export MEGATRON_PATH=${MEGATRON_PATH:-"$_miles_root/.cache/megatron-agentic"}
_miles_site=$("$AGENTIC_MILES_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export CUDA_HOME="$_miles_site/nvidia/cu13"
export PATH="$AGENTIC_MILES_VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$_miles_site/nvidia/cu13/lib:$_miles_site/nvidia/cudnn/lib:$_miles_site/nvidia/nccl/lib:$_miles_site/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$_miles_root:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
export MILES_RUNTIME=native
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MAX_JOBS=${MAX_JOBS:-8}
export RAY_USAGE_STATS_ENABLED=0
unset _miles_root _miles_site
