#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATA_ROOT:?Set the sources directory created by prepare_data.sh}"
SPLIT=${SPLIT:-train}
if [[ "$SPLIT" == test ]]; then
  PROBLEMS_DIR=$DATA_ROOT/Frontier-CS/algorithmic/problems
else
  [[ "$SPLIT" == train || "$SPLIT" == validation ]] || { echo "Invalid SPLIT" >&2; exit 1; }
  PROBLEMS_DIR=$DATA_ROOT/FrontierSmith/Frontier-CS/algorithmic/problems
fi
# Reviewed upstream build with explicit infrastructure-error propagation.
# The pinned checkout stays unchanged; the patch applies only to this build tree.
PYTHON=${PYTHON:-python}
build_root=$(mktemp -d)
trap 'rm -rf -- "$build_root"' EXIT
"$PYTHON" -m agentic_critic_pipeline.prepare_judge_build \
  --source "$DATA_ROOT/Frontier-CS/algorithmic" --output "$build_root/context"
docker build -t agentic-frontier-judge "$build_root/context"
JUDGE_STATE=${JUDGE_STATE:-$PWD/artifacts/agentic_critic/judge-$SPLIT}
mkdir -p "$JUDGE_STATE/submissions" "$JUDGE_STATE/data"
cp "$build_root/context/agentic_judge_patch.json" "$JUDGE_STATE/agentic_judge_patch.json"
rm -rf -- "$build_root"
trap - EXIT
exec docker run --name "agentic-judge-$SPLIT" --privileged --shm-size=4g \
  --ulimit stack=-1:-1 -p "${JUDGE_BIND_ADDR:-127.0.0.1}:${JUDGE_PORT:-8081}:8081" \
  -e PORT=8081 -e GJ_ADDR=http://127.0.0.1:5050 -e SAVE_OUTPUTS=false -e ES_NO_FALLBACK=true \
  -e "JUDGE_WORKERS=${JUDGE_WORKERS:-16}" -e "GJ_PARALLELISM=${JUDGE_WORKERS:-16}" \
  -v "$PROBLEMS_DIR:/app/problems:ro" -v "$JUDGE_STATE/submissions:/app/submissions" \
  -v "$JUDGE_STATE/data:/app/data" agentic-frontier-judge
