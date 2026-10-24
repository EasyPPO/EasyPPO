#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$PWD/artifacts/agentic_critic/sources}
MANIFEST=${MANIFEST:-$PWD/artifacts/agentic_critic/manifest.json}
# Verified FrontierSmith main on 2026-10-24; includes checker and reference-answer repairs.
SMITH_REVISION=${SMITH_REVISION:-ac3046cce7a270e3177cbb41376f36e03de1d518}
FRONTIER_REVISION=${FRONTIER_REVISION:-aec2aff05f3ece1b150c7c40827b0cea449e0b28}
mkdir -p "$DATA_ROOT"
fetch() {
  local name=$1 url=$2 revision=$3
  if [[ ! -d "$DATA_ROOT/$name/.git" ]]; then
    git init "$DATA_ROOT/$name"
    git -C "$DATA_ROOT/$name" remote add origin "$url"
    git -C "$DATA_ROOT/$name" fetch --depth 1 origin "$revision"
    git -C "$DATA_ROOT/$name" checkout --detach FETCH_HEAD
  fi
  [[ $(git -C "$DATA_ROOT/$name" rev-parse HEAD) == "$revision" ]] || {
    echo "Revision mismatch: $DATA_ROOT/$name; choose a fresh DATA_ROOT" >&2; exit 1;
  }
}
fetch FrontierSmith https://github.com/FrontierCS/FrontierSmith.git "$SMITH_REVISION"
fetch Frontier-CS https://github.com/FrontierCS/Frontier-CS.git "$FRONTIER_REVISION"
if [[ "${FETCH_ONLY:-0}" == 1 ]]; then
  exit 0
fi
QUALITY_ROOT=${QUALITY_ROOT:-$DATA_ROOT/quality}
SMITH_QUALITY_REPORT=${SMITH_QUALITY_REPORT:-$QUALITY_ROOT/frontiersmith.json}
FRONTIER_QUALITY_REPORT=${FRONTIER_QUALITY_REPORT:-$QUALITY_ROOT/frontiercs.json}
TESTLIB_DIR=${TESTLIB_DIR:-$DATA_ROOT/Frontier-CS/algorithmic/judge/include}
mkdir -p "$QUALITY_ROOT"
if [[ ! -f "$SMITH_QUALITY_REPORT" ]]; then
  echo "Running required FrontierSmith checker-input preflight"
  "$PYTHON" -m agentic_critic_pipeline.validate_tasks \
    --root "$DATA_ROOT/FrontierSmith/Frontier-CS/algorithmic/problems" --dataset frontiersmith \
    --testlib "$TESTLIB_DIR" --workers "${QUALITY_WORKERS:-16}" \
    --probe-cpu-seconds "${CHECKER_CPU_SECONDS:-60}" \
    --probe-memory-bytes "${CHECKER_MEMORY_BYTES:-268435456}" \
    --cache "$QUALITY_ROOT/checkers-smith" --output "$SMITH_QUALITY_REPORT"
fi
if [[ ${TEST_PROBLEMS:-0} -gt 0 && ! -f "$FRONTIER_QUALITY_REPORT" ]]; then
  echo "Running required Frontier-CS checker-input preflight"
  "$PYTHON" -m agentic_critic_pipeline.validate_tasks \
    --root "$DATA_ROOT/Frontier-CS/algorithmic/problems" --dataset frontiercs \
    --testlib "$TESTLIB_DIR" --workers "${QUALITY_WORKERS:-16}" \
    --probe-cpu-seconds "${CHECKER_CPU_SECONDS:-60}" \
    --probe-memory-bytes "${CHECKER_MEMORY_BYTES:-268435456}" \
    --cache "$QUALITY_ROOT/checkers-frontier" --output "$FRONTIER_QUALITY_REPORT"
fi
"$PYTHON" -m agentic_critic_pipeline.data \
  --smith-root "$DATA_ROOT/FrontierSmith/Frontier-CS/algorithmic/problems" \
  --frontier-root "$DATA_ROOT/Frontier-CS/algorithmic/problems" \
  --smith-quality-report "$SMITH_QUALITY_REPORT" --frontier-quality-report "$FRONTIER_QUALITY_REPORT" \
  --training-batches "${MAX_BATCHES:-50}" --prompts-per-batch "${PROBLEMS_PER_BATCH:-32}" \
  --episodes-per-group "${EPISODES_PER_PROBLEM:-16}" --validation-group-size "${VALIDATION_GROUP_SIZE:-32}" \
  --output "$MANIFEST" --train-count 128 --test-count "${TEST_PROBLEMS:-0}" --seed "${DATA_SEED:-20261021}" "$@"
