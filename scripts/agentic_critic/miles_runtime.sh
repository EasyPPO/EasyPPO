#!/usr/bin/env bash
# Execute preparation, training, or conversion in the Miles runtime.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export MILES_DIR="$ROOT"
export MEGATRON_PATH=${MEGATRON_PATH:-/root/Megatron-LM}
export PYTHONPATH="$ROOT:$MEGATRON_PATH${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${MILES_RUNTIME:-docker} == native ]]; then
  exec "$@"
fi
[[ ${MILES_RUNTIME:-docker} == docker ]] || { echo 'MILES_RUNTIME must be docker or native' >&2; exit 2; }
scratch=${MILES_SCRATCH_DIR:-"$ROOT/.cache/miles-runtime"}
mkdir -p "$scratch"/{tmp,vartmp,cache,triton,cupy}
scratch=$(realpath -- "$scratch")
external_paths=()
mounts=(--mount "type=bind,src=$ROOT,dst=$ROOT")
mounts+=(--mount "type=bind,src=$scratch/tmp,dst=/tmp"
  --mount "type=bind,src=$scratch/vartmp,dst=/var/tmp"
  --mount "type=bind,src=$scratch/cache,dst=/root/.cache"
  --mount "type=bind,src=$scratch/triton,dst=/root/.triton"
  --mount "type=bind,src=$scratch/cupy,dst=/root/.cupy")
for name in MODEL ROLLOUT_DIR OUTPUT OUTPUT_ROOT TRAIN_DATA VALIDATION_DATA VALIDATION_PLAN MANIFEST DATA PROMPT_DATA MILES_MODEL_DIR; do
  path=${!name:-}
  [[ -n "$path" ]] || continue
  if [[ ! -e "$path" ]]; then
    case "$name" in OUTPUT|OUTPUT_ROOT|MILES_MODEL_DIR) mkdir -p "$path" ;; *) continue ;; esac
  fi
  path=$(realpath -- "$path")
  [[ -d "$path" ]] || path=$(dirname -- "$path")
  [[ "$path" == "$ROOT" || "$path" == "$ROOT/"* || "$path" == "$MILES_DIR" ]] && continue
  duplicate=0
  for mounted in "${external_paths[@]:-}"; do [[ "$mounted" == "$path" ]] && duplicate=1; done
  if [[ $duplicate == 0 ]]; then mounts+=(--mount "type=bind,src=$path,dst=$path"); external_paths+=("$path"); fi
done
exec docker run --rm --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 "${mounts[@]}" \
  --workdir "$MILES_DIR" --env PYTHONPATH --env MILES_DIR --env MEGATRON_PATH \
  --env WANDB_API_KEY --env WANDB_MODE --env WANDB_PROJECT --env WANDB_ENTITY --env WANDB_BASE_URL \
  --env WANDB_GROUP --env WANDB_RUN_NAME \
  "${MILES_IMAGE:-radixark/miles:latest}" "$@"
