#!/usr/bin/env bash
# Native equivalent of the Miles CUDA 13 image for offline critic training.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Native Miles requires Linux x86_64' >&2; exit 2; }
missing=0
for command in git gcc g++ make rsync pkill ss; do
  command -v "$command" >/dev/null || missing=1
done
system_libs=$(ldconfig -p 2>/dev/null || true)
for library in libnuma.so.1 libibverbs.so.1 librdmacm.so.1 libaio.so.1; do
  [[ "$system_libs" == *"$library"* ]] || missing=1
done
if [[ $missing == 1 ]]; then
  [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null || {
    echo 'Native Miles needs git, build-essential, rsync, procps and iproute2 installed by the system administrator' >&2
    exit 2
  }
  apt-get update
  aio_package=libaio1
  if ! apt-cache show libaio1 >/dev/null 2>&1; then aio_package=libaio1t64; fi
  DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential rsync procps iproute2 \
    ninja-build libnuma1 libibverbs1 librdmacm1 "$aio_package" pkg-config
fi
export AGENTIC_MILES_VENV=${AGENTIC_MILES_VENV:-"$ROOT/.venv-agentic-miles"}
export MILES_DIR="$ROOT"
export MEGATRON_PATH=${MEGATRON_PATH:-"$ROOT/.cache/megatron-agentic"}
if [[ ! -x "$AGENTIC_MILES_VENV/bin/python" ]]; then uv venv --python 3.12 "$AGENTIC_MILES_VENV"; fi
uv pip sync --python "$AGENTIC_MILES_VENV/bin/python" "$ROOT/scripts/agentic_critic/requirements-miles-native.lock"
MEGATRON_COMMIT=2fbe25a442a539b2796cdf40e77dfb2c90782529
if [[ ! -d "$MEGATRON_PATH/.git" ]]; then
  [[ ! -e "$MEGATRON_PATH" ]] || { echo 'MEGATRON_PATH exists without Git metadata' >&2; exit 2; }
  mkdir -p "$(dirname -- "$MEGATRON_PATH")"
  git init -q "$MEGATRON_PATH"
  git -C "$MEGATRON_PATH" remote add origin https://github.com/radixark/Megatron-LM.git
  git -C "$MEGATRON_PATH" fetch --depth 1 origin "$MEGATRON_COMMIT"
  git -C "$MEGATRON_PATH" checkout -q --detach FETCH_HEAD
fi
[[ $(git -C "$MEGATRON_PATH" rev-parse HEAD) == "$MEGATRON_COMMIT" ]] || { echo 'Megatron source pin mismatch' >&2; exit 2; }
git -C "$MEGATRON_PATH" diff --quiet
uv pip install --python "$AGENTIC_MILES_VENV/bin/python" --no-deps -e "$MEGATRON_PATH"
"$AGENTIC_MILES_VENV/bin/python" "$ROOT/scripts/agentic_critic/prepare_miles_native.py"
source "$ROOT/scripts/agentic_critic/activate_miles_native.sh"
python -c 'import matplotlib; import miles.backends.megatron_utils.actor; print("Native Miles imports ready")'
echo "Native Miles ready. Source $ROOT/scripts/agentic_critic/activate_miles_native.sh"
