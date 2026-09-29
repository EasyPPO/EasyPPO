#!/usr/bin/env bash
# Download pinned DAPO/AIME parquet sources, verify their checksums, remove the
# upstream sampling repetitions, and emit canonical veRL-ready data files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/aime}"
RAW_DIR="${RAW_DIR:-$DATA_DIR/raw}"
OVERWRITE="${OVERWRITE:-0}"
KEEP_RAW="${KEEP_RAW:-0}"

TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/aime-2024.parquet}"
TRAIN_RAW_FILE="${TRAIN_RAW_FILE:-$RAW_DIR/dapo-math-17k.repeated.parquet}"
VAL_RAW_FILE="${VAL_RAW_FILE:-$RAW_DIR/aime-2024.repeated.parquet}"

TRAIN_REVISION="65877096c24ffa7abc4e4fa5edb95cf3413a5674"
VAL_REVISION="aa49075e24ad594b79fdf0bdcefa735c2181be67"
TRAIN_SHA256="534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941"
VAL_SHA256="12154e38a716d12db5731f9a022ae69a610c4f7d0e0dcc04e902887a686877e7"
TRAIN_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/$TRAIN_REVISION/data/dapo-math-17k.parquet?download=true"
VAL_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/$VAL_REVISION/data/aime-2024.parquet?download=true"

case "$OVERWRITE" in 0|1) ;; *) echo "ERROR: OVERWRITE must be 0 or 1" >&2; exit 1 ;; esac
case "$KEEP_RAW" in 0|1) ;; *) echo "ERROR: KEEP_RAW must be 0 or 1" >&2; exit 1 ;; esac

if [ -n "${PYTHON_BIN:-}" ]; then
  python_bin="$PYTHON_BIN"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  python_bin="$PROJECT_ROOT/.venv/bin/python"
else
  python_bin=python3
fi

if ! "$python_bin" -c 'import pyarrow' >/dev/null 2>&1; then
  echo "ERROR: pyarrow is required for AIME data preparation." >&2
  echo "Install pyarrow first, or set PYTHON_BIN to the project environment." >&2
  exit 1
fi

verify_args=(
  --train-output "$TRAIN_FILE"
  --validation-output "$VAL_FILE"
  --verify-only
)
if [ "$OVERWRITE" != "1" ] && [ -s "$TRAIN_FILE" ] && [ -s "$VAL_FILE" ] && \
  "$python_bin" "$SCRIPT_DIR/prepare_aime_data.py" "${verify_args[@]}"; then
  echo "[data] Canonical AIME data is already ready."
  exit 0
fi

mkdir -p "$DATA_DIR" "$RAW_DIR"

download_file() {
  target="$1"
  url="$2"
  expected_sha256="$3"

  if [ -s "$target" ] && printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --status; then
    if [ "$OVERWRITE" != "1" ]; then
      echo "[data] Reusing verified upstream file $target"
      return
    fi
  fi

  tmp="${target}.part.$$"
  trap 'rm -f "$tmp"' EXIT
  echo "[data] Downloading pinned source $url"
  curl --fail --location --retry 5 --retry-delay 2 --output "$tmp" "$url"
  printf '%s  %s\n' "$expected_sha256" "$tmp" | sha256sum --check --status || {
    echo "ERROR: SHA-256 mismatch for $url" >&2
    exit 1
  }
  mv "$tmp" "$target"
  trap - EXIT
}

download_file "$TRAIN_RAW_FILE" "$TRAIN_URL" "$TRAIN_SHA256"
download_file "$VAL_RAW_FILE" "$VAL_URL" "$VAL_SHA256"

prepare_args=(
  --train-input "$TRAIN_RAW_FILE"
  --validation-input "$VAL_RAW_FILE"
  --train-output "$TRAIN_FILE"
  --validation-output "$VAL_FILE"
)
if [ "$OVERWRITE" = "1" ]; then
  prepare_args+=(--force)
fi
"$python_bin" "$SCRIPT_DIR/prepare_aime_data.py" "${prepare_args[@]}"

if [ "$KEEP_RAW" != "1" ]; then
  rm -f "$TRAIN_RAW_FILE" "$VAL_RAW_FILE"
  rmdir "$RAW_DIR" 2>/dev/null || true
  echo "[data] Removed repeated upstream cache; set KEEP_RAW=1 to retain it."
fi

echo "[data] Training parquet:   $TRAIN_FILE (17,917 unique prompts)"
echo "[data] Validation parquet: $VAL_FILE (30 unique AIME-2024 prompts)"
