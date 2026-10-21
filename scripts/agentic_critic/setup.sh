#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
kind=${1:-train}
if [[ "$kind" != train && "$kind" != miles-native && "$kind" != sglang ]]; then
  echo 'Usage: setup.sh train|miles-native|sglang' >&2
  exit 2
fi
exec bash scripts/agentic_critic/setup_miles_native.sh
