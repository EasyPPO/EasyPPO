#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8081}"
GJ_ADDR="${GJ_ADDR:-http://127.0.0.1:5050}"
GJ_PARALLELISM="${GJ_PARALLELISM:-$(nproc)}"
JUDGE_WORKERS="${JUDGE_WORKERS:-$(nproc)}"

export PORT GJ_ADDR JUDGE_WORKERS

go_judge_http_addr="${GJ_ADDR#http://}"
go_judge_http_addr="${go_judge_http_addr#https://}"

cleanup() {
  if [[ -n "${GO_JUDGE_PID:-}" ]]; then
    kill "$GO_JUDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

go-judge \
  -http-addr "$go_judge_http_addr" \
  -parallelism "$GJ_PARALLELISM" \
  -dir /tmp/go-judge \
  &
GO_JUDGE_PID=$!

node /app/server.js
