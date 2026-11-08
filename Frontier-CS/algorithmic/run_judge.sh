#!/bin/bash
# Run judge container without docker compose.
# Usage: ./run_judge.sh
# Stop: docker stop Competitive-Programming

set -e
cd "$(dirname "$0")"

JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
GJ_PARALLELISM="${GJ_PARALLELISM:-8}"

# Build image
# echo "Building image..."
# docker build -t frontiercs-judge .

# Stop existing container if any
docker rm -f Competitive-Programming 2>/dev/null || true

# Run
echo "Starting container..."
docker run -d \
  --name Competitive-Programming \
  --privileged \
  --shm-size=4g \
  -p 8082:8082 \
  -e PORT=8082 \
  -e GJ_ADDR=http://127.0.0.1:5050 \
  -e JUDGE_WORKERS="$JUDGE_WORKERS" \
  -e GJ_PARALLELISM="$GJ_PARALLELISM" \
  -e SAVE_OUTPUTS=false \
  -v "$(pwd)/problems:/app/problems" \
  -v "$(pwd)/submissions:/app/submissions" \
  -v "$(pwd)/data:/app/data" \
  frontiercs-judge

echo "Judge running at http://localhost:8082"
echo "Check: curl http://localhost:8082/health"
