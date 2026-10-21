#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
SPLIT=${SPLIT:-train}
SHARD_INDEX=${SHARD_INDEX:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
MANIFEST=${MANIFEST:-artifacts/agentic_critic/manifest.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/agentic_critic/collections/$SPLIT/shard-$SHARD_INDEX}
# Training quotas come from MANIFEST: exactly 5120 trajectories across 120 problems.
# Test tasks require a judge mounted with Frontier-CS, training/validation with FrontierSmith.
# The actor settings are pinned below. MAX_FULL_TOKENS is a
# high log-size safety ceiling, including copied resume prompts. It does not
# admit those lengths to training: the collector accepts only natural finishes
# with full input strictly below MAX_FULL_TOKENS_EXCLUSIVE and refills each quota.
# MAX_EPISODE_RESPONSE_TOKENS caps generated actions + summaries, including thinking;
# it excludes task/feedback tokens and copied summaries in resumed prompts.
: "${JUDGE_URL:?Set the official judge URL serving this split}"
exec "$PYTHON" -m agentic_critic_pipeline.collect \
  --manifest "$MANIFEST" --split "$SPLIT" --output-root "$OUTPUT_ROOT" \
  --backend sglang --model "${MODEL:-HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1}" \
  --model-revision "${MODEL_REVISION:-016efe4f9b8d1f472b412a26bd43e9b3f690a223}" \
  --served-model-name qwen35-9b --actor-url "${ACTOR_URL:-http://127.0.0.1:8000}" \
  --judge http --judge-url "$JUDGE_URL" --workers "${WORKERS:-4}" \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-1.0}" \
  --context-length "${CONTEXT_LENGTH:-131072}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-24576}" \
  --max-summary-tokens "${MAX_SUMMARY_TOKENS:-24576}" \
  --max-episode-response-tokens "${MAX_EPISODE_RESPONSE_TOKENS:-${MAX_FULL_TOKENS_EXCLUSIVE:-120000}}" \
  --max-full-tokens "${MAX_FULL_TOKENS:-2000000}" \
  --max-full-tokens-exclusive "${MAX_FULL_TOKENS_EXCLUSIVE:-120000}" \
  --min-acceptance-rate "${MIN_ACCEPTANCE_RATE:-0.25}" \
  --pilot-attempts "${PILOT_ATTEMPTS:-32}" \
  --max-attempts-per-problem "${MAX_ATTEMPTS_PER_PROBLEM:-1024}" \
  --compact-threshold-tokens "${COMPACT_THRESHOLD_TOKENS:-12000}" \
  --max-turns "${MAX_TURNS:-10}" --max-submissions "${MAX_SUBMISSIONS:-6}" "$@"
