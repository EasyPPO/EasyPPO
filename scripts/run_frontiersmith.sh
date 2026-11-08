#!/usr/bin/env bash
# FrontierSmith training with Frontier-CS validation. Pass trainer.val_only=true for evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path "$ROOT/configs" --config-name easyppo_aime \
  actor_rollout_ref.model.path="${MODEL_PATH:-HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1}" \
  data.train_files='[data/frontiercs/train_synthetic_200.parquet]' \
  data.val_files='[data/frontiercs/frontiercs_val.parquet]' \
  data.max_prompt_length=8192 data.max_response_length=12000 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20192 \
  actor_rollout_ref.rollout.max_model_len=20192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=20192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20192 \
  critic.ppo_max_token_len_per_gpu=20192 \
  critic.ppo_infer_max_token_len_per_gpu=20192 \
  critic.forward_max_token_len_per_gpu=20192 \
  reward.custom_reward_function.path="$ROOT/verl/utils/reward_score/frontiercs.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.judge_url="${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:8082}" \
  trainer.project_name=EasyPPO-FrontierSmith \
  trainer.experiment_name=frontiersmith-200 \
  trainer.default_local_dir="${EASYPPO_OUTPUT_DIR:-outputs}/frontiersmith/checkpoints" \
  trainer.rollout_data_dir="${EASYPPO_OUTPUT_DIR:-outputs}/frontiersmith/rollouts" \
  trainer.validation_data_dir="${EASYPPO_OUTPUT_DIR:-outputs}/frontiersmith/validation" \
  "$@"
