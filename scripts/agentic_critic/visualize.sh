#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${TRAIN_DATA:?Set TRAIN_DATA to the frozen training JSONL}"
: "${EVAL_DATA:?Set EVAL_DATA to the frozen validation or test JSONL}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the completed two-critic experiment}"
args=(
  --actor-summary "$OUTPUT_ROOT/actor_summary/train" "$OUTPUT_ROOT/actor_summary/eval_actor_summary"
  --full-history "$OUTPUT_ROOT/full_history/train" "$OUTPUT_ROOT/full_history/eval_full_history"
  --episodes "$TRAIN_DATA" "$EVAL_DATA"
  --window "${WINDOW:-32}" --span "${SPAN:-256}" --bin-size "${BIN_SIZE:-16}"
  --bootstrap "${BOOTSTRAP:-2000}" --seed "${ANALYSIS_SEED:-0}"
)
python -m agentic_critic_pipeline.analyze "${args[@]}" --out "$OUTPUT_ROOT/analysis"
python -m agentic_critic_pipeline.figures "${args[@]}" --out "$OUTPUT_ROOT/figures"
# Residual/score references use pre-update values; gradient diagnostics are held-out only.
python -m agentic_critic_pipeline.diagnostics "${args[@]}" --out "$OUTPUT_ROOT/diagnostics"
for trained in actor_summary full_history; do
  swap_args=(
    --actor-summary "$OUTPUT_ROOT/$trained/eval_actor_summary"
    --full-history "$OUTPUT_ROOT/$trained/eval_full_history"
    --episodes "$TRAIN_DATA" "$EVAL_DATA" --allow-input-swap-diagnostic
    --window "${WINDOW:-32}" --span "${SPAN:-256}" --bin-size "${BIN_SIZE:-16}"
    --bootstrap "${BOOTSTRAP:-2000}" --seed "${ANALYSIS_SEED:-0}"
  )
  python -m agentic_critic_pipeline.analyze "${swap_args[@]}" --out "$OUTPUT_ROOT/input_swap_$trained/analysis"
  python -m agentic_critic_pipeline.figures "${swap_args[@]}" --out "$OUTPUT_ROOT/input_swap_$trained/figures"
  python -m agentic_critic_pipeline.diagnostics "${swap_args[@]}" --out "$OUTPUT_ROOT/input_swap_$trained/diagnostics"
done
if [[ -n "${VALIDATION_DATA:-}" ]]; then
  validation_args=(
    --actor-summary "$OUTPUT_ROOT/actor_summary/validation_actor_summary"
    --full-history "$OUTPUT_ROOT/full_history/validation_full_history"
    --episodes "$TRAIN_DATA" "$VALIDATION_DATA"
    --window "${WINDOW:-32}" --span "${SPAN:-256}" --bin-size "${BIN_SIZE:-16}"
    --bootstrap "${BOOTSTRAP:-2000}" --seed "${ANALYSIS_SEED:-0}"
  )
  for module in analyze figures diagnostics; do
    python -m "agentic_critic_pipeline.$module" "${validation_args[@]}" --out "$OUTPUT_ROOT/validation/$module"
  done
  for trained in actor_summary full_history; do
    validation_swap_args=(
      --actor-summary "$OUTPUT_ROOT/$trained/validation_actor_summary"
      --full-history "$OUTPUT_ROOT/$trained/validation_full_history"
      --episodes "$TRAIN_DATA" "$VALIDATION_DATA" --allow-input-swap-diagnostic
      --window "${WINDOW:-32}" --span "${SPAN:-256}" --bin-size "${BIN_SIZE:-16}"
      --bootstrap "${BOOTSTRAP:-2000}" --seed "${ANALYSIS_SEED:-0}"
    )
    for module in analyze figures diagnostics; do
      python -m "agentic_critic_pipeline.$module" "${validation_swap_args[@]}" --out "$OUTPUT_ROOT/validation/input_swap_$trained/$module"
    done
  done
fi
