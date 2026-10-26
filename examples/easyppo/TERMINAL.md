# Terminal Training with TB2.1 Validation

Train Qwen3.5-9B on all eligible TMAX tasks and validate on the 89 official
Terminal-Bench 2.1 tasks. Thinking is enabled, using the mini-swe-agent harness.

## 1. Install

Use a [Miles GPU environment](../../docs/getting-started/installation.md) with
Python 3.12+, Docker Engine, and Docker Compose v2. The default recipe uses
8 GPUs. Docker must be accessible from the training environment.

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO_fork.git
cd EasyPPO_fork
python -m pip install -e '.[terminal]'
```

## 2. Prepare the data

```bash
python scripts/make_miles_terminal_data.py \
  --tasks-dir /data/harbor \
  --out /data/prepared
```

This downloads the pinned TMAX and TB2.1 datasets if missing, then creates
`tmax.train.jsonl` and `tb21.eval.jsonl`. No separate TB2.1 download or TMAX
validation split is needed. Keep the complete task directories under
`/data/harbor`, accessible at the same path on every rollout worker.

## 3. Start training

Adjust the model, Megatron-LM, and output paths for your machine:

```bash
python examples/easyppo/run_qwen3_5_9b_terminal.py \
  --prompt-data /data/prepared/tmax.train.jsonl \
  --tasks-dir /data/harbor \
  --data-dir /data/prepared \
  --model-dir /data/models \
  --megatron-path /opt/Megatron-LM \
  --output-dir /data/training \
  --trials-dir /data/terminal-trials \
  --eval-interval 10
```

The launcher prepares the model and runs TB2.1 validation before training and
every 10 rollouts. It uses the training model service; no separate evaluation
server is needed. Checkpoints are saved under `/data/training/<run-id>/checkpoints`
and task logs under `/data/terminal-trials`. Use `--help` to adjust batch sizes
and GPU resources.
