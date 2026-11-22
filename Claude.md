# Qwen3.5-9B PPO

**Current runs use Terminal Bench only.**

Run from the repository root in a Miles training image with Megatron installed.
Set `MODEL_DIR`, `DATA_DIR` and `RUN_DIR` to shared directories accessible at
the same paths on every worker. On an existing Ray cluster, run the launchers
on the head node with `MILES_SCRIPT_EXTERNAL_RAY=1`, `MASTER_ADDR` set to its
worker-reachable address and `RAY_ADDRESS` set to its Ray Jobs HTTP endpoint.

```bash
python -m pip install -e '.[agentic,terminal]'
```

## FCS

**Configure.** Train synchronous PPO from `HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1`
on 200 FrontierSmith tasks. Set `NUM_NODES` and `GPUS_PER_NODE` to the allocation.
Training and rollout share GPUs with offload. Training uses TP2 and rollout uses
one GPU per engine with four concurrent trials. Set `JUDGE_URL` to the judge's
internal HTTP address reachable from every worker.

**Prepare.** Start the CPU judge on its host, then run the preparation command
on the training cluster to generate Harbor tasks and JSONL, download the model,
and convert its checkpoint on GPU.

```bash
(cd Frontier-CS/algorithmic && FRONTIERCS_JUDGE_BIND_IP=0.0.0.0 docker compose up -d --build)
curl --fail "$JUDGE_URL/problems"
```

```bash
FCS=(
  examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE"
  --frontier-judge-url "$JUDGE_URL"
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
)
python "${FCS[@]}" --action prepare
```

**Launch.** Select `full_detach` or `actor-summary`. Each batch contains 32 prompts
× 16 trajectories, split into four critic mini-batches of 128. The actor updates
once per batch after 30 critic-only batches. Generation uses a 131,072-token
context, 24,576 tokens per action, an 8,192-token summary cap, a 12,000-token
compaction threshold and a 150,000-token episode budget.

```bash
MODE=full_detach
python "${FCS[@]}" --action train --critic-context-mode "$MODE" --run-id "fs-9b-$MODE" \
  --num-rollout 100 --num-critic-only-steps 30 --critic-steps-per-rollout 4 \
  --rollout-batch-size 32 --samples-per-prompt 16 \
  --eval-interval 10 --eval-max-tasks 0 --eval-samples-per-prompt 3 \
  --save-interval 5 --max-keep 2
```

Evaluation uses all 180 Frontier-CS tasks with three trials each at startup and
every ten batches. Reuse the run ID and output directory to resume. For a
Modal-hosted FCS cluster, use the [Modal launch instructions](modal_launchers/README.md)
with `--model-size 9b`.

## Terminal Bench

### Set shared paths

Run from the repository root. Replace `/shared/easyppo` with a writable mount
visible at the same path on the training, rollout and evaluation nodes.

```bash
export TB_ROOT=/shared/easyppo
export MODEL_DIR="$TB_ROOT/models" DATA_DIR="$TB_ROOT/data" RUN_DIR="$TB_ROOT/runs"
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$RUN_DIR"
```

### Prepare the base model

Run once on a node with one allocated GPU, the Miles training image, the
`hf` CLI and Megatron at `/root/Megatron-LM`. The command downloads
`Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
and converts it to a Megatron distributed checkpoint.

```bash
python -m examples.easyppo.terminus_full_detach --prepare-model-only --model-dir "$MODEL_DIR"
test -s "$MODEL_DIR/Qwen3.5-9B/config.json"
test -s "$MODEL_DIR/Qwen3.5-9B/model.safetensors.index.json"
grep -qx release "$MODEL_DIR/Qwen3.5-9B_torch_dist/latest_checkpointed_iteration.txt"
```

Both critic modes use these same artifacts. SGLang and the tokenizer load
`$MODEL_DIR/Qwen3.5-9B`. Fresh actor and critic runs initialize from
`$MODEL_DIR/Qwen3.5-9B_torch_dist`. Training saves their checkpoints separately
under `$RUN_DIR/<run-id>/checkpoints` and `$RUN_DIR/<run-id>/checkpoints_critic`.

### Prepare the training and evaluation data

Configure Modal credentials on the rollout and evaluation workers. Prepare
both datasets once on the shared mount, starting with a new
`$DATA_DIR/harbor/agentpick-csk4gby2` directory path.

```bash
python -m pip install 'harbor[modal]==0.24.0'
python scripts/make_miles_agentpick_data.py --tasks-dir "$DATA_DIR/harbor"
python scripts/make_miles_terminal_data.py --dataset terminal-bench-2.1 \
  --tasks-dir "$DATA_DIR/harbor" --out "$DATA_DIR"
wc -l "$DATA_DIR/harbor/agentpick-csk4gby2/train.jsonl" "$DATA_DIR/tb21.eval.jsonl"
```

To update an existing dataset, prepare it in a fresh directory and point all workers at that directory. Preparation applies the pinned CPU, memory and disk quotas, sets agent and verifier timeouts to 2,400 and 600 seconds, and preserves task entrypoints and protected-file checks for the Modal runner. The pinned AgentPick split produces 543 training rows. Terminal-Bench 2.1
produces 89 evaluation rows. Both launchers read these JSONL files.

**Configure.** The two launchers use the same asynchronous PPO recipe.

| Setting | Default |
|---|---|
| GPU allocation | 2 nodes × 8 GPUs. Actor and critic share 6, training rollout uses 8, evaluation uses 2. The split scales as 3:4:1 for eight-GPU nodes. Trainer TP1, one GPU per rollout or eval engine. |
| Async rollout | 32 prompts × 12 trajectories per batch. Concurrency cap 1,000 (996 in complete groups of 12), 512 requests per engine. Weight publication uses `in_place`; maximum weight staleness is 4 updates. |
| Generation | Terminus-2 XML, 63,488-token context, up to 16,384 new tokens per action. Compact at 12,000 context tokens, up to 2,048 new tokens per summary. Each compaction replaces the actor history with the original task and the visible summary, including an empty summary. Episode budget 102,400 generated tokens including summaries, with one reminder at 51,200. |
| Updates | 150 batches, first 30 critic-only. Four critic updates of 96 trajectories per batch (LR 5e-6, value clip 0.2, value loss coefficient 0.5). One actor update per subsequent batch (LR 3e-6, one warmup step). |
| Actor loss | Score centering with PPO advantages, prompt means, sampled rollout log probabilities, top-128 candidates and TIS clipped at 2. Fused log probabilities with in-place backward. |
| Memory and kernels | Per-GPU microbatch limits are 122,880 input tokens for the actor and 153,600 for the critic. SGLang static memory fraction 0.92; dense attention uses Triton on Blackwell or FA3 with `--gpu-arch hopper`. Trainer linear attention uses FLA. |
| Evaluation and checkpoints | Evaluate the base and every 15 rollout batches, 89 tasks × 5 trials at temperature 0.7 and top-p 0.95. Save every 5 batches and at the final batch. Keep the latest three complete actor, critic and rollout checkpoints. |

**Launch.** Set the allocation below and run the command for the desired critic.
The full-detach critic reads chronological history with gradients detached at
segment boundaries. The actor-summary critic reads the actor's summarized context.
Reuse the same run ID and output directory to resume that mode.

```bash
TB=(
  --num-nodes "${NUM_NODES:-2}" --num-gpus-per-node "${GPUS_PER_NODE:-8}"
  --tasks-dir "$DATA_DIR/harbor" --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR"
  --prompt-data "$DATA_DIR/harbor/agentpick-csk4gby2/train.jsonl"
  --eval-data "$DATA_DIR/tb21.eval.jsonl"
)
```

```bash
python -m examples.easyppo.terminus_full_detach "${TB[@]}" --run-id tb21-9b-full-detach
```

```bash
python -m examples.easyppo.terminus_actor_summary "${TB[@]}" --run-id tb21-9b-actor-summary
```

Override training capacity with `--trainer-num-nodes`, `--trainer-gpus-per-node`
and `--eval-num-gpus`. Set `--max-tokens-per-gpu` for actor packs, `--critic-max-tokens-per-gpu` for critic packs
and `--async-max-concurrent-samples` for the rollout concurrency cap.
