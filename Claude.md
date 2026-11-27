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

**Mandatory storage cleanup for Terminal Bench-only allocations.** Before
launching or resuming, inspect the shared data mount and confirm no active FCS
or FrontierSmith job uses it. Delete the obsolete prepared FCS/FrontierSmith
task directories (`$DATA_DIR/harbor/frontier-cs` and
`$DATA_DIR/harbor/frontiersmith`) and their obsolete generated JSONL/manifest
files. Do not prepare either dataset for Terminal Bench runs. Do not delete
the active AgentPick or TB2.1 task directories, model files, or checkpoints.
Use `du --inodes --max-depth=3 "$DATA_DIR"` to verify the reclaimed inode
count. If FCS training is needed again, regenerate its prepared data with the
FCS instructions above; do not delete the tracked Frontier-CS source checkout.


Configure Modal credentials on the rollout and evaluation workers. Prepare
both datasets once on the shared mount, starting with a new
`$DATA_DIR/harbor/agentpick-802` directory path.

```bash
python -m pip install 'harbor[modal]==0.24.0'
python scripts/make_miles_agentpick_data.py --published-release --tasks-dir "$DATA_DIR/harbor"
python scripts/make_miles_terminal_data.py --dataset terminal-bench-2.1 \
  --tasks-dir "$DATA_DIR/harbor" --out "$DATA_DIR"
wc -l "$DATA_DIR/harbor/agentpick-802/train.jsonl" "$DATA_DIR/tb21.eval.jsonl"
```

To update an existing dataset, prepare it in a fresh directory and point all workers at that directory. Preparation records the pinned CPU, memory and disk allocations in the task configuration. The current Modal backend enforces CPU and memory and uses its own sandbox filesystem capacity. Preparation sets agent and verifier timeouts to 2,400 and 600 seconds, and preserves task entrypoints and protected-file checks for the Modal runner. The pinned runbook release produces 802 training rows. Each derived environment contains the published Dockerfile and its declared build files. AgentPick runs the agent and verifier as root and scores the fresh reward.txt produced by the task verifier. Harbor allows public egress during verification and retains the task network policy during agent execution. Terminal-Bench 2.1
produces 89 evaluation rows pointing to `harbor/terminal-bench-2-1-tmux`. Each evaluation image extends its official task image with tmux and checks `tmux -V` during the build. Rerun the evaluation preparation command above to update an existing `tb21.eval.jsonl`. Both launchers read these JSONL files.

**Configure.** The two launchers use the same asynchronous PPO recipe.
Each allocated node runs 32 persistent training trial workers and 4 separate
evaluation trial workers, each reserving one Ray CPU. Allocate 64 CPUs per
training node. Dedicated evaluation admits 64 in-flight trials per evaluation
GPU (64 on the default allocation) and each TP1 eval engine runs at most 64 requests.

| Setting | Default |
|---|---|
| GPU allocation | 2 nodes × 8 GPUs. Actor and critic share 6, training rollout uses 9, evaluation uses 1. Trainer TP1, one GPU per rollout or eval engine. |
| Async rollout | 32 prompts × 12 trajectories per batch. The finished-group buffer holds four batches (128 groups). Concurrency cap 1,000 (996 in complete groups of 12), 512 requests per engine. Ray actors handle HTTP POSTs by default (disable with `--no-use-distributed-post`). Weight publication uses `in_place`; maximum weight staleness is 4 updates. |
| Generation | Terminus-2 XML, 32,768-token context, up to 8,192 new tokens per action. Compact at 16,384 context tokens, up to 2,048 new tokens per summary including thinking. Each compaction replaces the actor history with the original task and the visible summary, including an empty summary. Episode budget 102,400 generated tokens including summaries, with one reminder at 51,200. |
| Updates | 150 batches, first 30 critic-only. Four critic updates of 96 trajectories per batch (LR 5e-6, value clip 0.2, value loss coefficient 0.5). One actor update per subsequent batch (LR 3e-6, one warmup step). |
| Critic weighting | Inverse prompt return STD with floor `1 / (2 * sqrt(n))` for binary rewards in {0, 1}. The configured group size n = 12 gives 0.1443375673, including for partial groups. |
| Actor loss | Score centering with PPO advantages, sampled rollout log probabilities, top-128 candidates and TIS clipped at 2. Fused log probabilities with in-place backward. |
| Memory and kernels | Per-GPU microbatch limits are 122,880 input tokens for the actor and 245,760 for the critic. Both training modes check the complete chronological history before batching, including after a resume. A draw above that length is excluded from actor and critic updates; other draws from its prompt remain eligible. W&B records one `critic_overlength/dropped` event per excluded draw and the interval count in `rollout/fully_async/critic_overlength_dropped`. SGLang static memory fraction 0.92; dense attention uses Triton on Blackwell or FA3 with `--gpu-arch hopper`. Trainer linear attention uses FLA. |
| Evaluation and checkpoints | Evaluate the base and every 15 rollout batches, 89 tasks × 5 trials at temperature 0.7 and top-p 0.95. Save every 5 batches and at the final batch. Keep the latest three complete actor, critic and rollout checkpoints. |

Modal retries a transient image startup failure once in a fresh sandbox before agent execution. An unexpected error inside a task's Harbor trial aborts that draw. The fully async collector keeps successful siblings, skips empty groups, and fills each training batch with valid draws. Each evaluation trial with an infrastructure failure scores zero and counts toward the mean reward and pass@k. Model decode errors, authentication failures, and explicit cancellation propagate. W&B updates the running training count `infra/failed_draws` and `infra/by_type/<exception>` when groups with failed draws finish, even while the trainer waits for a batch. It also reports failed draws by exception type under `rollout/fully_async/terminal_infra_failures` (since the last metrics collection) and `eval/tb21/terminal_infra_failures` (per evaluation).

**Trial diagnostics.** Terminus launchers stage each trial under node-local
`/tmp` and hand only completed or torn-down trial directories to one archive
writer per node and role. Writers publish immutable `archives/train` and
`archives/eval` tar.zst shards (128 trial roots or 30 seconds; zstd level 1),
then remove the staged files. Backpressure bounds the completed staging queue;
flush the remaining archives on graceful shutdown or completion of evaluation.
Top-128 candidates stay in Sample and the existing resume checkpoint, not
in diagnostic NPZ files. A hard kill can lose unflushed diagnostic logs but
does not change the checkpoint resume contract. Old persisted trial trees
are not migrated automatically; archive or remove obsolete historical logs
after verifying they are no longer used.

**Launch.** Set the allocation below and run the command for the desired critic.
The full-detach critic reads chronological history with gradients detached at
segment boundaries. The actor-summary critic reads the actor's summarized context.
Reuse the same run ID and output directory to resume that mode.

```bash
TB=(
  --num-nodes "${NUM_NODES:-2}" --num-gpus-per-node "${GPUS_PER_NODE:-8}"
  --tasks-dir "$DATA_DIR/harbor" --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR"
  --prompt-data "$DATA_DIR/harbor/agentpick-802/train.jsonl"
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

To provision the default two B200 nodes through Modal, add `--launch-modal`
to either training command above, plus `--modal-volume NAME --modal-mount "$TB_ROOT"`.
Use the existing volume containing the prepared model and both datasets.
Pass `--modal-sandbox-secret NAME` with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
for Harbor, or export those two variables before launch. The launcher requests
64 CPUs and 1 TiB RAM per node and registers 64 CPUs with each Ray raylet.
