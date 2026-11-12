# Qwen3.5-27B Synchronous PPO on FrontierSmith

The main objective is to test whether, in synchronous agentic PPO, giving the
critic access to the full episode history improves value estimation compared
with the current approach of managing critic context through actor-generated
summaries. We then evaluate whether any improvement in critic estimates
translates into more effective end-to-end PPO training and stronger final
policy performance.

Entry point: [`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py).
Adapt the commands below to the cluster's image, scheduler, storage, and service configuration.

## 1. Environment

Use a Miles-compatible GPU image and this branch's code. Follow the
[installation guide](docs/getting-started/installation.md) when building an image.
For a fresh checkout:

```bash
git clone --branch miles https://github.com/EasyPPO/EasyPPO.git EasyPPO
cd EasyPPO
```

To update an existing checkout from **EasyPPO/EasyPPO, branch `miles`**:

```bash
git remote set-url origin https://github.com/EasyPPO/EasyPPO.git
git switch miles
git pull --ff-only origin miles
```

If the job image needs the project or its terminal dependencies, install:

```bash
python -m pip install -e '.[agentic,terminal]'
```

Choose the GPU allocation to fit the available hardware and memory budget.
Set `--num-nodes` and `--num-gpus-per-node` to match the cluster allocation.
Rollout, critic training, and actor training alternate on all allocated GPUs
with offload enabled. Provision host memory for offload.

Training and rollout default to TP4. With `N` total GPUs, this gives training
DP `N / 4` and `N / 4` rollout engines. Adjust `--tensor-model-parallel-size`
and `--rollout-gpus-per-engine` for the hardware, keeping the GPU count divisible
by each parallel group size and the training batches divisible by training DP.

Set these variables for the commands below:

| Variable | Value |
|---|---|
| `NUM_NODES` | Number of allocated training nodes |
| `GPUS_PER_NODE` | GPUs allocated per node |
| `MODEL_DIR` | Model and converted checkpoint directory |
| `DATA_DIR` | Prepared Harbor tasks and datasets |
| `RUN_DIR` | Training outputs |
| `JUDGE_URL` | Judge service URL reachable from Harbor task containers |

Make the code and assets available to their execution processes through the
platform's storage or distribution mechanism. Give Harbor access to Docker;
task bind-mount source paths must exist on the corresponding daemon host.
Set `--megatron-path` when using a custom Megatron installation path.

## 2. Judge deployment

Use one shared judge service for all Harbor trials. The bundled
`Frontier-CS/algorithmic/` contains the judge and both corpora: **200 FrontierSmith
training tasks and 180 FCS evaluation tasks**. The manifest records ten known
cross-corpus overlaps retained in these sets.

Deploy on a CPU service host using the bundled Compose configuration:

```bash
cd Frontier-CS/algorithmic
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
docker compose up -d --build
curl --fail http://127.0.0.1:8081/problems
```

For another service orchestrator, preserve the Compose image, privileged-container
setting, and mounts for `problems/`, `data/`, and `submissions/`. Size
`JUDGE_WORKERS` and `GJ_PARALLELISM` for the available CPU and memory.

Set `JUDGE_URL` to the service's reachable DNS name or address.
`FRONTIERCS_JUDGE_BIND_IP` controls the server's local listening interface.
Keep the unauthenticated API behind the platform's internal network or access controls.

## 3. Model preparation

Run from the repository root in an allocated GPU job environment:

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action prepare \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --frontier-judge-url "$JUDGE_URL" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

This prepares the pinned `Qwen/Qwen3.5-27B` model, Harbor tasks, and JSONL datasets,
then converts the Megatron checkpoint. Reuse compatible existing artifacts:
`$MODEL_DIR/Qwen3.5-27B` and `$MODEL_DIR/Qwen3.5-27B_torch_dist`.
Checkpoint conversion requires GPUs. Regenerate task data after changing the
judge URL or `--frontier-judge-timeout`, which are embedded in the task configuration.

## 4. Training

Submit through the platform's job environment. For platform-managed Ray:

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
# Set RAY_ADDRESS to the platform's Ray Jobs HTTP endpoint.
# Set MASTER_ADDR to the head address reachable by training workers.
```

Run the launcher inside this experiment's job/container environment: the current
Ray backend cleans old training processes in the submission environment.
For SSH-based bootstrap or another deployment backend, follow the
[cluster integration guide](README.md#2-cluster-integration).

To enable W&B, set `WANDB_API_KEY` in the job environment before running the
launcher. When resuming a run, append
`--extra-env-vars 'WANDB_RUN_ID=<wandb-run-id> WANDB_RESUME=must'` to the training
command below. Use the ID after `/runs/` in the original W&B run URL and keep the
same W&B entity and project. The launcher uses `--run-id` as the display name;
the W&B run ID is what attaches new metrics to the existing run.

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --num-rollout 100 \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

Set `--sglang-server-concurrency` to fit the Harbor host and judge capacity.
The default is 128 per engine; a value of 1 allows one concurrent trial per engine.

Actor generation uses a 131072-token context window, up to 24576 tokens per
action and 3072 per summary, with compaction at 12000 context tokens. Thinking
counts toward these limits. Length-limited turns increment `truncated_turns`
and continue; the cumulative episode generation budget is 120000 tokens.

Training runs for 100 rounds, each with 32 prompt groups × 16 episodes.
Every round performs four critic updates, including the first 30 critic-only
rounds. Rounds 31–100 also perform one actor update. Evaluation runs initially,
every ten rounds, and at the end, with 180 tasks × 5 trials per evaluation.

Outputs are under `$RUN_DIR/<run-id>/`: `checkpoints/`, `checkpoints_critic/`,
and `trials/`. Use a new run ID for each new experiment. See [section 6](#6-resume-training)
to resume an existing run.

## 5. Primary experiment: full_detach vs. actor-summary

Our current primary focus is comparing **`full_detach` and `actor-summary`**
to measure how critic access to episode history affects synchronous PPO training.

| Critic mode | Context | Example run ID |
|---|---|---|
| `full_detach` | Full chronological episode history, with gradients detached across segment boundaries | `sync-fs-full-detach-001` |
| `actor-summary` | The actor's context for each segment: task, current summary, and local conversation | `sync-fs-actor-summary-001` |

Complete model/data preparation in sections 1–3 and configure the job/Ray
environment in section 4. Use the same `NUM_NODES`, `GPUS_PER_NODE`, `MODEL_DIR`,
`DATA_DIR`, and `RUN_DIR` for both commands below. Run each in its allocated job
environment with a fresh run ID.

**Full-detach critic:**

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --num-rollout 100 \
  --num-critic-only-steps 30 \
  --critic-steps-per-rollout 4 \
  --rollout-batch-size 32 --samples-per-prompt 16 \
  --eval-interval 10 --eval-max-tasks 0 --eval-samples-per-prompt 5 \
  --save-interval 10 \
  --max-context-tokens 131072 \
  --max-new-tokens 24576 --summary-tokens 3072 \
  --compact-threshold 12000 --max-total-tokens 120000 \
  --continue-on-length \
  --tensor-model-parallel-size 4 --rollout-gpus-per-engine 4
```

**Actor-summary critic:**

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --run-id sync-fs-actor-summary-001 \
  --critic-context-mode actor-summary \
  --num-rollout 100 \
  --num-critic-only-steps 30 \
  --critic-steps-per-rollout 4 \
  --rollout-batch-size 32 --samples-per-prompt 16 \
  --eval-interval 10 --eval-max-tasks 0 --eval-samples-per-prompt 5 \
  --save-interval 10 \
  --max-context-tokens 131072 \
  --max-new-tokens 24576 --summary-tokens 3072 \
  --compact-threshold 12000 --max-total-tokens 120000 \
  --continue-on-length \
  --tensor-model-parallel-size 4 --rollout-gpus-per-engine 4
```

A step here means one rollout round of 512 episodes. Each round has four critic
optimizer updates, each on 128 episodes. The first 30 rounds therefore perform
120 critic updates and zero actor updates. Rounds 31–100 also update the actor
once per round. Eval runs before training and after rounds 10, 20, …, 100, using
all 180 FCS tasks with five trials each (`--eval-max-tasks 0` selects all tasks).

Keep the initial model, 200-task training set, resource allocation, and other
hyperparameters identical. Start both runs from the same base checkpoint;
their run IDs select separate output directories. Compare training curves and
full Frontier-CS evaluation rewards at matched rounds.

## 6. Resume training

Resubmit the original `--action train` command from section 5 through the job
environment in section 4. Use the same `RUN_DIR`, run ID, critic context mode,
trainer topology, and training parameters. For the 100-round runs above, keep
`--num-rollout 100`. This sets the total number of rounds for the run.

The launcher loads the actor and critic from `checkpoints/` and
`checkpoints_critic/` under the run directory. Megatron restores their model,
optimizer, scheduler, and RNG state. Both checkpoints must point to the same
rollout ID. For example, after round 20 of `sync-fs-full-detach-001`, check:

```bash
run="$RUN_DIR/sync-fs-full-detach-001"
cat "$run/checkpoints/latest_checkpointed_iteration.txt"
cat "$run/checkpoints_critic/latest_checkpointed_iteration.txt"
ls "$run/checkpoints/rollout/global_dataset_state_dict_19.pt"
```

Both trackers should read `19` in this example because rollout IDs start at
zero. The dataset state file restores the prompt sampling position and epoch.
If the file is missing, prompt sampling restarts from the beginning of the
dataset. With the settings above, the launcher first evaluates the restored
policy, then starts round 21 at rollout ID `20`. The critic-only phase ends
after round 30 as scheduled.
