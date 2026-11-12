# Qwen3.5-27B Synchronous PPO on FrontierSmith

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

```bash
python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
  --action train \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

Set `--sglang-server-concurrency` to fit the Harbor host and judge capacity.
The default is 128 per engine; a value of 1 allows one concurrent trial per engine.

Training runs for 200 rounds, each with 32 prompt groups × 16 episodes.
The first 30 rounds update the critic; subsequent rounds perform four critic
updates and one actor update. Evaluation runs initially, every ten rounds,
and at the end, with 180 tasks × 5 trials per evaluation.

Outputs are under `$RUN_DIR/<run-id>/`: `checkpoints/`, `checkpoints_critic/`,
and `trials/`. Resume with the same run ID, context mode, topology, and training
parameters. Use a new run ID for each new experiment.

## 5. Primary experiment: full_detach vs. actor-summary

Our current primary focus is comparing **`full_detach` and `actor-summary`**
to measure how critic access to episode history affects synchronous PPO training.

| Critic mode | Context | Example run ID |
|---|---|---|
| `full_detach` | Full chronological episode history, with gradients detached across segment boundaries | `sync-fs-full-detach-001` |
| `actor-summary` | The actor's context for each segment: task, current summary, and local conversation | `sync-fs-actor-summary-001` |

Run the training command in section 4 once per mode, setting
`--critic-context-mode` and `--run-id` from the table. Keep the initial model,
200-task training set, 180-task evaluation set, training hyperparameters,
resource allocation, and evaluation protocol identical. Start each run from
the same base checkpoint and use its own output directory through the run ID.
Compare the training curves and full Frontier-CS evaluation rewards at matched rounds.
