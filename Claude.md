# Qwen3.5-27B Synchronous PPO on FrontierSmith

Entry point: [`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py).
Adapt the commands below to the cluster's image, scheduler, storage, and service configuration.

## 1. Environment

Use a Miles-compatible GPU image and this branch's code. Follow the
[installation guide](docs/getting-started/installation.md) when building an image.
For a fresh checkout:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git EasyPPO
cd EasyPPO
```

If the job image needs the project or its terminal dependencies, install:

```bash
python -m pip install -e '.[agentic,terminal]'
```

The default allocation is **4 nodes × 8 B200 GPUs**. Rollout, critic training,
and actor training alternate on the same 32 GPUs with offload enabled.
Training uses TP4/DP8; rollout uses eight TP4 engines. Set `--num-nodes` and
`--num-gpus-per-node` to match the allocation, and provision host memory for offload.

Set these variables for the commands below:

| Variable | Value |
|---|---|
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
  --run-id sync-fs-full-detach-001 \
  --critic-context-mode full_detach \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

Set `--sglang-server-concurrency` to fit the Harbor host and judge capacity.
The default is 128 per engine; a value of 1 allows up to eight concurrent trials
with the default allocation.

Training runs for 200 rounds, each with 32 prompt groups × 16 episodes.
The first 30 rounds update the critic; subsequent rounds perform four critic
updates and one actor update. Evaluation runs initially, every ten rounds,
and at the end, with 180 tasks × 5 trials per evaluation.

Outputs are under `$RUN_DIR/<run-id>/`: `checkpoints/`, `checkpoints_critic/`,
and `trials/`. Resume with the same run ID, context mode, topology, and training
parameters. Use a new run ID for a new experiment, including a switch to
`--critic-context-mode actor-summary`.
