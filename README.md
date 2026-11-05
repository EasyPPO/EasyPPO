<p align="center">
  <img src="assets/easyppo.png" width="200" alt="EasyPPO logo">
</p>

<h2 align="center">EasyPPO — Miles terminal training recipes</h2>

<p align="center">
  <a href="https://easyppo.github.io/">Project</a> ·
  <a href="https://arxiv.org/abs/2609.36802">EasyPPO paper</a>
</p>

This branch trains **Qwen3.5-27B** on terminal tasks with an independent memory
critic. It includes one-step asynchronous Score Centering/TIS training and
synchronous clipped PPO. Each recipe supports `full_detach` and `actor-summary`
critic contexts.

| Launcher | Training → evaluation | Actor objective |
|---|---|---|
| [`run_qwen3_5_27b_async_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py) | FrontierSmith → Frontier-CS | Official Miles Score Centering + TIS |
| [`run_qwen3_5_27b_async_tmax.py`](examples/easyppo/run_qwen3_5_27b_async_tmax.py) | TMAX → Terminal-Bench 2.1 | Official Miles Score Centering + TIS |
| [`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py) | FrontierSmith → Frontier-CS | Synchronous clipped PPO |
| [`run_qwen3_5_27b_sync_tmax.py`](examples/easyppo/run_qwen3_5_27b_sync_tmax.py) | TMAX → Terminal-Bench 2.1 | Synchronous clipped PPO |

The **async default is 64 B200s**: eight nodes, with 16 training GPUs and
48 rollout GPUs. The **sync default is 32 B200s**: four nodes, with 8 training
GPUs and 24 rollout GPUs. Both use separate training and rollout pools; the
actor and critic share the training pool through CPU offload. Each round accepts
**32 prompt groups × 16 episodes = 512 episodes**; the critic makes four updates
and the actor makes one after the initial 30 critic-only rounds.

The sections below are the operating guide. [RECIPES.md](examples/easyppo/RECIPES.md)
describes the objectives, memory, batching, budgets, and checkpoint contracts.

## 1. Set up the Miles GPU environment

Use the [Miles NVIDIA image](docs/getting-started/installation.md),
`radixark/miles:latest`, or an image built from this checkout's
[`docker/Dockerfile`](docker/Dockerfile). Use the same image digest on every
node. The GPU stack includes patched SGLang and Megatron-LM, Transformer Engine,
and CUDA kernels; terminal packages require **Python 3.12+** and
**Transformers 5.12.1**. Training nodes should provision roughly 1.5–2 TB of
host RAM for offloaded models and optimizer state.

The examples use `/shared` at the **same absolute path** on every worker and
Docker daemon host. Clone once on that shared filesystem:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git /shared/EasyPPO
mkdir -p /shared/models /shared/data /shared/runs
```

Install this checkout inside every worker's Miles environment:

```bash
cd /shared/EasyPPO
git submodule update --init third_party/compaction
python -m pip install -e '.[agentic,terminal]'
python -m pip install 'transformers==5.12.1'
python -c "import miles, agentic_critic, transformers; print(miles.__file__); print(agentic_critic.__file__); print(transformers.__version__)"
```

The import paths must resolve to `/shared/EasyPPO`. The extras install Harbor
0.24.0 and mini-swe-agent 1.17.1. Use these extras; Harbor's optional `tinker`
dependency set uses a different Transformers range.

For containerized Miles workers, bind `/shared` to `/shared` and expose the host
Docker daemon through `/var/run/docker.sock`; the worker environment needs the
Docker CLI and Compose plugin. The daemon resolves task/source bind mounts on
its host. Hosts must allow Frontier's privileged judge sidecars. Expose the
site's RDMA devices when using an InfiniBand/RDMA fabric. Provision CPU, RAM,
image storage, and package/image network access for Harbor trials separately
from GPU allocation.

## 2. Join the Ray cluster

Choose the allocation for the launcher you will run: **eight eight-GPU nodes
for async**, or **four eight-GPU nodes for sync**. Start Ray after installation
in each Miles worker environment. Replace the example IPs with the cluster's
reachable data-network addresses. Set `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME` to the site's network interface when needed.

On the head node:

```bash
export MASTER_ADDR=10.0.0.10
ray start --head --node-ip-address "$MASTER_ADDR" --port=6379 \
  --dashboard-host=0.0.0.0 --num-gpus=8 --disable-usage-stats
```

Join the remaining seven async workers or three sync workers, using each
node's own IP:

```bash
export MASTER_ADDR=10.0.0.10
export EASYPPO_NODE_IP=10.0.0.11
ray start --address "$MASTER_ADDR:6379" --node-ip-address "$EASYPPO_NODE_IP" \
  --num-gpus=8 --disable-usage-stats
```

On the head, `ray status` should show eight live nodes / 64 GPUs for the async
allocation, or four live nodes / 32 GPUs for sync. Submit from that allocation's
head worker, with `MASTER_ADDR` still set and `MILES_SCRIPT_EXTERNAL_RAY=1`.

## 3. Download the model and prepare both datasets

Run on an eight-GPU head worker from `/shared/EasyPPO`, in either the 32-GPU
or 64-GPU environment. Prepared model/data assets are shared by all four launchers:

```bash
python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action prepare \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

python examples/easyppo/run_qwen3_5_27b_async_tmax.py \
  --action prepare \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
```

Preparation downloads `Qwen/Qwen3.5-27B` at revision
`fc05daec18b0a78c049392ed2e771dde82bdf654` and converts its Megatron checkpoint.
The HF model is stored in `/shared/models/Qwen3.5-27B`; the converted model is
`/shared/models/Qwen3.5-27B_torch_dist`. Preparation requires the GPU environment
for model conversion. It does not start task containers.

| Dataset | Prepared training input | Full evaluation input |
|---|---|---|
| FrontierSmith | `/shared/data/frontiersmith.train.jsonl` — 190 tasks | `/shared/data/fcs.eval.jsonl` — 179 tasks |
| TMAX | `/shared/data/tmax.train.jsonl` — 14,601 eligible tasks in the current preparation | `/shared/data/tb21.eval.jsonl` — 89 tasks |

Missing Frontier source checkouts are downloaded into
`/shared/data/FrontierSmith` at `ac3046cce7a270e3177cbb41376f36e03de1d518` and
`/shared/data/Frontier-CS-eval` at `59687e5428266a22a49cf51f981b912103bfa91d`.
The preparation removes ten published training aliases and follows the fixed
179-problem Frontier-CS scoring set. TMAX/TB2.1 downloads use pinned Harbor
registry references; training is audited for exact evaluation overlap,
duplicate executable fixtures, and initial-context fit.

Keep the complete `/shared/data/harbor` task tree and **both full Frontier source
checkouts**, not just JSONL files. Generated Compose files bind the corresponding
source by absolute path. Existing task-download directories are reused; use a
fresh preparation directory when regenerating a source-versioned dataset.
Source references and manifests are documented in [ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md).

## 4. Run the task environments and judges

**Harbor starts and stops them automatically for each trial.** Miles starts
SGLang and supplies the in-process Qwen generation callback. No separate Harbor
agent server or global judge HTTP service is required.

For FrontierSmith and Frontier-CS, each generated
`environment/docker-compose.yaml` contains `main` and `judge` services. Harbor
builds/starts the Compose project, runs the terminal agent, invokes the verifier,
and tears down the project. The privileged judge sidecar starts go-judge and its
Node server; the verifier reaches it over private Compose DNS at
`http://judge:8081`.

The agent reads `/app/statement.txt`, writes `/app/solution.cpp`, and can compile
and self-test through bash. The verifier submits the solution, polls its result,
and writes `reward.json` and `judge-result.json` under the trial's verifier logs.
Each sidecar requests 4 GiB of shared memory and configures eight judge workers;
size rollout-host CPU/RAM and Docker image storage for the chosen concurrency.

TMAX and TB2.1 retain each task's supplied image or Dockerfile, auxiliary
services, and `tests/test.sh`. Harbor runs those tests and reads the task reward
after the agent. Initial image builds and some verifier scripts download
packages, so image caching alone may not remove runtime network access.

## 5. Launch training

Run the selected block from the head of its matching allocation. Each block
reuses that allocation sequentially, with a separate `run-id` for every
experiment. All four launchers accept the same context-mode and path options.

**Asynchronous SC/TIS — default eight nodes / 64 GPUs, four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-fs-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_tmax.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-tmax-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
done
```

**Synchronous PPO — default four nodes / 32 GPUs, four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "sync-fs-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_sync_tmax.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "sync-tmax-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
done
```

To run async on a **32-GPU allocation**, add
`--num-nodes 4 --trainer-num-nodes 1`; sync already uses these defaults.
For sync on a 64-GPU allocation, set `--num-nodes 8 --trainer-num-nodes 2`.
Use a matching Ray allocation for the chosen resource settings.
For a short integration run, use a fresh run ID and add:

```bash
--rollout-batch-size 4 --over-sampling-batch-size 5 --samples-per-prompt 2 \
--critic-steps-per-rollout 1 --num-rollout 3 --num-critic-only-steps 1 \
--eval-max-tasks 1 --eval-samples-per-prompt 1
```

The short run keeps the selected launcher's GPU allocation. It exercises task startup,
updates, offload, and weight publication. Training logs are always available in
the Ray job; exporting `WANDB_API_KEY` before submission additionally enables
the launcher's W&B configuration.

## 6. Evaluate and resume

Training performs initial validation, then validation and checkpointing every
ten rounds, including the final round. By default, `--eval-max-tasks 0` selects
the **full validation set** and `--eval-samples-per-prompt 5` runs five independent
episodes per task at temperature/top-p/top-k `1/1/-1`.

| Dataset | Trials per validation point | Trials over 21 points in a fresh 200-round run |
|---|---:|---:|
| Frontier-CS | 179 × 5 = **895** | **18,795** |
| Terminal-Bench 2.1 | 89 × 5 = **445** | **9,345** |

Full-set results use `fcs` or `tb21`. For an explicit quick subset, add
`--eval-max-tasks 16`; selection and order are fixed with seed 42, and result
names become `fcs_quick` or `tb21_quick`. Use `--eval-samples-per-prompt 1` when
one trial per task is sufficient, including smoke runs. Evaluation reward
summaries average numeric returned episodes; five trials per task do not
change training's 16-sample prompt groups.

Outputs for a run are organized as:

```text
/shared/runs/<run-id>/
  checkpoints/          # Megatron actor
  checkpoints_critic/   # Native HF memory critic, optimizer and RNG state
  trials/               # Harbor trajectories, candidates and verifier results
```

Resume with the **same launcher, dataset, context mode, run ID, and trainer
topology**, loading the matching critic checkpoint explicitly:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action train --critic-context-mode full_detach \
  --run-id async-fs-full_detach \
  --critic-load /shared/runs/async-fs-full_detach/checkpoints_critic \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
```

The actor checkpoint is selected from that run's `checkpoints` directory. To
start a **new** run from critic weights alone, choose a new run ID and add
`--critic-load <source-run>/checkpoints_critic --critic-weights-only`. This starts
fresh optimizer/step state. The same resume and warm-start options apply to the
other launchers.

## Recipe reference and validation

[RECIPES.md](examples/easyppo/RECIPES.md) gives the current training mechanics and
key parameters. [ASYNC_PPO.md](examples/easyppo/ASYNC_PPO.md),
[PPO_SCORE_CENTERING.md](examples/easyppo/PPO_SCORE_CENTERING.md), and
[ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md) cover the async interface, official
SC integration, and data preparation. Each launcher exposes its full options
through `--help`.

CPU regression covers memory/value alignment, critic scheduling, optimizer and
checkpoint behavior, terminal cancellation, data conversion, and the official
SC probability/loss pipeline. Full 27B training, Docker judge execution on the
target cluster, B200 long-sequence capacity, and throughput require cluster
validation; the resource and token budgets are configured targets.
