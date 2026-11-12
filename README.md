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
**Transformers 5.12.1**. Size training-host memory for the offload
configuration and the observed workload.

The examples use `/shared` at the **same absolute path** on every worker and
Docker daemon host. Clone once on that shared filesystem:

```bash
git clone --branch miles https://github.com/EasyPPO/EasyPPO.git /shared/EasyPPO
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
its host. The shared judge host must allow a privileged judge container. Expose the
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
  --frontier-judge-url http://10.0.0.20:8081 \
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

Keep the complete `/shared/data/harbor` task tree on every Docker worker host.
Keep **both full Frontier source checkouts** and this repository at their prepared
absolute paths on the shared judge host. `frontier-judge.compose.yaml` binds each
selected problem from its pinned source read-only. Existing task-download directories
are reused; rerun preparation to update their verifier URL and remove judge sidecars.
Use a fresh preparation directory when changing dataset source versions.
Source references and manifests are documented in [ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md).

## 4. Run the task environments and judges

**Start one shared Frontier judge before training.** All FrontierSmith and
Frontier-CS trials submit to that service. On the CPU judge host (the example
uses private address `10.0.0.20`), run:

```bash
export FRONTIER_JUDGE_BIND_IP=10.0.0.20
export FRONTIER_JUDGE_PORT=8081
export JUDGE_WORKERS=8 GJ_PARALLELISM=8
docker compose -p easyppo-frontier-judge \
  -f /shared/data/frontier-judge.compose.yaml up -d --build --wait
curl --fail http://10.0.0.20:8081/problems
```

The endpoint must be reachable **from every task container**. Use a trusted
private network: the upstream HTTP API has no authentication. Set
`--frontier-judge-url` during preparation to that endpoint; this option configures
generated tasks, not an already prepared dataset. Its single-host default is
`http://host.docker.internal:8081` with a Linux host-gateway mapping. On multiple
Docker hosts, use the actual judge host address. The Compose port defaults to
8081 on all interfaces; set `FRONTIER_JUDGE_BIND_IP` to the private interface.

The service uses the pinned Frontier-CS judge implementation for both corpora,
with their own pinned checkers/test data and unchanged problem IDs. Scores come
from checker/interactor output and are scaled from `[0,100]` to `[0,1]`. Its
queue, worker pool and compiled-checker cache are shared across trials.

Harbor starts only the `main` agent sandbox for each trial, runs its verifier,
and tears down that trial. Trial cleanup does not stop the shared judge. Miles
starts SGLang and supplies the Qwen generation callback; no Harbor agent server
is required.

The agent reads `/app/statement.txt`, writes `/app/solution.cpp`, and can compile
and self-test through bash. The verifier submits the solution, polls its result,
and writes `reward.json` and `judge-result.json` under the trial's verifier logs.
The single judge has a 4 GiB shared-memory limit and eight workers by default.
`--frontier-judge-timeout` defaults to 3600 seconds including queue wait; Harbor's
verifier deadline adds 90 seconds for readiness and result handling. Tune the
worker pool and timeout together. Data and submissions persist in Compose volumes.
Stop the service separately after all runs:

```bash
docker compose -p easyppo-frontier-judge \
  -f /shared/data/frontier-judge.compose.yaml down
```

The rollout executor starts local Docker trials; SGLang engine count does not
distribute these sandboxes across Docker hosts. For an initial run, add
`--sglang-server-concurrency 1` to limit the default sync allocation to six
concurrent trials. Increase this after measuring sandbox and judge capacity.

To check the sandbox-to-judge path on CPU with Harbor's Codex agent, run the data
preparation command below first (no model download or checkpoint conversion).
Then start the shared judge as above before running `harbor run`:

```bash
python scripts/make_miles_frontiersmith_data.py \
  --source-dir /shared/data/FrontierSmith \
  --frontier-cs-source-dir /shared/data/Frontier-CS-eval \
  --tasks-dir /shared/data/harbor --out /shared/data \
  --judge-url http://10.0.0.20:8081

# Use your configured Codex credentials and model; no local model or GPU is used.
CODEX_AUTH_JSON_PATH="$HOME/.codex/auth.json" harbor run \
  --path /shared/data/harbor/frontiersmith/frontier-cs-algorithm-frontiersmith_0 \
  --agent codex --model "$CODEX_MODEL" --n-attempts 2 --n-concurrent 2 \
  --jobs-dir /shared/runs/harbor-codex-check
```

Set `CODEX_MODEL` to your available model. Confirm both trial rewards are numeric
and the shared judge remains healthy after Harbor removes the trial containers.

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

Both actor and critic use TP4 with PP=CP=1, giving DP2 on 8 training
GPUs and DP4 on 16. The launchers enable sequence parallelism, full-layer
activation recomputation, and the CPU optimizer options
`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`.

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
  checkpoints/          # Native Megatron actor checkpoint
  checkpoints_critic/   # Native Megatron critic checkpoint
  trials/               # Harbor trajectories, candidates and verifier results
```

Resume with the **same launcher, dataset, context mode, run ID, and trainer
topology**:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action train --critic-context-mode full_detach \
  --run-id async-fs-full_detach \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
```

The actor and critic automatically load from that run's `checkpoints` and
`checkpoints_critic` directories through the native checkpoint loader. A fresh
run initializes both backbones from the converted reference checkpoint when
its checkpoint directories have no saved iteration. Checkpoints include native
model, optimizer, scheduler, and RNG state.

To start a **new** run from critic weights alone, choose a new run ID and add
`--critic-load <source-run>/checkpoints_critic --critic-weights-only`. This starts
fresh optimizer/step state through native finetune loading. The same resume
and warm-start options apply to the other launchers.

## Recipe reference and validation

[RECIPES.md](examples/easyppo/RECIPES.md) gives the current training mechanics and
key parameters. [ASYNC_PPO.md](examples/easyppo/ASYNC_PPO.md),
[PPO_SCORE_CENTERING.md](examples/easyppo/PPO_SCORE_CENTERING.md), and
[ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md) cover the async interface, official
SC integration, and data preparation. Each launcher exposes its full options
through `--help`.

Episode conversion supplies critic boundaries, value alignment, prompt-group
schedules, and inverse-STD loss weights; Miles/Megatron own the training
lifecycle, value loss, metrics, and persistence. The Qwen3.5 memory extension
uses the existing Transformer Engine attention and FLA convolution/recurrent
kernels. See [RECIPES.md](examples/easyppo/RECIPES.md#critic-memory-and-token-alignment)
for the detach and reset boundaries.

Full 27B GPU training, native TE/FLA execution and checkpoint reload, offload
capacity and timing, target-cluster judges, and throughput still require
validation on the target GPU cluster.
