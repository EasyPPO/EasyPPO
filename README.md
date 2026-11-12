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

The training examples use `/shared` at the **same absolute path** on every worker and
task Docker daemon host. The independent judge needs only its own checkout. Clone once on that shared filesystem:

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
its host. The shared judge host must allow a privileged judge container. Expose the
site's RDMA devices when using an InfiniBand/RDMA fabric. Provision CPU, RAM,
image storage, and package/image network access for Harbor trials separately
from GPU allocation.

## 2. Configure Ray startup

The launchers start the Ray head and join workers over SSH automatically. Use
**eight eight-GPU nodes for async**, or **four eight-GPU nodes for sync**.
On the head, write an MPI-style hostfile with one node per line. For sync:

```bash
cat > /shared/ray-hostfile <<'EOF'
10.0.0.10 slots=8
10.0.0.11 slots=8
10.0.0.12 slots=8
10.0.0.13 slots=8
EOF
export MASTER_ADDR=10.0.0.10
```

For async, list all eight allocated nodes. Pass `--ray-hostfile /shared/ray-hostfile`
when launching; its default is `/root/mpi_rack_hostfile`. Include the head using
exactly `MASTER_ADDR`, or set `MLP_WORKER_0_HOST` to its hostfile name so it is skipped.
Do not put the independent CPU judge host in this file.

The head must have passwordless `ssh root@<worker>` access, and the remote SSH
shell must run `ray` from the same configured Miles environment. With containers,
SSH must reach the worker container environment. All workers need the prepared
code/model/data paths. Configure `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` for
the site's network when needed. Automatic startup cleans old Miles/SGLang/Ray
processes on the listed nodes; use nodes allocated exclusively to this run.

Submit from the head with `MASTER_ADDR` set. No manual `ray start` is needed.
If the platform already provides a joined Ray cluster, set
`MILES_SCRIPT_EXTERNAL_RAY=1`; this skips head startup and SSH joining.
`--no-join-ray-workers` disables only SSH joining.

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
| FrontierSmith | `/shared/data/frontiersmith.train.jsonl` — 200 tasks | `/shared/data/fcs.eval.jsonl` — 180 tasks |
| TMAX | `/shared/data/tmax.train.jsonl` — 14,601 eligible tasks in the current preparation | `/shared/data/tb21.eval.jsonl` — 89 tasks |

The Frontier judge and both problem corpora are bundled under
[`Frontier-CS/`](Frontier-CS/README.md). Data preparation uses these local files;
it does not clone additional repositories. All 200 Smith and 180 FCS problems are used.
The ten published cross-corpus aliases are retained and recorded in the manifest.
Keep `/shared/data/harbor` available at the same path on every task Docker host.
See [ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md) for source versions and manifests.

## 4. Run the task environments and judges

On the **independent CPU judge host**, clone this branch and start the bundled
Frontier-CS service from its own directory:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git EasyPPO-judge
cd EasyPPO-judge/Frontier-CS/algorithmic
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
export JUDGE_WORKERS=8 GJ_PARALLELISM=8
docker compose up -d --build
curl --fail --retry 10 --retry-connrefused --retry-delay 2 http://127.0.0.1:8081/problems
```

The judge needs Git, Docker Engine and Compose, with privileged containers enabled.
It does not need Miles, CUDA, Ray, model weights or the training cluster's shared
filesystem. Compose mounts one merged `problems/` directory, plus local `data/`
and `submissions/` directories. The checkout can live at any path.

All FrontierSmith and Frontier-CS trials share this service. Set
`--frontier-judge-url` during preparation to a URL reachable from every task
container, such as `http://10.0.0.20:8081`. `FRONTIERCS_JUDGE_BIND_IP` controls
which local interface the judge listens on; it is not the client's target URL.
The upstream API has no authentication; expose port 8081 on a trusted network.

The merged bank contains 200 Smith and 180 FCS problems; the training and evaluation
JSONL files include all 200 and 180 respectively. Each corpus retains its pinned checkers
and test data. Checker/interactor scores are scaled from `[0,100]` to `[0,1]`.
Harbor starts only each trial's `main` sandbox. Its verifier submits
`/app/solution.cpp` to the shared judge, then writes `reward.json` and
`judge-result.json`. Trial teardown does not stop the judge.

`--frontier-judge-timeout` defaults to 3600 seconds including queue wait; the
Harbor verifier deadline adds 90 seconds. Rerun preparation when changing the
judge URL or timeout. The single-host URL default is
`http://host.docker.internal:8081`; use the actual remote judge URL for a
separate machine. The rollout executor uses its local Docker daemon. For a
conservative initial concurrency, use `--sglang-server-concurrency 1` (six
concurrent trials with the default sync allocation).

From the judge's `Frontier-CS/algorithmic` directory, use `docker compose logs -f`
to view logs and `docker compose down` to stop the service. Results remain in
`data/` and `submissions/`.

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
  python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
    --action train --ray-hostfile /shared/ray-hostfile --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-fs-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

  python examples/easyppo/run_qwen3_5_27b_async_tmax.py \
    --action train --ray-hostfile /shared/ray-hostfile --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-tmax-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
done
```

**Synchronous PPO — default four nodes / 32 GPUs, four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
    --action train --ray-hostfile /shared/ray-hostfile --critic-context-mode "$MEMORY_MODE" \
    --run-id "sync-fs-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

  python examples/easyppo/run_qwen3_5_27b_sync_tmax.py \
    --action train --ray-hostfile /shared/ray-hostfile --critic-context-mode "$MEMORY_MODE" \
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
| Frontier-CS | 180 × 5 = **900** | **18,900** |
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
python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action train --ray-hostfile /shared/ray-hostfile --critic-context-mode full_detach \
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
