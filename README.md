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
| [`run_qwen3_5_27b_async_ppo.py`](examples/easyppo/run_qwen3_5_27b_async_ppo.py) | `--dataset frontiersmith` or `--dataset tmax` | Official Miles Score Centering + TIS |
| [`run_qwen3_5_27b_sync_frontiersmith.py`](examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py) | FrontierSmith → Frontier-CS | Synchronous clipped PPO |
| [`run_qwen3_5_27b_sync_tmax.py`](examples/easyppo/run_qwen3_5_27b_sync_tmax.py) | TMAX → Terminal-Bench 2.1 | Synchronous clipped PPO |

Default allocation: **64 B200s**, with 16 training GPUs and 48 rollout GPUs.
A 32-GPU allocation uses 8 training GPUs and 24 rollout GPUs. The actor and
critic share the training pool through CPU offload. Each round accepts
**32 prompt groups × 16 episodes = 512 episodes**; the critic makes four updates
and the actor makes one after the initial 30 critic-only rounds.

The sections below are the operating guide. [RECIPES.md](examples/easyppo/RECIPES.md)
describes the objectives, memory, batching, budgets, and checkpoint contracts.

## 1. Set up the Miles GPU environment

Use Linux GPU hosts with NVIDIA drivers, NVIDIA Container Toolkit, Docker
Engine, the `docker compose` plugin, and a high-bandwidth interconnect. The
terminal packages require **Python 3.12+**. Training nodes should provision
roughly 1.5–2 TB of host RAM for offloaded models and optimizer state.

Use the [Miles NVIDIA image](docs/getting-started/installation.md), which includes
its patched SGLang, Megatron-LM, Transformer Engine, and CUDA kernels. Pull the
same image on every node and record its digest for the run:

```bash
docker pull radixark/miles:latest
docker image inspect radixark/miles:latest --format '{{index .RepoDigests 0}}'
```

The following layout assumes `/shared` is available at the **same absolute path**
on every host. Clone once on that shared filesystem:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git /shared/EasyPPO
mkdir -p /shared/models /shared/data /shared/runs
```

On **each node**, start a Miles container. Mount the shared directory at the
same path and expose the host Docker daemon so Harbor can start task sandboxes:

```bash
docker run -d --name easyppo-miles \
  --gpus all --network=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount type=bind,source=/shared,target=/shared \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  --workdir /shared/EasyPPO \
  radixark/miles:latest sleep infinity

docker exec -it easyppo-miles bash
```

For an InfiniBand/RDMA fabric, also expose the site's RDMA devices to the
container, for example with `--device=/dev/infiniband`.

Run the remaining setup inside **every** training/rollout container:

```bash
cd /shared/EasyPPO
git submodule update --init third_party/compaction
python -m pip install -e '.[agentic,terminal]'
python -m pip install 'transformers==5.12.1'
python -c "import miles, agentic_critic, transformers; print(miles.__file__); print(agentic_critic.__file__); print(transformers.__version__)"
nvidia-smi
```

The import paths must resolve to `/shared/EasyPPO`. The extras install Harbor
0.24.0 and mini-swe-agent 1.17.1. Install the named extras above; Harbor's optional
`tinker` dependency set uses a different Transformers range.

The container also needs a Docker CLI and Compose plugin. Check both the client
and host daemon connection:

```bash
docker version
docker compose version
```

If the client packages are absent, install them inside the Ubuntu-based Miles
container using [Docker's package repository](https://docs.docker.com/engine/install/ubuntu/):

```bash
apt-get update
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
  "$(dpkg --print-architecture)" "${UBUNTU_CODENAME:-$VERSION_CODENAME}" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce-cli docker-buildx-plugin docker-compose-plugin
```

This uses the mounted host daemon; a second Docker daemon is not needed inside
the training container. Hosts must allow the privileged judge containers used
by FrontierSmith/Frontier-CS. Provide CPU, RAM, disk, and outbound package/image
access for task containers independently of GPU allocation.

## 2. Join the Ray cluster

Start Ray **after installation** in those containers. Replace the example IPs
with the cluster's reachable data-network addresses. Set `NCCL_SOCKET_IFNAME`
and `GLOO_SOCKET_IFNAME` to the site's network interface when needed.

On the head node:

```bash
export MASTER_ADDR=10.0.0.10
ray start --head --node-ip-address "$MASTER_ADDR" --port=6379 \
  --dashboard-host=0.0.0.0 --num-gpus=8 --disable-usage-stats
```

On each of the other seven nodes, using that node's own IP:

```bash
export MASTER_ADDR=10.0.0.10
export EASYPPO_NODE_IP=10.0.0.11
ray start --address "$MASTER_ADDR:6379" --node-ip-address "$EASYPPO_NODE_IP" \
  --num-gpus=8 --disable-usage-stats
```

Back on the head, `ray status` should show eight live nodes and 64 GPUs.
For the 32-GPU allocation, join four eight-GPU nodes instead. Submit all launchers
from the head container, with `MASTER_ADDR` still set and
`MILES_SCRIPT_EXTERNAL_RAY=1` to use this prejoined cluster.

## 3. Download the model and prepare both datasets

Run on the head container from `/shared/EasyPPO`:

```bash
python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action prepare --dataset frontiersmith \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action prepare --dataset tmax \
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

Run one experiment at a time on the shared allocation. Each command writes to
its own `run-id` directory. Both context modes are accepted by all three
launchers.

**Asynchronous SC/TIS — four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
    --action train --dataset frontiersmith --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-fs-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs

  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
    --action train --dataset tmax --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-tmax-$MEMORY_MODE" \
    --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
done
```

**Synchronous PPO — the two dataset-specific launchers:**

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

For **32 GPUs**, add `--num-nodes 4 --trainer-num-nodes 1` to the selected command.
For a short integration run, use a fresh run ID and add:

```bash
--rollout-batch-size 4 --over-sampling-batch-size 5 --samples-per-prompt 2 \
--critic-steps-per-rollout 1 --num-rollout 3 --num-critic-only-steps 1 \
--eval-max-tasks 1
```

The short run keeps the requested GPU allocation. It exercises task startup,
updates, offload, and weight publication. Training logs are always available in
the Ray job; exporting `WANDB_API_KEY` before submission additionally enables
the launcher's W&B configuration.

## 6. Evaluate and resume

Training performs initial validation, then validation and checkpointing every
ten rounds, including the final round. Validation uses the same seed-42 subset
of 16 tasks each time, one episode per task, temperature/top-p/top-k `1/1/-1`.
A fresh 120-round run schedules 13 validation points: **208 trials**. Results use
`fcs_quick` or `tb21_quick`. Add `--eval-max-tasks 0` to evaluate all 179 FCS or
89 TB2.1 tasks at each validation point.

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
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/easyppo/run_qwen3_5_27b_async_ppo.py \
  --action train --dataset frontiersmith --critic-context-mode full_detach \
  --run-id async-fs-full_detach \
  --critic-load /shared/runs/async-fs-full_detach/checkpoints_critic \
  --model-dir /shared/models --data-dir /shared/data --output-dir /shared/runs
```

The actor checkpoint is selected from that run's `checkpoints` directory. To
start a **new** run from critic weights alone, choose a new run ID and add
`--critic-load <source-run>/checkpoints_critic --critic-weights-only`. This starts
fresh optimizer/step state. The same resume and warm-start options apply to the
two synchronous launchers.

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
