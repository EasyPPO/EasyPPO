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
48 rollout GPUs in separate pools. The **sync default is 32 B200s**: four nodes
sharing all 32 GPUs between SGLang rollout, critic training, and actor training
in sequence, with `--colocate --offload-rollout --offload-train`. In async,
the actor and critic alternate on the 16-GPU training pool through CPU offload.
Each round accepts
**32 prompt groups × 16 episodes = 512 episodes**; the critic makes four updates
and the actor makes one after the initial 30 critic-only rounds.

The sections below are the operating guide. [RECIPES.md](examples/easyppo/RECIPES.md)
describes the objectives, memory, batching, budgets, and checkpoint contracts.

## 1. Environment

Use the cluster's Miles-compatible GPU image and job environment; see
[installation.md](docs/getting-started/installation.md) for image construction.
Do not reinstall components already provided by that image. If this checkout
and its terminal dependencies are missing, install them in the job image:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git EasyPPO
cd EasyPPO
python -m pip install -e '.[agentic,terminal]'
```

The current terminal recipes do not require `third_party/compaction`; their
text summaries are implemented in the terminal agent. The submodule is only
needed for the separate KV compression experiments.

Set `MODEL_DIR`, `DATA_DIR`, and `RUN_DIR` to the site's model, prepared-data,
and output locations. Provide the code and assets to the workers through the
platform's storage or distribution mechanism; no particular filesystem,
mount point, or deployment user is required. Task bind-mount source paths
must be accessible to the Docker daemon that runs Harbor tasks. Configure
that daemon access and offload CPU/RAM capacity through the platform.
Use `--megatron-path` if Megatron lives outside the launcher's default location.

## 2. Cluster integration

Use the scheduler's allocated job environment and existing cluster configuration.
For a platform-managed Ray cluster:

```bash
export MILES_SCRIPT_EXTERNAL_RAY=1
# RAY_ADDRESS: the platform's Ray Jobs HTTP endpoint (head-local default: port 8265).
# MASTER_ADDR: the head address reachable by training workers; retain the platform setting.
```

This skips Ray startup, shutdown, and SSH joining. The current Ray backend
still cleans old training processes in the submission environment, so run
inside this experiment's job/container environment. Make the checkout and
prepared assets available to the execution processes.

Only when the platform does not manage Ray, the existing SSH bootstrap can
be used: leave `MILES_SCRIPT_EXTERNAL_RAY` unset, supply `MASTER_ADDR` and
`--ray-hostfile`, and ensure the allocated environment supports the helper's
root SSH access. This is an optional deployment route, not a training prerequisite.
Use the corresponding Miles backend configuration for other cluster platforms.
Match `--num-nodes` and `--num-gpus-per-node` to the actual allocation;
defaults are four eight-GPU nodes for sync and eight eight-GPU nodes for async.

## 3. Download the model and prepare both datasets

Run from the checkout in an allocated GPU job environment. Set `JUDGE_URL`
to the service deployed in section 4. Prepared assets can be reused by all four launchers:

```bash
python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action prepare \
  --frontier-judge-url "$JUDGE_URL" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"

python examples/easyppo/run_qwen3_5_27b_async_tmax.py \
  --action prepare \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

Preparation downloads `Qwen/Qwen3.5-27B` at revision
`fc05daec18b0a78c049392ed2e771dde82bdf654` and converts its Megatron checkpoint.
The HF model is stored in `$MODEL_DIR/Qwen3.5-27B`; the converted model is
`$MODEL_DIR/Qwen3.5-27B_torch_dist`. Preparation requires the GPU environment
for model conversion. It does not start task containers.

| Dataset | Prepared training input | Full evaluation input |
|---|---|---|
| FrontierSmith | `$DATA_DIR/frontiersmith.train.jsonl` — 200 tasks | `$DATA_DIR/fcs.eval.jsonl` — 180 tasks |
| TMAX | `$DATA_DIR/tmax.train.jsonl` — 14,601 eligible tasks in the current preparation | `$DATA_DIR/tb21.eval.jsonl` — 89 tasks |

The Frontier judge and both problem corpora are bundled under
[`Frontier-CS/`](Frontier-CS/README.md). Data preparation uses these local files;
it does not clone additional repositories. All 200 Smith and 180 FCS problems are used.
The ten published cross-corpus aliases are retained and recorded in the manifest.
Keep `$DATA_DIR/harbor` available at the same path on every task Docker host.
See [ASYNC_DATA.md](examples/easyppo/ASYNC_DATA.md) for source versions and manifests.

## 4. Run the task environments and judges

Use one compatible existing judge service or deploy the bundled service in an
independent CPU environment. With Docker Compose:

```bash
git clone --branch miles https://github.com/dyxg/EasyPPO.git EasyPPO-judge
cd EasyPPO-judge/Frontier-CS/algorithmic
export FRONTIERCS_JUDGE_BIND_IP=0.0.0.0
export JUDGE_WORKERS=8 GJ_PARALLELISM=8
docker compose up -d --build
curl --fail --retry 10 --retry-connrefused --retry-delay 2 http://127.0.0.1:8081/problems
```

The Compose deployment requires privileged containers. For another service
orchestrator, preserve the image, mounts, and permissions defined in the Compose file.
It does not need Miles, CUDA, Ray, model weights or the training cluster's shared
filesystem. Compose mounts one merged `problems/` directory, plus local `data/`
and `submissions/` directories. The checkout can live at any path.

All FrontierSmith and Frontier-CS trials share this service. Set
`--frontier-judge-url` during preparation to a URL reachable from every task
container, using the service DNS name or reachable address. `FRONTIERCS_JUDGE_BIND_IP` controls
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
conservative initial concurrency, use `--sglang-server-concurrency 1` (eight
concurrent trials with the default sync allocation).

From the judge's `Frontier-CS/algorithmic` directory, use `docker compose logs -f`
to view logs and `docker compose down` to stop the service. Results remain in
`data/` and `submissions/`.

TMAX and TB2.1 retain each task's supplied image or Dockerfile, auxiliary
services, and `tests/test.sh`. Harbor runs those tests and reads the task reward
after the agent. Initial image builds and some verifier scripts download
packages, so image caching alone may not remove runtime network access.

## 5. Launch training

Run the selected block through the configured cluster job environment. Each block
reuses that allocation sequentially, with a separate `run-id` for every
experiment. All four launchers accept the same context-mode and path options.

**Asynchronous SC/TIS — default eight nodes / 64 GPUs, four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-fs-$MEMORY_MODE" \
    --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"

  python examples/easyppo/run_qwen3_5_27b_async_tmax.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "async-tmax-$MEMORY_MODE" \
    --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
done
```

**Synchronous PPO — default four nodes / 32 GPUs, four dataset/context runs:**

```bash
for MEMORY_MODE in full_detach actor-summary; do
  python examples/easyppo/run_qwen3_5_27b_sync_frontiersmith.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "sync-fs-$MEMORY_MODE" \
    --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"

  python examples/easyppo/run_qwen3_5_27b_sync_tmax.py \
    --action train --critic-context-mode "$MEMORY_MODE" \
    --run-id "sync-tmax-$MEMORY_MODE" \
    --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
done
```

To run async on a **32-GPU allocation**, add
`--num-nodes 4 --trainer-num-nodes 1`.
For sync on a 64-GPU allocation, set `--num-nodes 8`. Sync derives both
training and rollout sizes from the full allocation; explicit partial-pool
overrides are rejected.
Use a matching Ray allocation for the chosen resource settings.

Both actor and critic use TP4 with PP=CP=1, giving DP8 on the default sync
32-GPU pool and DP4 on the default async 16-GPU training pool. The launchers
enable sequence parallelism, full-layer
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
$RUN_DIR/<run-id>/
  checkpoints/          # Native Megatron actor checkpoint
  checkpoints_critic/   # Native Megatron critic checkpoint
  trials/               # Harbor trajectories, candidates and verifier results
```

Resume with the **same launcher, dataset, context mode, run ID, and trainer
topology**:

```bash
python examples/easyppo/run_qwen3_5_27b_async_frontiersmith.py \
  --action train --critic-context-mode full_detach \
  --run-id async-fs-full_detach \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
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
