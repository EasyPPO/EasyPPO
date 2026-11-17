# Synchronous FrontierSmith PPO on Modal

Run from the repository root. Install `modal_launchers/requirements.txt` and
configure the Modal CLI (`modal setup`). Set
`WANDB_API_KEY` and optionally `WANDB_ENTITY` to enable W&B; its project is
`miles-run_qwen3_5_27b_sync_frontiersmith` and its run ID is `--run-id`.

The image starts from the immutable Miles v0.1.3 digest in
[`release-lock.json`](../release-lock.json), with Python 3.12, CUDA 13 and
PyTorch 2.14.1. Its build installs the locked SGLang 0.5.22 source and
`radix_transfer_plugins`, Megatron-LM, torch-memory-saver, and Transformers
5.19.0. A custom `--image` must provide the same compiled runtime and the
`/sgl-workspace/sglang` and `/root/Megatron-LM` source trees.

## Preview and launch

Every command defaults to a local plan. Add `--launch` to allocate resources.
Select `--nodes 2` or `--nodes 4` and `--gpu H200` or `--gpu B200`.

| Training nodes | GPUs per node | Total GPUs |
|---|---|---|
| `--nodes 1` (default) | 8 | 8 |
| `--nodes 2` | 8 | 16 |
| `--nodes 4` | 8 | 32 |

Each node requests 32 CPUs and `--memory-gib 1024` host RAM by default.
`--memory-gib` is per node. H200 is the default GPU; B200 uses the same recipe.
Multi-node allocations use RDMA and require eight GPUs per node; `--gpus 4`
is available only with `--nodes 1`. Each invocation selects one experiment.
Review the allocation before launching. Keep the client running in tmux.

Prepare the shared pinned Qwen3.5-27B model once (CPU download, then single-node GPU conversion, even with `--nodes 2/4`):

```bash
python -m modal_launchers.frontiersmith --action prepare \
  --mode full_detach --run-id sync-fs-full-detach-001
```

Two-node plans (16 GPUs):

```bash
python -m modal_launchers.frontiersmith --nodes 2 --gpu H200 \
  --mode full_detach --run-id sync-fs-full-detach-2n-h200-001
python -m modal_launchers.frontiersmith --nodes 2 --gpu H200 \
  --mode actor-summary --run-id sync-fs-actor-summary-2n-h200-001
```

Four-node plans (32 GPUs):

```bash
python -m modal_launchers.frontiersmith --nodes 4 --gpu B200 \
  --mode full_detach --run-id sync-fs-full-detach-4n-b200-001
python -m modal_launchers.frontiersmith --nodes 4 --gpu B200 \
  --mode actor-summary --run-id sync-fs-actor-summary-4n-b200-001
```

Either topology accepts either GPU type. Set `--gpu H200` or `--gpu B200`
and use a corresponding fresh run ID for a new experiment.

Append `--launch` to the selected command to start it. Both use 100 rounds,
30 critic-only rounds, four critic updates per round, 32 × 16 episodes, and
180 × 5 evaluation trials initially and every ten rounds. Checkpoints are
saved every five rounds; two complete sets are retained. Generation limits
match `Claude.md`. Use identical resource options for both experiments.

`--concurrency` sets SGLang/Harbor concurrency per engine (default 8).
Training starts a CPU judge with 32 CPUs and 64 GiB memory; adjust
`--judge-cpus`, `--judge-memory-gib`, and `--judge-workers` together with trial
concurrency. Each GPU node starts one local Docker daemon; Harbor creates
episode containers on that node using its Docker backend. These containers
share the training node's CPU and RAM. `--terminal-memory-mb 16384` sets the
per-container memory request; size node RAM and concurrency together.
The gVisor Docker daemon uses host networking and the supplied Compose overlay.
The shared judge runs in a CPU VM because go-judge fails to initialize its
execution environments under gVisor. Infrastructure and verifier failures stop
the run with an error.
The judge endpoint uses a per-attempt connection token, passed only to verifier
execution. Prepared task files store an environment placeholder for that token.

Preview one actor/critic training round with a small final evaluation:

```bash
python -m modal_launchers.frontiersmith --mode full_detach --run-id sync-fs-smoke-001 \
  --num-rollout 1 --num-critic-only-steps 0 \
  --rollout-batch-size 4 --samples-per-prompt 2 --critic-steps-per-rollout 1 \
  --skip-eval-before-train --eval-max-tasks 2 --eval-samples-per-prompt 1 \
  --concurrency 1
```

This remains a local preview until `--launch` is added. For an initial small
evaluation too, omit `--skip-eval-before-train`. The task/sample limits apply to
every evaluation point; use a fresh run ID when changing the experiment settings.

The default Miles image is pinned by digest. `--image` accepts a compatible
replacement with `/root/Megatron-LM`. The launcher installs the branch's source
and its pinned terminal dependencies into that image.

## Resume and storage

The `easyppo-miles` Volume holds `models/`, `cache/`, and `outputs/<run-id>/`.
Set `--volume` to use another Volume. Each run stores actor checkpoints,
critic checkpoints, rollout state, prepared tasks, and trial traces separately.

Rerun the original command with the same run ID and settings to resume. A
manifest rejects changes to the critic mode, topology, image, or other launch
settings. Do not run two clients with the same run ID concurrently.

Timeouts automatically restart the selected experiment, up to `--max-restarts 3`.
The old cluster call is cancelled before the judge is replaced. All nodes restart together. Other training
errors stop the launcher. Ctrl-C stops the current attempt; rerun to resume.
The local client must stay alive for automatic restarts. Each attempt has a
24-hour platform limit and is stopped ten minutes before its judge expires.

A checkpoint becomes resumable only after actor, critic, and rollout state are
saved and committed to the Volume. On multiple nodes, every node commits its
shards, all mounts reload, and only then does the head validate completeness. The completion marker is committed before
old checkpoints are pruned. Interrupted first saves restart from the base;
later interrupted saves roll back to the last complete set. A legacy partial
checkpoint without a completion marker fails explicitly for inspection.
Recovery runs on the head before workers reload their mounts and join Ray.
Optimizer, scheduler, RNG, and rollout progress follow the framework checkpoint
load path. A final checkpoint whose evaluation was interrupted reruns evaluation.
`training_complete.json` is written only after the training process exits successfully.

## CPU check

```bash
python -m modal_launchers.frontiersmith --action cpu-smoke --launch
```

This makes two sequential CPU calls, commits a small probe to the Volume,
reads it in a fresh container, and removes it. It allocates zero GPUs.
The CPU check does not validate GPU kernels, model conversion, or training.
