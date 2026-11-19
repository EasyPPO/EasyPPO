# Synchronous FrontierSmith PPO on Modal

Run from the repository root. Install `modal_launchers/requirements.txt` and
configure the Modal CLI (`modal setup`). Set
`WANDB_API_KEY` and optionally `WANDB_ENTITY` to enable W&B; its project is
`miles-run_qwen3_5_<model-size>_sync_frontiersmith` and its run ID is `--run-id`.

FrontierSmith/FCS use the [submit-only interaction and reward rules](../examples/easyppo/RECIPES.md#rewards-validation-and-checkpoints).
The rollout worker sends each C++ submission to an isolated CPU judge for compilation and execution.

The image starts from the immutable Miles v0.1.3 digest in
[`release-lock.json`](../release-lock.json), with Python 3.12, CUDA 13 and
PyTorch 2.14.1. Its build installs the locked SGLang 0.5.22 source and
`radix_transfer_plugins`, Megatron-LM, torch-memory-saver, and Transformers
5.19.0. A custom `--image` must provide the same compiled runtime and the
`/sgl-workspace/sglang` and `/root/Megatron-LM` source trees.

## Preview and launch

Use `--model-size 9b` for the Qwen3.5-9B recipe in [Claude.md](../Claude.md), including preparation. The default is `27b` for Qwen3.8-27B.

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

Prepare the shared pinned Qwen3.8-27B model once (CPU download, then single-node GPU conversion, even with `--nodes 2/4`):

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
180 × 3 = 540 evaluation trials initially and every ten rounds. The FCS score averages
the three episode rewards per task across all 180 tasks. Checkpoints are
saved every five rounds; two complete sets are retained. Generation limits
match `Claude.md`. Use identical resource options for both experiments.

The default `--terminal-environment shared-judge` sends requests directly to
the CPU judge. `--concurrency` limits active episodes per inference engine.
With two 8-GPU nodes, TP4 and the default of 128, Miles admits up to 512 episodes.
The judge queues submissions separately with 32 workers, 32 CPUs and 64 GiB of memory.
Tune its capacity with `--judge-workers`, `--judge-cpus` and `--judge-memory-gib`.
A judge error verdict scores that submission zero. Transport failures stop the run.
Each direct trial records its start, finish and cumulative judge wait in `timing.json`
alongside its trajectory and verifier results.

Use `--terminal-environment docker` to run each submission client and verifier
in a local Harbor container. This mode defaults to eight episodes per engine
and `--terminal-memory-mb 16384` per sandbox. The shared judge runs in a CPU VM
in both modes. Each attempt authenticates with its own judge connection token.

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

The local client rotates the cluster ten minutes before the 24-hour judge expires,
up to `--max-restarts 3`. It cancels the old cluster before replacing the judge.
Preflight errors, training failures, and cleanup failures stop the launcher.
Ctrl-C stops the current attempt. Rerun the original command to resume.

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
