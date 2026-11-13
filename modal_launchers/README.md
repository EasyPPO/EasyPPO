# Synchronous FrontierSmith PPO on Modal

Run from the repository root. Install `modal_launchers/requirements.txt` and
configure the Modal CLI (`modal setup`). The launcher passes the active Modal
profile to training workers so Harbor can create CPU task sandboxes. Set
`WANDB_API_KEY` and optionally `WANDB_ENTITY` to enable W&B; its project is
`miles-run_qwen3_5_27b_sync_frontiersmith` and its run ID is `--run-id`.

## Preview and launch

Every command defaults to a local plan. Add `--launch` to allocate resources.
The default allocation is **one node, 8 H200 GPUs, 32 CPUs, 1024 GiB host RAM**;
`--gpus 4` and `--gpu B200` are available. Each invocation selects one experiment.
Review the allocation before launching. Keep the client running in tmux.

Prepare the shared pinned Qwen3.5-27B model once (CPU download, then GPU conversion):

```bash
python -m modal_launchers.frontiersmith --action prepare \
  --mode full_detach --run-id sync-fs-full-detach-001
```

The two training plans corresponding to `Claude.md` are:

```bash
python -m modal_launchers.frontiersmith \
  --mode full_detach --run-id sync-fs-full-detach-001

python -m modal_launchers.frontiersmith \
  --mode actor-summary --run-id sync-fs-actor-summary-001
```

Append `--launch` to the selected command to start it. Both use 100 rounds,
30 critic-only rounds, four critic updates per round, 32 × 16 episodes, and
180 × 5 evaluation trials initially and every ten rounds. Checkpoints are
saved every five rounds; two complete sets are retained. Generation limits
match `Claude.md`. Use identical resource options for both experiments.

`--concurrency` sets SGLang/Harbor concurrency per engine (default 128).
Training starts a CPU judge with 32 CPUs and 64 GiB memory; adjust
`--judge-cpus`, `--judge-memory-gib`, and `--judge-workers` together with trial
concurrency. Task sandboxes use Harbor's Modal backend and CPU resources.
The judge endpoint uses a per-attempt connection token, passed only to verifier
execution. Prepared task files store an environment placeholder for that token.

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
The old GPU call is cancelled before the judge is replaced. Other training
errors stop the launcher. Ctrl-C stops the current attempt; rerun to resume.
The local client must stay alive for automatic restarts. Each attempt has a
24-hour platform limit and is stopped ten minutes before its judge expires.

A checkpoint becomes resumable only after actor, critic, and rollout state are
saved and committed to the Volume. The completion marker is committed before
old checkpoints are pruned. Interrupted first saves restart from the base;
later interrupted saves roll back to the last complete set. A legacy partial
checkpoint without a completion marker fails explicitly for inspection.
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
