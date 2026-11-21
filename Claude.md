# Qwen3.5-9B PPO critic comparison

**Current objective.** Compare the training effectiveness of `full_detach` and
`actor-summary` on Qwen3.5-9B using Terminal-Bench 2.1 (TB2.1) evaluation.
Train from the same Qwen3.5-9B base on AgentPick. Compare official TB2.1
success rates at matching policy versions and report the final checkpoint
and training budget for each run. Frontier-CS is a separate experiment with
its own initialization, training data, judge, and evaluation score.

Run the commands from the repository root on the checkout containing both
Terminus launch scripts. Set `MODEL_DIR`, `DATA_DIR`, and `RUN_DIR` to
paths accessible inside the job. Keep dataset versions, seeds, evaluation
settings, and compute allocation fixed across critic modes. Use distinct
output directories so the two runs retain independent checkpoints and logs.
Use a Miles-compatible image with the matching Megatron checkout. Install
the project and terminal extras in the training environment:

```bash
python -m pip install -e '.[agentic,terminal]'
```

## FCS

Train synchronous PPO on FrontierSmith and evaluate on Frontier-CS with the
[9B FrontierSmith launcher](examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py).

### Environment

Choose the GPU allocation to fit the available hardware and memory budget.
Set `--num-nodes` and `--num-gpus-per-node` to match the cluster allocation.
Rollout, critic training, and actor training alternate on all allocated GPUs
with offload enabled. Provision host memory for offload.

Training uses TP2 and rollout uses one GPU per engine by default. With `N`
total GPUs, this gives training DP `N / 2` and `N` rollout engines. Adjust `--tensor-model-parallel-size`
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
| `JUDGE_URL` | Judge service URL reachable from all rollout workers |

Make the code and assets available to their execution processes through the
platform's storage or distribution mechanism. Each rollout worker needs the
prepared tasks and HTTP access to the shared judge.
Set `--megatron-path` when using a custom Megatron installation path.

### Judge deployment

Use one shared judge service for all training and evaluation episodes. The bundled
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

### Data and model preparation

To prepare only the 200 training tasks and 180 evaluation tasks on CPU:

```bash
python scripts/make_miles_frontiersmith_data.py \
  --tasks-dir "$DATA_DIR/harbor" --out "$DATA_DIR" --judge-url "$JUDGE_URL"
```

Run from the repository root in an allocated GPU job environment:

```bash
python examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py \
  --action prepare \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --frontier-judge-url "$JUDGE_URL" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

This prepares the pinned `HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1` model, Harbor tasks, and JSONL datasets,
then converts the Megatron checkpoint. Reuse compatible existing artifacts:
`$MODEL_DIR/Qwen3.5-9B-FrontierCS-SFT-DSv3.1` and `$MODEL_DIR/Qwen3.5-9B-FrontierCS-SFT-DSv3.1_torch_dist`.
Checkpoint conversion requires GPUs. Regenerate task data after changing
`--frontier-judge-timeout`. Supply the shared judge URL when launching training.

### Training

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

To enable W&B, set `WANDB_API_KEY` in the job environment before running the
launcher. In the W&B project `miles-run_qwen3_5_9b_sync_frontiersmith`, plot
`rollout/episode_raw_reward` against `rollout/step` for the training reward.
Each episode contributes its best judge score divided by 100. The rollout mean
weights each episode equally across different numbers of compaction segments. When resuming a run, append
`--extra-env-vars 'WANDB_RUN_ID=<wandb-run-id> WANDB_RESUME=must'` to the training
command below. Use the ID after `/runs/` in the original W&B run URL and keep the
same W&B entity and project. The launcher uses `--run-id` as the display name;
the W&B run ID is what attaches new metrics to the existing run.

```bash
python examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py \
  --action train --frontier-judge-url "$JUDGE_URL" \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --run-id sync-fs-9b-full-detach-001 \
  --critic-context-mode full_detach \
  --num-rollout 100 \
  --save-interval 5 --max-keep 2 \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR"
```

Set `--sglang-server-concurrency` to fit the rollout workers and judge capacity.
The default is 4 per engine; a value of 1 allows one concurrent trial per engine.

Actor generation uses a 131072-token context window, up to 24576 tokens per
action (including thinking) and 8192 per summary (action thinking enabled, summary thinking disabled), with
compaction above 12000 context tokens. Length-limited turns increment `truncated_turns`
and continue; the cumulative episode generation budget is 150000 tokens.

Training runs for 100 rounds, each with 32 prompt groups × 16 episodes.
Every round performs four critic updates, including the first 30 critic-only
rounds. Rounds 31–100 also perform one actor update. Checkpoints are saved every
five rounds. With `--max-keep 2`, only the two most recent complete checkpoint
sets are kept. Older checkpoints are removed after the new actor, critic, and
rollout state have all been saved. Evaluation runs initially, every ten rounds, and at the end,
with 180 tasks × 3 trials per evaluation.

Outputs are under `$RUN_DIR/<run-id>/`: `checkpoints/`, `checkpoints_critic/`,
and `trials/`. Use a new run ID for each new experiment. See [Resume training](#resume-training)
to resume an existing run.

### Critic comparison

Our current primary focus is comparing **`full_detach` and `actor-summary`**
to measure how critic access to episode history affects synchronous PPO training.

| Critic mode | Context | Example run ID |
|---|---|---|
| `full_detach` | Full chronological episode history, with gradients detached across segment boundaries | `sync-fs-9b-full-detach-001` |
| `actor-summary` | The actor's context for each segment: task, current summary, and local conversation | `sync-fs-9b-actor-summary-001` |

Complete model/data preparation in sections 1–3 and configure the job/Ray
environment described under [Training](#training). Use the same `NUM_NODES`, `GPUS_PER_NODE`, `MODEL_DIR`,
`DATA_DIR`, and `RUN_DIR` for both commands below. Run each in its allocated job
environment with a fresh run ID.

**Full-detach critic:**

```bash
python examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py \
  --action train --frontier-judge-url "$JUDGE_URL" \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --run-id sync-fs-9b-full-detach-001 \
  --critic-context-mode full_detach \
  --num-rollout 100 \
  --num-critic-only-steps 30 \
  --critic-steps-per-rollout 4 \
  --rollout-batch-size 32 --samples-per-prompt 16 \
  --eval-interval 10 --eval-max-tasks 0 --eval-samples-per-prompt 3 \
  --save-interval 5 --max-keep 2 \
  --max-context-tokens 131072 \
  --max-new-tokens 24576 --summary-tokens 8192 --no-fcs-summary-thinking \
  --compact-threshold 12000 --max-total-tokens 150000 \
  --continue-on-length \
  --tensor-model-parallel-size 2 --rollout-gpus-per-engine 1
```

**Actor-summary critic:**

```bash
python examples/easyppo/run_qwen3_5_9b_sync_frontiersmith.py \
  --action train --frontier-judge-url "$JUDGE_URL" \
  --num-nodes "$NUM_NODES" --num-gpus-per-node "$GPUS_PER_NODE" \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --run-id sync-fs-9b-actor-summary-001 \
  --critic-context-mode actor-summary \
  --num-rollout 100 \
  --num-critic-only-steps 30 \
  --critic-steps-per-rollout 4 \
  --rollout-batch-size 32 --samples-per-prompt 16 \
  --eval-interval 10 --eval-max-tasks 0 --eval-samples-per-prompt 3 \
  --save-interval 5 --max-keep 2 \
  --max-context-tokens 131072 \
  --max-new-tokens 24576 --summary-tokens 8192 --no-fcs-summary-thinking \
  --compact-threshold 12000 --max-total-tokens 150000 \
  --continue-on-length \
  --tensor-model-parallel-size 2 --rollout-gpus-per-engine 1
```

A step here means one rollout round of 512 episodes. Each round has four critic
optimizer updates, each on 128 episodes. The first 30 rounds therefore perform
120 critic updates and zero actor updates. Rounds 31–100 also update the actor
once per round. Eval runs before training and after rounds 10, 20, …, 100, using
all 180 FCS tasks with three trials each (`--eval-max-tasks 0` selects all tasks).

Keep the initial model, 200-task training set, resource allocation, and other
hyperparameters identical. Start both runs from the same base checkpoint;
their run IDs select separate output directories. Compare training curves and
full Frontier-CS evaluation rewards at matched rounds.

### Resume training

Rerun the same training command from [Critic comparison](#critic-comparison)
with the same `--output-dir` and `--run-id`. Training automatically resumes
from the latest complete checkpoint.

### Modal launchers

See [Modal setup and resume](modal_launchers/README.md). Select two or four nodes
and either H200 or B200; each node has eight GPUs:

```bash
python -m modal_launchers.frontiersmith --model-size 9b --nodes 2 --gpu H200 --mode full_detach --run-id sync-fs-9b-full-detach-2n-h200-001
python -m modal_launchers.frontiersmith --model-size 9b --nodes 4 --gpu B200 --mode actor-summary --run-id sync-fs-9b-actor-summary-4n-b200-001
```

Both critic modes support both topologies and GPU types. Keep allocation and
GPU type identical when comparing modes. Prepare the shared model with
`--model-size 9b --action prepare` (single-node conversion). Add `--launch` after reviewing the
resource plan. CPU and host-memory settings are per node; defaults are 32 CPUs
and 1024 GiB RAM. Omitting `--nodes` retains the single-node default.

Each invocation runs one experiment and resumes its latest complete checkpoint.
Timeouts trigger up to three sequential cluster restarts. Keep the same node
count, GPU type and run ID when resuming; use a new ID to change topology.

## Terminal Bench

### Prepare the training and evaluation data

Use the Qwen3.5-9B base model and Modal's native CPU sandboxes for terminal
interaction and verification. Configure Modal credentials on rollout workers.
Prepare both datasets from the repository root:

```bash
python -m pip install 'harbor[modal]==0.24.0'
python scripts/make_miles_agentpick_data.py --tasks-dir "$DATA_DIR/harbor"
python scripts/make_miles_terminal_data.py --dataset terminal-bench-2.1 \
  --tasks-dir "$DATA_DIR/harbor" --out "$DATA_DIR"
```

AgentPick preparation selects 543 tasks from the pinned upstream
[split_assignment.tsv](https://huggingface.co/datasets/Fzz1/tb-training-mix-agentpick/blob/6e3e6c8a01d84f06b9f18f61782170f4801d5330/metadata/split_assignment.tsv)
and writes `$DATA_DIR/harbor/agentpick-csk4gby2/train.jsonl`. The official
89 evaluation tasks go into `$DATA_DIR/tb21.eval.jsonl`. The AgentPick output
directory must be new on the first preparation. Reuse it for both modes.

### Prepare the base model

Run checkpoint conversion once on an allocated GPU and reuse the result:

```bash
python -m examples.easyppo.terminus_full_detach --prepare-model-only --model-dir "$MODEL_DIR"
```

Both modes initialize from `Qwen/Qwen3.5-9B` and use the converted
`$MODEL_DIR/Qwen3.5-9B_torch_dist` checkpoint.

### Launch the single-GPU training checks

```bash
python -m examples.easyppo.terminus_full_detach \
  --tasks-dir "$DATA_DIR/harbor" --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR/full_detach"

python -m examples.easyppo.terminus_actor_summary \
  --tasks-dir "$DATA_DIR/harbor" --model-dir "$MODEL_DIR" --output-dir "$RUN_DIR/actor-summary"
```

Each command collects four prompt groups with two responses per group on the
AgentPick configuration-drift ETL task. Terminus-2 drives the terminal using
XML and tmux. EasyPPO triggers an actor-generated summary at 12,000 context
tokens. The single-call context limit is 63,488, the action generation limit
is 16,384, and the episode budget is 120,000 generated tokens including
summaries. PPO uses the sampled behavior-policy log probabilities. On B200
the scripts use SGLang’s Triton dense attention backend. The pinned FA4
kernel rejects Qwen3.5-9B’s head dimension of 256 during paged decode. On H100/H200 pass
`--gpu-arch hopper` for FA3. SGLang’s Triton GDN backend uses its FLA
chunk and fused recurrent kernels. Megatron uses FlashAttention for dense
attention and FLA for GDN in both actor and critic. Inspect the resolved
server arguments and GDN dispatcher in each run log.

Each batch takes four independent critic optimizer steps and one actor step.
The single-card scripts set `--ppo-head-chunk-size 8192`. The actor projects
up to 8,192 response positions at a time and recomputes the output projection
for each chunk during backward. The training pack limit is set separately.
Both modes use critic LR `5e-6`, value clip `0.2`, and value loss coefficient
`0.5`. The smoke uses zero critic-only warmup steps. A successful check
requires a verified rollout and completed optimizer updates.

### Reference scale for the comparison

| Setting | csk4gby2 reference | Single-GPU check |
|---|---:|---:|
| Prompt groups × responses per step | 32 × 12 | 4 × 2 |
| Controller training steps | 150 | 1 |
| Actor learning rate / weight decay | 3e-6 / 0 | 3e-6 / 0 |
| Actor warmup steps | 1 | 0 |
| Evaluation corpus | 89 TB2.1 tasks | Configuration-drift ETL fixture |
| Checkpoint interval | 5 steps | Final step |

### Compare the TB2.1 evaluation

For AgentPick training and TB2.1 evaluation, pass these additional arguments to either script:

```bash
--prompt-data "$DATA_DIR/harbor/agentpick-csk4gby2/train.jsonl" \
--eval-data "$DATA_DIR/tb21.eval.jsonl" --num-rollout 150
```

Terminus-2 uses Harbor’s XML prompt. A capped action is salvaged when it contains a complete XML response; otherwise Harbor requests a shorter response. Context exhaustion and the episode token budget end the attempt. Unsubmitted tasks receive zero reward. Their sampled action tokens remain eligible for PPO actor loss. At each comparison checkpoint, evaluate all 89 tasks with the same
harness and Modal environment for both modes. Follow the reference sampling
protocol of five trials per task, temperature 0.7 and top-p 0.95. Average
binary verifier rewards within each task, then over tasks. Record the policy
version, number of graded trials, timeouts and infrastructure failures with
the score. The [reference recipe](https://github.com/yichuan-w/torchtitan/tree/60facb4549bd9a1c73ca8492c4715c0e2edca59b/torchtitan/experiments/rl/experiment%20record/csk4gby2)
runs evaluation every 20 policy steps.
