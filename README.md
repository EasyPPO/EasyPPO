<p align="center">
  <img src="assets/easyppo.png" width="200" alt="EasyPPO logo">
</p>

<h2 align="center">EasyPPO: Stabilizing the Critic Is Key</h2>

<p align="center">
| <a href="https://easyppo.github.io/"><b>Webpage</b></a> | <a href="https://arxiv.org/pdf/2609.36802"><b>Paper</b></a> |
</p>

EasyPPO improves PPO training stability by addressing how the critic learns from
truncated rollouts and noisy returns. It retains token-level GAE, the clipped PPO
policy objective, and KL regularization.

## Method

EasyPPO combines three changes:

1. **Actor-only overlong filtering.** Keep all rollouts for advantage estimation
   and critic training; mask truncated responses only during actor updates.
2. **Noise-normalized critic regression.** Weight each prompt's critic loss
   inversely by its rollout group's return standard deviation, with a floor for
   low-variance groups.
3. **Smaller critic mini-batches.** Split each rollout batch into four critic
   mini-batches, recomputing and clipping gradients at each optimizer step.

## Datasets

### Search-R1

We train on **Natural Questions (NQ) and HotpotQA** using the
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) data and retrieval
environment. Validation uses a subset of **NQ, TriviaQA, PopQA, HotpotQA,
2WikiMultiHopQA, MuSiQue, and Bamboogle**. Follow the Search-R1 repository for
data preparation and retrieval setup.

### FrontierSmith

We use **200 synthetic open-ended coding problems** from
[FrontierSmith](https://github.com/FrontierCS/FrontierSmith) for training and
**172 [Frontier-CS](https://github.com/FrontierCS/Frontier-CS) algorithmic problems**
for validation. Generated C++ solutions
are evaluated by the problem-specific checkers.

The 200 FrontierSmith problems (`frontiersmith_0` through `frontiersmith_199`)
and `data/frontiercs/train_synthetic_200.parquet` are pinned to
[FrontierSmith commit 166c8be](https://github.com/FrontierCS/FrontierSmith/commit/166c8be14a62b013ef78e6868e4355421c8c6456).
Each Smith problem contains its final statement, checker, judge configuration,
and test inputs/answers. Generation traces, draft statements, and candidate
solutions are not included.
The bundled `Frontier-CS/algorithmic/` judge and 172 Frontier-CS validation
problems (`frontiercs_val_*` IDs) are pinned to
[FrontierReward commit 0b379d7](https://github.com/dyxg/FrontierReward/commit/0b379d769aec019f591d8e163a10be04489e6507).
The prepared Parquet files include the prompts and judge IDs.

For judge setup, local training, W&B, and evaluation commands, see
[Local Frontier training](#local-frontier-training).

### DAPO-Math-17k / AIME24

We train on [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
with **17,917 problems** and validate on the **30 AIME 2024 problems** from
[AIME24](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024).
Repeated copies in the upstream files are removed; AIME24 is used only for
validation. The prepared training and validation files are included in
[`data/aime/`](data/aime/) and used by the commands below.

## Run local

If you already have a training environment, run directly on a single node with
**8 GPUs**. Use Python 3.12 with PyTorch, vLLM, FlashAttention, and the dependencies
in [`docker/requirements-easyppo.txt`](docker/requirements-easyppo.txt).

```bash
git clone https://github.com/EasyPPO/EasyPPO.git
cd EasyPPO
uv pip install --no-deps -e .
bash scripts/run_aime_ours.sh
```

The script loads [`configs/easyppo_aime.yaml`](configs/easyppo_aime.yaml), checks
the bundled data, and starts training with `Qwen/Qwen3.5-9B-Base`. To use local
model weights and choose an output directory:

```bash
export EASYPPO_OUTPUT_DIR=/path/to/outputs
bash scripts/run_aime_ours.sh \
  actor_rollout_ref.model.path=/path/to/Qwen3.5-9B-Base
```

Hydra overrides can be appended to the command, for example
`trainer.total_training_steps=150`. Checkpoints, rollouts, and validation results
are saved under `$EASYPPO_OUTPUT_DIR/aime-ours/` (default: `outputs/aime-ours/`).

## Local Frontier training

This workflow trains on **FrontierSmith** and evaluates on the **172 held-out
Frontier-CS problems**. The judge runs as a separate CPU service; training uses
one Linux node with **8 GPUs**. The default initialization is
[`HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1`](https://huggingface.co/HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1).
Run all commands from the repository root.

### Checkout and training environment

Until this branch is merged, use the checkout containing the Frontier files:

```bash
git clone --branch FCS https://github.com/EasyPPO/EasyPPO.git EasyPPO-frontier
cd EasyPPO-frontier
python3 -m pip install --no-deps -e .
```

Use an activated Python 3.12 GPU environment matching
[`docker/requirements-easyppo.txt`](docker/requirements-easyppo.txt), with
FlashAttention built for its PyTorch/CUDA version. Editable installation above
installs this checkout, not its dependencies. The [Docker instructions](#docker)
provide a prebuilt B200 environment; the Frontier container command is below.
The GRPO recipes use 8192 prompt tokens, 32768 response tokens, 32 training
responses per prompt, and 5 validation responses per prompt.

The data is already committed; no download or regeneration is needed:

| Purpose | File | Rows |
| --- | --- | ---: |
| Default Smith training | `data/frontiercs/train_synthetic_200.parquet` | 200 |
| Held-out FCS validation | `data/frontiercs/frontiercs_val.parquet` | 172 |

### Start and check the judge

Docker must be running on the judge host. Build and start the bundled service:

```bash
docker build -t frontiercs-judge Frontier-CS/algorithmic
JUDGE_WORKERS=8 GJ_PARALLELISM=8 bash Frontier-CS/algorithmic/run_judge.sh
export FRONTIERCS_JUDGE_URL=http://127.0.0.1:8082
curl --fail --retry 10 --retry-connrefused --retry-delay 2 "$FRONTIERCS_JUDGE_URL/health"
```

The health response should be `{"ok":true}`. The launcher uses a privileged
container named `Competitive-Programming` and replaces an existing container
with that name. Problem files, judge state, and submissions are mounted from
`Frontier-CS/algorithmic/`.

Check the complete submission/checker path before training. This valid vertex
cover includes every vertex and should receive a positive partial score:

```bash
python3 - <<'PY'
import os
from verl.utils.reward_score.frontiercs import compute_score

code = '#include <iostream>\nint main(){int n,m;std::cin>>n>>m;for(int i=0;i<n;i++)std::cout<<1<<"\\n";}'
result = compute_score(
    "frontiercs", code, "frontiercs_val_182",
    judge_url=os.environ["FRONTIERCS_JUDGE_URL"],
)
print(result)
assert result["score"] > 0, "Check the judge URL, problem files, and container logs"
PY
```

For a remote judge, set `FRONTIERCS_JUDGE_URL` to its reachable host/IP and port.
Check `/health` from every reward-worker host. In a container, `127.0.0.1`
refers to that container unless it uses host networking.

```bash
docker logs --tail 100 Competitive-Programming
# Stop the judge after the training/evaluation job finishes:
# docker stop Competitive-Programming
```

The adapter extracts C++, submits to `/submit`, and polls `/result/<sid>`.
`score` is normalized to 0–1; `acc`, `raw_score`, and `judge_raw_score` retain
the 0–100 scale. Submission failures and a 300-second polling timeout return
zero reward, so a successful health check alone does not replace the smoke test.

### W&B logging

The YAML files default to console-only logging. To upload training metrics,
authenticate and set the environment **before launching training/local Ray**:

```bash
wandb login
export WANDB_ENTITY=your-user-or-team
export WANDB_MODE=online
export WANDB_DIR="$PWD/outputs/wandb"
mkdir -p "$WANDB_DIR"
```

For unattended jobs, provide `WANDB_API_KEY` through the job's environment instead
of interactive login. Select the backend with `'trainer.logger=[console,wandb]'`.
The W&B project and run name come from `trainer.project_name` and
`trainer.experiment_name`; setting only `WANDB_PROJECT` does not override them.
For an existing remote Ray cluster, its worker environment must also have the
W&B credentials/settings. See [W&B settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py)
for the supported environment settings.

For local offline logs, use `WANDB_MODE=offline` with the W&B backend enabled;
for console-only runs, use `'trainer.logger=[console]'`. The repository's training
Dockerfile defaults `WANDB_MODE=disabled`, so explicitly override it for W&B runs.

### Choose a configuration and train

| Config name | Algorithm | Response overlong filtering |
| --- | --- | --- |
| `grpo_frontiersmith` | GRPO | Enabled |
| `dr_grpo_frontiersmith` | Dr. GRPO | Enabled |
| `grpo_frontiersmith_no_overlong_filter` | GRPO | Disabled |
| `dr_grpo_frontiersmith_no_overlong_filter` | Dr. GRPO | Disabled |

All four configs filter overlong prompts. All responses retain their original
judge rewards and participate in GRPO/Dr. GRPO advantage estimation and group
reward statistics. With response filtering enabled, overlong responses are
masked only during actor updates. With filtering disabled, they also participate
in actor updates. Response filtering applies only to training.

For a local GRPO run with W&B:

```bash
export MODEL_PATH=HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1
export TRAIN_DATA="$PWD/data/frontiercs/train_synthetic_200.parquet"
export VAL_DATA="$PWD/data/frontiercs/frontiercs_val.parquet"
export OUTPUT_DIR="$PWD/outputs/frontier-grpo-run1"
export FRONTIERCS_JUDGE_URL=http://127.0.0.1:8082

python3 -m verl.trainer.main_ppo \
  --config-path "$PWD/configs" --config-name grpo_frontiersmith \
  reward.custom_reward_function.reward_kwargs.judge_url="$FRONTIERCS_JUDGE_URL" \
  'trainer.logger=[console,wandb]' \
  trainer.project_name=EasyPPO-Frontier \
  trainer.experiment_name=frontier-grpo-run1
```

Choose any config name in the table for the other variants. Set `MODEL_PATH` to
a local model directory if desired. `TRAIN_DATA` defaults to the released
200-problem Smith training Parquet.
The explicit `judge_url` override sends the address to reward workers; the YAML's
literal default remains `http://127.0.0.1:8082`, independent of the shell variable.
For a configuration preview without starting training, append `--cfg job --resolve`.

Checkpoints, rollouts, and validation results go to `$OUTPUT_DIR/checkpoints`,
`$OUTPUT_DIR/rollouts`, and `$OUTPUT_DIR/validation`. The default training length
is 240 steps; validation runs before training and every 5 steps, and checkpoints
are saved every 5 steps, keeping the latest two. `resume_mode=auto` resumes from checkpoints in that
output directory; use a new directory for a new experiment.

The PPO recipe is also available:

```bash
export EASYPPO_OUTPUT_DIR="$PWD/outputs/frontier-ppo-run1"
bash scripts/run_frontiersmith.sh \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  'trainer.logger=[console,wandb]' \
  trainer.project_name=EasyPPO-Frontier \
  trainer.experiment_name=frontier-ppo-run1
```

This launcher reads `FRONTIERCS_JUDGE_URL` on the launch host and passes it
explicitly. It uses a critic, a 12000-token response limit, and the 200/172 splits;
its outputs go under `$EASYPPO_OUTPUT_DIR/frontiersmith/`. It reads `MODEL_PATH`
with the same SFT default as the GRPO configs. Use CLI overrides for its data
paths; `TRAIN_DATA` and `VAL_DATA` are defaults for the GRPO YAMLs, not inputs
read by the PPO shell launcher.

### Evaluation only

With the judge still running, evaluate model weights without training:

```bash
python3 -m verl.trainer.main_ppo \
  --config-path "$PWD/configs" --config-name grpo_frontiersmith \
  trainer.val_only=true trainer.resume_mode=disable \
  actor_rollout_ref.model.path=/path/to/huggingface-model \
  'data.val_files=[data/frontiercs/frontiercs_val.parquet]' \
  reward.custom_reward_function.reward_kwargs.judge_url="$FRONTIERCS_JUDGE_URL" \
  'trainer.logger=[console,wandb]' \
  trainer.project_name=EasyPPO-Frontier \
  trainer.experiment_name=frontier-eval
```

The model path must contain Hugging Face-compatible model weights and tokenizer
files, not an unmerged distributed checkpoint. To evaluate Smith, change
`data.val_files` to `[data/frontiercs/train_synthetic_200.parquet]`.

### Run the training process in Docker

On a Linux host with 8 B200 GPUs, start the judge on the host as above, then use
host networking for the training container to reach `127.0.0.1:8082`:

```bash
docker run --rm -it --gpus all --network host --shm-size=16g \
  --mount "type=bind,source=$PWD,target=/workspace/EasyPPO" \
  -w /workspace/EasyPPO -e WANDB_MODE=online \
  --entrypoint bash ghcr.io/easyppo/easyppo-b200:20261110
```

Inside the container, run `wandb login`, export the W&B/model/data/output variables,
and run the training command above. Host shell exports are not automatically
passed to Docker. If using local model files outside the checkout, mount their
directory into the container as well. Set paths to their container locations;
outputs under `/workspace/EasyPPO/outputs` persist in the host checkout.

## Docker

We also provide a prebuilt B200 training environment:
`ghcr.io/easyppo/easyppo-b200:20261110`.

This image includes `flash-linear-attention==0.6.0`, `fla-core==0.6.0`, and
`flash-attn==2.8.3`. The recipes in `configs/` select `flash_attention_2`.
The FrontierSmith recipes enable actor parameter and optimizer offload.

On a host with **8 B200 GPUs**, Docker, and NVIDIA Container Toolkit, run from the
repository root:

```bash
docker run --rm --gpus all --shm-size=16g \
  --mount "type=bind,source=$PWD,target=/workspace/EasyPPO" \
  ghcr.io/easyppo/easyppo-b200:20261110 \
  bash scripts/run_aime_ours.sh
```

This runs the same training script using the current checkout. Outputs are saved
to `outputs/aime-ours/` in your local repository.

To build the environment locally:

```bash
docker build -f docker/Dockerfile.b200 -t ghcr.io/easyppo/easyppo-b200:20261110 .
```

To check FA2 and FLA forward/backward kernels and a small randomly initialized
Qwen3.5 model on one B200, without downloading model weights:

```bash
docker run --rm --gpus all \
  --mount "type=bind,source=$PWD,target=/workspace/EasyPPO" \
  ghcr.io/easyppo/easyppo-b200:20261110 python3 scripts/check_b200_gpu.py
```

The `Publish B200 image` GitHub Actions workflow builds and checks the environment
before publishing a `b200-YYYYMMDD` Git tag as `easyppo-b200:YYYYMMDD`.

## FCS GRPO on Modal

[`modal_launchers/fcs_grpo.py`](modal_launchers/fcs_grpo.py) runs FrontierSmith
training and Frontier-CS validation with a GPU cluster and a separate CPU judge.
The default plan is **two nodes of 8 H200 GPUs**, with 32 CPUs and 256 GiB RAM per
node, a **32-CPU / 64-GiB judge VM** with 32 workers, and a **1-CPU / 2-GiB
coordinator**. The cluster uses
[Modal RDMA networking](https://modal.com/docs/guide/multi-node-clusters); the judge
uses the [VM runtime](https://modal.com/docs/guide/sandboxes) for Linux namespaces
and cgroups, and an authenticated HTTPS endpoint reachable by every reward worker.

Install the client from the repository root and configure your Modal account:

```bash
uv venv --python 3.12 .venv-modal
uv pip install --python .venv-modal/bin/python -r modal_launchers/requirements.txt
.venv-modal/bin/modal token new
```

The `easyppo` Volume already contains the FCS SFT model. The launcher defaults to
this cached snapshot, without downloading it again:

```text
/mnt/easyppo/cache/huggingface/hub/models--HuanzhiMao--Qwen3.5-9B-FrontierCS-SFT-DSv3.1/snapshots/016efe4f9b8d1f472b412a26bd43e9b3f690a223
```

Both `::train` and `::resume` accept `--model-path` to use another complete model
on the Volume. The coordinator checks config, tokenizer, and weight files before
creating the judge or invoking the GPUs. The separately stored
`EasyPPO/models/Qwen3.5-9B-Base` remains the default for the AIME launchers.

W&B is required. The run name becomes the W&B run ID, with `WANDB_RESUME=allow`;
reusing it resumes the same experiment. Console and file metrics are also enabled.

```bash
export WANDB_API_KEY='your-wandb-api-key'
export WANDB_ENTITY='your-user-or-team'         # optional
export WANDB_PROJECT='EasyPPO-FrontierSmith'    # optional; this is the default

# These defaults apply to both plan and Modal commands.
export EASYPPO_FCS_GPU=H200
export EASYPPO_FCS_NODES=2
.venv-modal/bin/python -m modal_launchers.fcs_grpo --run-name fcs-h200
```

The last command only prints the resource plan and does not allocate anything.
After approving that plan, start a detached run:

```bash
.venv-modal/bin/modal run --detach modal_launchers/fcs_grpo.py::train \
  --run-name fcs-h200
```

The remote CPU coordinator starts the judge, checks a real C++ submission through
compilation, execution, scoring, and reward normalization, then invokes the GPU
cluster. The detached attempt keeps running if the local terminal disconnects.
On completion or failure, the coordinator stops the judge; on interruption or
failure while waiting for the cluster, it also cancels the cluster call.

For automatic continuation after **Function timeouts only**, keep this command
alive in `tmux`:

```bash
.venv-modal/bin/modal run modal_launchers/fcs_grpo.py::resume \
  --run-name fcs-h200 --max-timeout-restarts 10
```

Each continuation uses the same model, recipe, output directory, and W&B run ID,
creates a fresh judge and cluster, and resumes the latest checkpoint. Training,
judge preflight, and other non-timeout errors propagate immediately. The restart
limit counts additional attempts after the first. Each Function and judge Sandbox
has a 24-hour timeout; the coordinator's limit also includes setup and GPU queue
waiting. `::train` runs one attempt; `::resume` supplies the continuation loop.

The default recipe is `grpo_frontiersmith`: standard GRPO with overlong-response
filtering enabled, 240 steps, FA2 attention, and actor parameter/optimizer offload
enabled. Both entrypoints accept `--recipe` to select any of the four Frontier
recipes. To change resources, set
`EASYPPO_FCS_GPU` (`H200`, `B200`, or `H100`) and `EASYPPO_FCS_NODES` (1, 2, 4, 8,
16, or 32) before both preview and launch. Judge resources can be changed through
`EASYPPO_FCS_JUDGE_CPUS`, `EASYPPO_FCS_JUDGE_MEMORY_GIB`, and
`EASYPPO_FCS_JUDGE_WORKERS`.

The launcher uploads this checkout, both dataset Parquets, and the bundled judge
problems. Image, existing Volume, cache, and output paths come from
[`modal_launchers/config.yaml`](modal_launchers/config.yaml). With the defaults,
checkpoints, rollouts, validation results, and timestamped metrics are stored under
`easyppo:outputs/<run-name>/`. Checkpoints are saved **every 5 steps, keeping at
most 2 actor checkpoints**. Use a new name when changing the recipe, model path,
or cluster size.

To check an already running judge locally on CPU, without creating Modal resources:

```bash
.venv-modal/bin/python -m modal_launchers.fcs_grpo \
  --check-judge http://127.0.0.1:8082
```

For an authenticated Modal judge endpoint, set `FRONTIERCS_JUDGE_TOKEN` before
using `--check-judge`. Local CPU tests cover configuration, authentication,
launch ordering, timeout continuation, and checkpoint coordination. The image
contains H200 `sm_90` kernels, but this launcher has not yet been validated by
running a GPU cluster.

## AIME B200 training on Modal

Launch on **8 B200 GPUs** or **32 B200 GPUs** using the public
`ghcr.io/easyppo/easyppo-b200:20261110` image. No registry credentials are needed.
Launchers, configuration, and client dependencies are in [`modal_launchers/`](modal_launchers/).

From the repository root, install the Modal client and sign in:

```bash
uv venv --python 3.12 .venv-modal
uv pip install --python .venv-modal/bin/python -r modal_launchers/requirements.txt
.venv-modal/bin/modal token new
```

Before launching:

- Create a Modal Volume named `easyppo` and upload the model weights, tokenizer,
  and `config.json` to `EasyPPO/models/Qwen3.5-9B-Base` inside it.
- If your Volume or model location differs, update `volume_name` and `model_dir`
  in [`modal_launchers/config.yaml`](modal_launchers/config.yaml).
- Set your W&B credentials. The 32-GPU option also requires Modal multi-node
  B200 access and quota for 32 GPUs.

```bash
export WANDB_API_KEY='your-wandb-api-key'
export WANDB_ENTITY='your-user-or-team'
export WANDB_PROJECT='EasyPPO'
```

**8 GPUs — one node:**

```bash
.venv-modal/bin/modal run --detach modal_launchers/b200.py::train \
  --run-name aime-b200-8gpu
```

**32 GPUs — four nodes with 8 GPUs each:**

```bash
.venv-modal/bin/modal run --detach modal_launchers/b200_32gpu.py::train \
  --run-name aime-b200-32gpu
```

Use a new run name for each experiment. With the default configuration, results
are saved in the `easyppo` Volume under `outputs/<run-name>/`. The 8-GPU launcher
runs 300 steps; the 32-GPU launcher runs 150 steps.

## How to cite this paper

```bibtex
@misc{zhou2026easyppo,
  title  = {EasyPPO: Stabilizing the Critic Is Key},
  author = {Xuanyi Zhou and Qiuyang Mang and Huanzhi Mao and
            Dacheng Li and Wenhao Chai and Mayank Mishra and Yichuan Wang and
            Karthik Narasimhan and Alvin Cheung and Joseph E. Gonzalez},
  year   = {2026},
  eprint = {2609.36802},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2609.36802}
}
```
