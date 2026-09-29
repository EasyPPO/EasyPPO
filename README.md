# EasyPPO

**EasyPPO: Stabilizing the Critic Is Key**

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

| Dataset | Source |
| --- | --- |
| Search-R1 | [GitHub](https://github.com/PeterGriffinJin/Search-R1) |
| FrontierSmith | [GitHub](https://github.com/FrontierCS/FrontierSmith) |
| DAPO-Math-17k / AIME24 | [Training data](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) · [Evaluation data](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) |

The commands below use the DAPO-Math-17k and AIME24 files included in
[`data/aime/`](data/aime/). For Search-R1 and FrontierSmith, follow the data and
environment setup in their repositories.

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

## Docker

We also provide a prebuilt B200 training environment:
`ghcr.io/easyppo/easyppo-b200:20260926`.

On a host with **8 B200 GPUs**, Docker, and NVIDIA Container Toolkit, run from the
repository root:

```bash
docker run --rm --gpus all --shm-size=16g \
  --mount "type=bind,source=$PWD,target=/workspace/EasyPPO" \
  ghcr.io/easyppo/easyppo-b200:20260926 \
  bash scripts/run_aime_ours.sh
```

This runs the same training script using the current checkout. Outputs are saved
to `outputs/aime-ours/` in your local repository.

## B200 training on Modal

Launch on **8 B200 GPUs** or **32 B200 GPUs** using the public
`ghcr.io/easyppo/easyppo-b200:20260926` image. No registry credentials are needed.
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
