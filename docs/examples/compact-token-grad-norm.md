# Compact-token gradient experiment

Compare the parameter-gradient norm of the first 64 actor tokens generated after
compaction with random 64-token windows from the same segment. The experiment
samples **100 segments** and runs across **eight GPUs**, with fixed actor and
critic checkpoints.

## Inputs

Run from the repository root using the pinned environment. Qwen3.5 GPU runs require
the repository's CUDA/FLA dependencies, including `transformers==5.12.1`.

| Input | Required content |
| --- | --- |
| Actor | Local Hugging Face checkpoint directory |
| Critic base | Compatible HF base model; defaults to the actor directory |
| Critic | Export containing `critic.safetensors`, `value_head.pt`, and `meta.json` |
| Episodes | Original collected JSONL, such as `validation/length_included.jsonl` |
| Output | A fresh directory |

Each episode must contain `episode_id`, `problem_id`, `split`, `reward`, and ordered
`segments`. Each segment contains `input_ids`, `response_start`, and a binary
`loss_mask` aligned to its response suffix. Preserve the complete episode history.
Use naturally completed trajectories with token-level data; aggregated validation
values and Miles replay files cannot replace the original episode JSONL.

If needed, export a native Megatron critic:

```bash
python scripts/miles/critic_to_hf.py \
  --input-dir /path/to/checkpoints_critic/iter_0000050 \
  --base-hf-dir /path/to/base-hf \
  --out /path/to/critic-hf
```

## Run on eight GPUs

Replace the paths below with the checkpoint, dataset, and output locations:

```bash
python -m agentic_critic_pipeline.compact_grad_parallel \
  --actor /path/to/actor-hf \
  --base-model /path/to/base-hf \
  --critic /path/to/critic-hf \
  --episodes /path/to/validation/length_included.jsonl \
  --output /path/to/compact-grad-100-summary \
  --critic-mode actor_summary \
  --device cuda --workers 8 --gpus 0,1,2,3,4,5,6,7 \
  --segments 100 --min-segment-tokens 1024 \
  --window-size 64 --controls 8 --seed 1234 \
  --advantage-normalization cohort \
  --dtype bfloat16 --qwen35-fla \
  --head-chunk-size 128 --max-tokens 120000 --bootstrap 2000
```

GPU indices refer to the devices visible through `CUDA_VISIBLE_DEVICES`. Each
worker loads a complete model on one GPU. The runner distributes critic inference
first, merges the value caches, and then distributes actor gradient measurements.
Longer contexts are assigned first to balance work. Critic and actor phases run
sequentially; the two models need not occupy GPU memory together.

For a full-history critic comparison, use its checkpoint, set
`--critic-mode full_history`, and choose a new output directory. Keep the actor,
source JSONL, seed, sample size, length threshold, and window settings identical.
Both runs will use the same selected segments and random control windows.

## Sampling

The runner selects exactly 100 distinct segments uniformly without replacement
from those satisfying both conditions:

- `segment_index > 0`: generation resumes after compaction.
- At least 1,024 generated actor tokens, counted using `loss_mask`.

Selection happens before model inference. Multiple selected segments may belong
to the same episode. The run fails if fewer than 100 segments qualify.

For each selected segment, the target is its first 64 generated actor tokens.
Eight control windows are sampled from the remaining generated-token positions,
with distinct starting positions. Controls never overlap the target but may
overlap one another. Windows are contiguous in actor-token order; intervening tool
tokens remain in the context but do not count toward the window size.

Complete episodes containing selected segments are retained for critic inference.
Only the selected segments receive gradient measurements. Prompt tokens, tool
outputs, and copied summaries remain in the model context throughout.

Inputs exceeding `--max-tokens` cause an error without truncation. For the
full-history critic, this limit applies to the concatenation of all segment
inputs in an episode.

## Loss and gradients

The critic predicts `V(s_t)` immediately before target token `t`. With terminal
return `R`, the raw advantage is `A_t = R - V(s_t)`. By default, advantages are
whitened once across all generated tokens in the complete selected episodes,
using sample variance and epsilon 1e-8. Every worker uses the same merged
statistics. Set `--advantage-normalization none` to disable whitening.

For a segment containing `N` generated actor tokens:

```text
loss_t = -stop_gradient(A_t) * log pi_theta(token_t | context_t)
g_all  = gradient_theta(sum_t loss_t / N)
g_W    = gradient_theta(sum_{t in W} loss_t / N)
g_rest = g_all - g_W
ratio  = norm(g_target) / mean_k(norm(g_control_k))
```

Gradients cover all actor parameters, including embeddings, decoder layers, and
the LM head. Shared parameters count once. `g_all` covers the current segment;
all subsets use the same denominator `N`.

Direction metrics compare each window with its own remainder, `g_rest`, excluding
the window's self-contribution. Removing a window removes its losses while
preserving its input tokens and their influence on other predictions.

This policy-gradient surrogate matches the clipped PPO policy gradient at
importance ratio 1. It excludes KL, entropy, off-policy correction, gradient
clipping, and optimizer preconditioning. Use the rollout-generating actor for an
on-policy interpretation. Terminal `R - V` assumes `gamma=lambda=1` and naturally
completed trajectories. Cohort whitening matches training-batch whitening only
when the input cohort is exactly that batch.

## Results

| File | Content |
| --- | --- |
| `selection.json` | Selected segments, eligibility count, seed, threshold, and worker assignments |
| `cohort.jsonl` | Complete episodes containing the selected segments |
| `cache/` | Merged critic values and shared advantage statistics |
| `boundaries.jsonl` | Per-segment measurements and exact window positions |
| `summary.json` | Aggregate norm ratio and confidence interval |
| `manifest.json` | Completion marker, configuration, checkpoint identities, and data hashes |
| `cache_*.log`, `measure_*.log` | Worker logs |

A complete run has `manifest.json` and reports `measured_boundaries = 100` and
`skipped_boundaries = 0` in `summary.json`. Worker failures stop the run; inspect
the corresponding log. Restart with a fresh output directory.

The primary per-segment result is `ratio_to_control_mean`: the target gradient
norm divided by the mean random-window gradient norm. A value of 2 means that the
target window has twice the average control norm for that segment.

`summary.json` reports `problem_balanced_geometric_ratio`: average log-ratios
within each problem, give problems equal weight, then exponentiate. The 95%
confidence interval uses 2,000 bootstrap samples of problems, preserving
within-problem dependence. Zero or undefined ratios are counted and excluded from
the log-ratio aggregate; a single-problem sample has no confidence interval.

Each `target` and `controls` entry contains:

| Field | Definition |
| --- | --- |
| `norm` | `norm(g_W)` |
| `window_mean_norm` | `norm(g_W) * N / window_size` |
| `rest_norm` | `norm(g_rest)` |
| `norm_over_rest` | `norm(g_W) / norm(g_rest)` |
| `projection_on_rest` | `dot(g_W, g_rest) / norm(g_rest)^2` |
| `cosine_with_rest` | `dot(g_W, g_rest) / (norm(g_W) * norm(g_rest))` |
| `norm_over_full` | `norm(g_W) / norm(g_all)` |

Positive `cosine_with_rest` indicates alignment with the other tokens; negative
values indicate conflict. `projection_on_rest` is a signed coefficient, not an
additive percentage. Zero denominators produce `null`.

Records also include advantage RMS, elapsed time, and peak allocated GPU memory.
Report these alongside model identities, precision, normalization, the primary
ratio, and its confidence interval. Differences in token content, position, and
advantage can affect the results, so the ratio alone does not establish a causal
effect of compaction.

## Resource requirements

Each segment uses one shared backbone forward and `K+2` backward passes for `K`
controls. With eight controls, the experiment performs 10 backward passes per
segment. A window's loss still backpropagates through the full segment graph.

Decoder checkpointing is enabled by default. `--head-chunk-size` bounds the number
of vocabulary-logit positions computed together. The full gradient is retained
in host memory; window/remainder comparisons use bounded FP64 chunks. Gradient
precision still depends on `--dtype`.

Each GPU must fit a complete model, its current parameter gradients, activations,
and temporary buffers. A 9B bf16 model requires approximately 18 GB for weights and
18 GB for gradients before activations. Each worker also retains roughly 18 GB of
reference gradients in host memory, plus checkpoint-loading overhead. Plan host
memory and storage bandwidth for eight concurrent workers. CPU-to-GPU reference
gradient transfers contribute to runtime.
