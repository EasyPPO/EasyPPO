# Actor gradients across trajectory compactions

Compare one frozen actor using **two independently pretrained critics**: the
existing actor-summary critic on actor-summary inputs and the existing
full-history critic on full-history inputs. Reuse the completed experiments'
checkpoints and frozen validation trajectories. This experiment performs no
training, collection, optimizer update, or re-tokenization.

Production execution is pending. Local checks use tiny random models and
synthetic trajectories; they do not measure the pretrained production critics.

## Deployment inputs

Use the repository's pinned runtime, including `transformers==5.12.1`, PyTorch,
Matplotlib, and the CUDA/FLA dependencies for Qwen3.5. Each critic export must
contain `critic.safetensors`, `value_head.pt`, and `meta.json`. If conversion is
needed, use `scripts/agentic_critic/critic_to_hf.py` on the existing native
checkpoint. Select the same completed training batch for both independently
trained critics and retain their original training provenance. Do not swap
inputs on a single critic and describe that as two pretrained critics.

The actor must be the original rollout-generating local HF checkpoint. The
critic base must match the base used to train/export both critics. Supported
actor loss decomposition covers Llama and Qwen3.5 text models.

The episode input is the original frozen validation JSONL with exact
`input_ids`, `response_start`, `loss_mask`, reward, and **every segment**, including
segment zero. Aggregated value dumps and replay files cannot replace it. Prompt,
tool feedback, and copied summaries remain in the context but have masked
losses. Generated summary tokens retain their original loss masks. Naturally
completed trajectories and finite rewards are required; incomplete, duplicate,
misaligned, or overlength input fails without truncation.

Fill these paths on the machine holding the completed experiment artifacts:

```bash
export ACTOR=""
export CRITIC_BASE=""
export ACTOR_SUMMARY_CRITIC=""
export FULL_HISTORY_CRITIC=""
export VALIDATION_JSONL=""
export TRAJECTORY_GRAD_OUTPUT=""

: "${ACTOR:?Fill ACTOR}"
: "${CRITIC_BASE:?Fill CRITIC_BASE}"
: "${ACTOR_SUMMARY_CRITIC:?Fill ACTOR_SUMMARY_CRITIC}"
: "${FULL_HISTORY_CRITIC:?Fill FULL_HISTORY_CRITIC}"
: "${VALIDATION_JSONL:?Fill VALIDATION_JSONL}"
: "${TRAJECTORY_GRAD_OUTPUT:?Fill TRAJECTORY_GRAD_OUTPUT}"
python -m agentic_critic_pipeline.trajectory_grad_comparison run \
  --actor "$ACTOR" --base-model "$CRITIC_BASE" \
  --actor-summary-critic "$ACTOR_SUMMARY_CRITIC" \
  --full-history-critic "$FULL_HISTORY_CRITIC" \
  --episodes "$VALIDATION_JSONL" --output "$TRAJECTORY_GRAD_OUTPUT" \
  --workers 8 --device cuda --gpus 0,1,2,3,4,5,6,7 \
  --dtype bfloat16 --qwen35-fla --max-tokens 120000 \
  --head-chunk-size 128 --advantage-normalization none \
  --seed 1234 --bootstrap 2000
```

By default, measure **all supplied trajectories and all their segments**,
including trajectories with zero compactions. For a smaller experiment, add
`--trajectories 100` to uniformly sample exactly 100 complete trajectories without
replacement before either critic runs. Both conditions and every worker share
the saved selection. If there are too few eligible trajectories, the experiment
fails. Sampling is over trajectories, not segments or compaction-count strata.
Preserve `selection.json` and report per-count coverage.

GPU indices are logical indices within inherited `CUDA_VISIBLE_DEVICES`.
Each worker holds one complete model, with no tensor/model sharding. Whole
trajectories are assigned longest-first. For each critic, caching and actor
measurement run as separate phases; models need not coexist on a GPU. Both
conditions run sequentially. Worker failures terminate sibling workers and
prevent the final completion manifest. Use a fresh output directory for retries.

## Gradient definitions

For trajectory `e`, segment `s` contains `n_s` generated actor tokens; let
`N = sum_s n_s`. For each critic condition:

```text
A_st = R_e - V_critic(state immediately before actor token t)
L_st = -stop_gradient(A_st) * log pi_actor(token_t | original actor segment context)
g_s  = gradient_theta(sum_t L_st / N)
g_e  = sum_s g_s = gradient_theta(sum_s sum_t L_st / N)

segment_grad_norm        = norm(g_s)
trajectory_grad_norm     = norm(g_e)
cosine_with_trajectory   = dot(g_s, g_e) / (norm(g_s) * norm(g_e))
norm_ratio               = norm(g_s) / norm(g_e)
segment_mean_grad_norm   = norm(g_s) * N / n_s
segment_mean_norm_ratio  = norm_ratio * N / n_s
projection_on_trajectory = dot(g_s, g_e) / norm(g_e)^2
```

The primary segment gradient uses the **same trajectory denominator** as `g_e`.
This gives additive vector contributions and avoids comparing incompatible loss
scales. The additional token-mean norm and ratio answer the alternative question
using each segment's own `n_s` denominator. Both definitions are retained rather
than silently conflating them. The trajectory loss is a token mean across all
segments, not an unweighted mean of segment means.

The actor always reads its original segment input independently, including its
summary context. Only the critic's checkpoint and input mode change. Even for
the full-history critic, the actor never receives concatenated full history.
Gradients cover all unique actor parameters, including tied embeddings/LM head
once. There is no parameter sub-sampling or random projection.

Default normalization is **none**, preserving the raw `R - V` scale when
comparing critic conditions. Optional `--advantage-normalization cohort` whitens
advantages once per critic over all generated tokens in the entire selected
cohort, using sample variance and epsilon `1e-8`. All workers use that condition's
merged moments. These are separate per-critic statistics on the same cohort;
normalization changes the estimand and must be reported. Raw and used advantage
RMS are saved per segment in either case.

This surrogate equals the unclipped PPO policy gradient at importance ratio one.
It excludes entropy, KL, clipping, optimizer preconditioning, and off-policy
correction. `R - V` uses a terminal Monte Carlo return with `gamma=lambda=1`.
Use the rollout-generating actor for an on-policy interpretation. Frozen
validation-cohort whitening is not training-batch whitening.

The implementation makes two forward/backward passes per nonempty segment.
First it accumulates `g_e` in FP32 host memory; then it recomputes each `g_s` to
measure products with `g_e` in bounded FP64 chunks. Only one segment graph and
one full reference vector are retained. Parameter-gradient precision still
depends on `--dtype`; FP64 reductions do not make BF16 gradients exact FP64
derivatives. Evaluation mode disables dropout; no `.grad` buffers or weights
are updated. Layer checkpointing and chunked LM-head loss are enabled by default.

A 9B BF16 actor requires roughly 18 GB for weights and 18 GB for gradients
before activations and workspaces; the FP32 trajectory reference requires
roughly **36 GB of host memory per worker**. Allow additional checkpoint-loading
and transfer buffers, and multiply host requirements by the worker count.
Full-history critic inference must fit its complete concatenated context.

## Saved data and analysis

| Path | Contents |
| --- | --- |
| `selection.json`, `cohort.jsonl`, `cache_input_*.jsonl` | Frozen sample, worker assignment, original tokens/masks and source/cohort hashes |
| `<mode>/cache/values.jsonl` | Every generated token's critic prediction, grouped by original segment |
| `<mode>/cache/manifest.json` | Critic/base hashes, common advantage moments and worker provenance |
| `<mode>/trajectories.jsonl` | One trajectory norm, length/count, parameter count, projection check, timing and GPU memory per trajectory |
| `<mode>/segments.jsonl` | All segment norms, cosines, ratios, token fractions, counts and advantage RMS |
| `<mode>/cache_*.log`, `<mode>/measure_*.log` | Worker logs; original worker results/manifests also remain available |
| `gradient_report/*.csv` | Flat copies of both conditions' segment and trajectory measurements |
| `gradient_report/paired_*.jsonl` | Matched actor-summary minus full-history differences |
| `gradient_report/summary.json` | Aggregate metrics, zero/undefined counts and interpretation notes |
| `gradient_report/plot_data.json` | All plotted means, confidence intervals, valid/undefined counts and a prior-count × total-count conditional table |
| `manifest.json` | Written last on complete success; checkpoint identities, software versions, source/selection hashes, worker manifests and result checksums |

Five figures are produced as PNG (220 dpi) and PDF:

- `segments_by_prior_compactions`: norms, cosine and ratios as segments encounter
  zero, one, two, … preceding compactions, plus token-mean norms and token fractions.
- `segments_by_total_compactions`: the same metrics grouped by total trajectory
  compactions, with every original segment included.
- `trajectories_by_total_compactions`: whole-trajectory gradient norm and actor
  token count versus total compactions; each trajectory appears once.
- `paired_by_prior_compactions` and `paired_by_total_compactions`: paired critic
  differences in contribution norm, cosine and norm ratio, with a zero reference.

Each cell averages records within a trajectory, then trajectories within a
problem, then problems equally. Shading is a 95% bootstrap interval resampling
whole problems, preserving dependence among segments/trajectories in a problem.
It is absent for single-problem cells. Exact per-cell counts and all raw metrics
remain available. Empty-loss segments have zero contribution norm and undefined
cosine/token-mean norm; zero denominator metrics are `null`, never fabricated
zeros. Undefined values are excluded only from the relevant metric's aggregate.

Interpretation should consider direction and magnitude together. Negative
cosine indicates opposition to the trajectory update; cosine here includes the
segment's own contribution in the trajectory reference. A norm ratio above one
is possible when segments cancel and is not an additive percentage. Signed
projections sum to approximately one for nonzero trajectories; the stored
`projection_sum` helps check the implementation and numerical precision.

Prior-count curves include only trajectories reaching that segment, so the
cohort composition can change with count. Total-count curves compare different
trajectories and lengths. Use the saved conditional prior/total-count table,
token fractions, token-mean norms and paired critic differences when assessing
patterns; these observational curves alone do not establish a causal effect of
compaction. Independently trained critic quality and advantage scale can also
change the actor gradients. Report actual measurements before making claims.

Re-render saved measurements on CPU without loading checkpoints or inference:

```bash
python -m agentic_critic_pipeline.trajectory_grad_figures \
  --output-root "$TRAJECTORY_GRAD_OUTPUT"
```

The renderer verifies the completed measurement checksums. It writes local files
only and does not upload artifacts or require a W&B account.
