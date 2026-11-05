# Score Centering with an independent critic

The asynchronous terminal launcher uses the existing Miles Score Centering
loss with advantages from an independent PPO critic. `gamma=lambda=1` and
terminal-only rewards give `A_t = R - V_old,t`. Actor gradients do not propagate
into the critic or the saved advantages.

## Probability and loss inputs

The sampler records the probability of each sampled token and its top-128
candidate IDs and log probabilities. The trainer computes current probabilities
for those positions. Action and summary tokens carry loss; observations and
formatting gaps are masked out. Candidate records follow the same alignment.

For one token, let `p` be the current trainer distribution and `q` the actual
sampler distribution. The default TIS weight is `min(p/q, 2)`. The weight and
centering coefficients are detached; gradients flow through the trainer log
probabilities. The existing implementation uses a top-K head and a proportional
tail approximation.

Implementation:

- [Loss integration](../../miles/backends/training_utils/loss/hub/score_centering_loss.py)
- [Estimator](../../miles/backends/training_utils/loss/hub/score_centering/estimator.py)
- [Importance weighting](../../miles/backends/training_utils/loss/hub/score_centering/importance_sampling.py)

## Launcher controls

| Option | Default | Effect |
|---|---|---|
| `--score-centering` / `--no-score-centering` | enabled | Select SC or the asynchronous clipped-PPO actor objective |
| `--score-centering-candidates` | 128 | Candidates saved per generated token |
| `--score-centering-is` | `tis` | `none`, `tis`, or `mis` importance weighting |
| `--score-centering-tis-clip` | 2 | Maximum TIS importance weight |
| `--score-centering-mis-low` / `--score-centering-mis-high` | 0.5 / 5 | MIS keeps ratios inside this inclusive interval and zeroes other weights |
| `--score-centering-chunk-size` | 128 | Trainer token chunk for selected log probabilities |
| `--score-centering-recompute-loss` | enabled | Recompute the loss to reduce saved activations |

`none` keeps SC and uses unit importance weights. Use the SC weighting controls
without the separate generic `--use-tis` option. Training sampling is temperature
1, top-p 1, top-k -1. The launcher enables the existing FP32 inference LM-head
option and `--use-rollout-logprobs`.

PPO ratio clipping does not enter the SC actor objective. The critic still uses
value clipping `0.2`, a value-loss coefficient of `0.5`, and four mini-batch
updates. Entropy and KL coefficients are zero.

## Trajectories and diagnostics

Harbor stores candidate arrays in `easyppo-topk.npz`, referenced by
`easyppo-trajectory.json`. A top-128 ID/log-prob pair uses approximately 1 KiB per
response position. At 512 episodes × 120000 generated tokens, generated
positions alone account for about 58.6 GiB; observations and template gaps add
response positions.

Monitor `sc_correction`, `sc_train_head_mass`, `sc_rollout_head_mass`,
`sc_tail_ratio`, `sc_importance_weight`, `train_rollout_kl`, task reward, critic
explained variance, and gradient norms. Validation does not collect candidates.

The [recipe guide](RECIPES.md) covers the asynchronous and synchronous entry
points, deployment, judges, and checkpoint handling.
