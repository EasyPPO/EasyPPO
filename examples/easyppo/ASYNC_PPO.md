# Asynchronous terminal actor-critic

Dataset-specific entry points:

- [`run_qwen3_5_27b_async_frontiersmith.py`](run_qwen3_5_27b_async_frontiersmith.py): FrontierSmith → Frontier-CS.
- [`run_qwen3_5_27b_async_tmax.py`](run_qwen3_5_27b_async_tmax.py): TMAX → Terminal-Bench 2.1.

Use the [README](../../README.md) for setup and launch commands, and
[RECIPES.md](RECIPES.md) for training mechanics, checkpoints, and the
synchronous PPO entry points. Both async launchers default to an eight-node,
64-GPU allocation; the sync launchers default to four nodes and 32 GPUs.
The filename selects the dataset, and all entry points accept the same
critic-memory options.

| Setting | Value |
|---|---|
| Model | Qwen3.5-27B |
| Training / evaluation | FrontierSmith / Frontier-CS, or TMAX-15K / Terminal-Bench 2.1 |
| Driver | `train_async.py`, one-batch lookahead |
| Actor objective | Miles Score Centering with TIS, weight cap 2 |
| Accepted rollout | 32 prompt groups × 16 episodes = 512 |
| Oversampling block | 35 groups × 16 episodes = 560 |
| Critic updates | Four mini-batches of 128 episodes |
| Critic memory | `full_detach` or `actor-summary` |
| Actor / critic LR | `1e-6` / `5e-6` |
| Adam / weight decay | `(0.9, 0.999)`, epsilon `1e-8`, decay `0.01` |
| Critic-only phase | 30 outer rounds |
| Training duration | 120 outer rounds |
| Actor context / action / summary | 32768 / 8192 / 2048 tokens |
| Compaction threshold | 24576 context tokens, subject to remaining-window reserve |
| Cumulative generation | 120000 tokens per episode, including summaries |
| Training episode packing limit | 262144 tokens |
| Default GPU allocation | 16 trainer + 48 rollout B200s |
| 32-GPU allocation | `--num-nodes 4 --trainer-num-nodes 1`: 8 trainer + 24 rollout |
| Validation | Full FCS 179 / TB2.1 89 tasks, five episodes per task, sampling `1/1/-1` |
| Eval / save cadence | Every 10 outer rounds and the final round; initial evaluation enabled |

The actor and critic have independent weights, optimizers, and memory. The
critic freezes pre-update values for the entire rollout, then trains four times
before those values are supplied to the actor. Inverse-STD weights use the 16
returns in each prompt group, a standard-deviation floor of `0.075`, and
normalization across the accepted rollout.

Generation for the next batch overlaps current training. Weight publication
waits for that batch and surplus terminal-trial cleanup. The default actor loss
uses the actual sampler probabilities and top-128 candidates; PPO ratio clips
apply only when selecting `--no-score-centering`.

See [Score Centering](PPO_SCORE_CENTERING.md) for the actor loss and
[data preparation](ASYNC_DATA.md) for dataset sources and judge mounts.

Validation defaults are `--eval-max-tasks 0 --eval-samples-per-prompt 5`.
Each validation point runs 895 Frontier-CS trials or 445 TB2.1 trials. Initial
validation plus rounds 10..120 gives 13 points: 11,635 or 5,785 trials per
experiment. Full-set result names are `fcs` / `tb21`.

An explicit positive `--eval-max-tasks N` selects a seed-42 fixed subset with
`fcs_quick` / `tb21_quick` names. For a smoke run, set
`--eval-max-tasks 1 --eval-samples-per-prompt 1`.
