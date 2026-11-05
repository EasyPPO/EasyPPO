# Qwen3.5-27B terminal recipes

The [README](../../README.md) provides environment setup, data/model downloads,
Ray startup, judge operation, and complete launch commands. This reference
specifies the current training behavior shared by those commands.

## Entry points

| Entry point | Driver | Actor objective |
|---|---|---|
| [`run_qwen3_5_27b_async_frontiersmith.py`](run_qwen3_5_27b_async_frontiersmith.py) | [`train_async.py`](../../train_async.py), one-batch prefetch | Miles Score Centering with TIS; FrontierSmith → Frontier-CS |
| [`run_qwen3_5_27b_async_tmax.py`](run_qwen3_5_27b_async_tmax.py) | [`train_async.py`](../../train_async.py), one-batch prefetch | Miles Score Centering with TIS; TMAX → TB2.1 |
| [`run_qwen3_5_27b_sync_frontiersmith.py`](run_qwen3_5_27b_sync_frontiersmith.py) | [`train.py`](../../train.py), synchronous | Clipped PPO; FrontierSmith → Frontier-CS |
| [`run_qwen3_5_27b_sync_tmax.py`](run_qwen3_5_27b_sync_tmax.py) | [`train.py`](../../train.py), synchronous | Clipped PPO; TMAX → TB2.1 |

All four launchers use an independent HF memory critic and accept
`--critic-context-mode full_detach|actor-summary`. A run trains one critic in
one selected mode. They share data preparation, rewards, batching, optimizer,
token budgets, validation, and checkpoint conventions. Async defaults to
64 GPUs (16 train + 48 rollout); sync defaults to 32 GPUs (8 train + 24 rollout).
The launch interface is the same shape, with the dataset selected by filename.

## Model and data

The model is `Qwen/Qwen3.5-27B`, pinned to HF revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`. The Megatron actor uses the converted
checkpoint. The critic starts from its text backbone and a separate scalar
value head. Its parameters, optimizer, and memory are independent of the actor.

| Dataset | Training | Full evaluation | Source version |
|---|---:|---:|---|
| FrontierSmith → Frontier-CS | 190 synthetic tasks | 179 algorithmic tasks | FrontierSmith `ac3046cce7a270e3177cbb41376f36e03de1d518`; Frontier-CS `59687e5428266a22a49cf51f981b912103bfa91d` |
| TMAX → TB2.1 | 14,601 eligible tasks in the current preparation | 89 tasks | Pinned Harbor registry references in [`make_miles_terminal_data.py`](../../scripts/make_miles_terminal_data.py) |

Frontier preparation removes ten explicitly published training aliases and
excludes problem 61 from the fixed Frontier-CS scoring set. TMAX preparation
filters exact evaluation overlap, duplicate executable fixtures, and initial
instructions that exceed the action budget. The manifests and generated
JSONL files record the resulting task sets. See [data preparation](ASYNC_DATA.md).

The terminal harness runs bash in Harbor sandboxes. Frontier tasks use their
per-trial Compose judge sidecar and native partial scores; TMAX/TB use their
supplied task verifiers. Model decoding is handled by Miles/SGLang.

## Batch and update schedule

Each round accepts **32 prompt groups × 16 independent episodes = 512 episodes**.
The first collection block launches 35 groups × 16 = 560 episodes. Additional
blocks may be needed to collect 32 complete groups; the collector cancels the
unused tail and waits for trial cleanup. Sampling is temperature=1, top-p=1,
top-k=-1. Training seed is 1234; the data-shuffle seed is 42.

The actor performs one optimizer update over the accepted round. The critic
runs one epoch over four mini-batches, each **8 groups × 16 = 128 episodes**,
with one optimizer update per mini-batch. A prompt group stays within one
critic update. Its episodes can occupy different DP shards; each episode's
segments retain matching actor/value placement.

With PP=CP=1, let `D=trainer_GPUs/actor_TP`, `P` be accepted prompt groups,
`n` episodes per group, and `K` critic steps. The batch constraints are:

```text
(P*n) % D       = 0       # Actor episodes distribute evenly across DP
P % K           = 0       # Critic updates contain complete prompt groups
(P*n/K) % D     = 0       # Each critic mini-batch distributes evenly across DP
```

`P` itself need not divide evenly across DP. Rollout GPUs must divide into
complete engine-TP groups; prompt groups, oversampling, and evaluation counts
are independent of the total GPU count and number of rollout engines.

An episode can produce multiple physical actor samples after compaction.
Those segments share one terminal reward, episode identity, and critic weight.
They do not increase the logical batch or count as independent return samples.

The default run contains 120 outer rounds. Rounds 1–30 train only the critic;
the first actor update is round 31. This gives **480 critic updates and 90 actor
updates**, including 120 critic updates during warmup. LR warmup is separately
set to zero. Accepted training volume is nominally 61,440 episodes; actual trial
starts include oversampling and any additional collection.

## Synchronous training

Each synchronous round generates the entire batch, computes critic values,
performs the four critic updates, and performs the actor update before starting
the next training rollout. The new actor is published for the next round.

The two synchronous launchers use the standard `policy_loss` with PPO bounds
`[1-0.2, 1+0.28] = [0.8, 1.28]`. The denominator is scored by the trainer before
the actor update for that round. They do not enable SC, TIS, or an old-actor
queue. The rollout's actor weights remain unchanged until collection finishes.

```text
L_actor,t = -min(r_t A_t, clip(r_t, 0.8, 1.28) A_t)
r_t       = p_current(a_t) / p_before_update(a_t)
```

PPO clipping acts through this surrogate; it is not a hard bound on the
parameter update. Entropy, reference-KL loss, reward KL, dual clipping, and
advantage whitening are disabled.

## One-step asynchronous training

The asynchronous driver starts collection of the next batch before training
the current batch. It computes the critic's pre-update values, makes four
critic updates, and then makes the actor update. Before publishing weights,
it finishes the prefetched batch and cancels/cleans up excess terminal trials.
This keeps each decoding call within one published model version.

With initial weights `theta0`, the first batch and its prefetched successor
both use `theta0`. The first actor update produces `theta1`; publication waits
for that successor to finish. Subsequent batches can therefore be generated
one publication behind the trainer. During critic-only rounds the actor
weights stay fixed.

The default async actor uses the existing Miles `score_centering` loss with
critic-derived GAE and `--use-rollout-logprobs`. It consumes the actual sampler
probabilities rather than recomputing an old actor distribution. Sampling
records top-128 candidates and their probabilities for action and summary
tokens. Trainer probability computation uses chunks of 128; loss recomputation
is enabled. SGLang's FP32 LM head is explicitly enabled.

Let `p` be the current trainer distribution and `q` the recorded sampler.
Official TIS uses the detached weight `f(p/q)=min(p/q,2)`, centered together
with the score. Alternatives are `--score-centering-is none` (`f=1`) and `mis`
(`f=r` for `0.5<=r<=5`, zero otherwise). The SC IS switch supplies the weighting;
the generic `--use-tis` switch is not combined with it.

The saved candidate head `H` and the official proportional-tail approximation
form the following per-token loss. All coefficients marked `stop` are detached:

```text
P_H   = sum_H p(v)                         Q_H = sum_H q(v)
rho   = max(1-Q_H,1e-6) / max(1-P_H,1e-6)
alpha = rho * f(1/rho)
c_v   = q(v)*f(p(v)/q(v)) - alpha*p(v)
L_t   = -stop(A_t) * [stop(f(p(a)/q(a)))*log p(a)
                     - sum_H stop(c_v)*log p(v)]
```

SC is a REINFORCE-style actor objective: **PPO policy clip bounds do not apply**.
The critic still uses value clipping. The top-128 tail model is an
approximation; monitor head masses, tail ratio, importance weights, and
train/rollout probability differences. The existing math and IS modes are
described in [PPO_SCORE_CENTERING.md](PPO_SCORE_CENTERING.md) and the
[official Miles reference](https://github.com/radixark/miles/blob/ff97a63b6971f4c2f176d7d99f902163aac7dc01/docs/examples/infra-features/score-centering.md).

The async launchers also accept `--no-score-centering`, selecting its existing
clipped-PPO path with `--keep-old-actor` and bounds `[0.8,1.28]`. The synchronous
launchers select their fixed PPO configuration directly.

## Critic targets and inverse-STD regression

All recipes set `gamma=lambda=1` and reward KL=0. Every trainable token's return
is the episode terminal reward `R`, and its advantage is `R-V_old,t`. Values
are aligned to the position before the predicted token. Values, returns, and
advantages are detached inputs to the actor objective.

Before any of the four critic updates, the critic scores **all** accepted
episodes and freezes `V_old`. Every mini-batch uses that same snapshot for
value clipping, and the actor uses it for GAE after the critic finishes. Each
backward rebuilds memory using the current critic weights; caches and graphs
are not reused across optimizer updates.

The existing rollout-conversion helper computes each prompt group's population
return standard deviation, counting an episode once across its segments:

```text
std_g   = sqrt(population_variance(rewards_g))
raw_w_g = 1 / max(std_g,0.075)
w_g     = raw_w_g / mean_valid_prompt_groups(raw_w)
Vclip_t = V_old,t + clip(V_t-V_old,t,-0.2,0.2)
L_value = 0.5 * sum_valid_tokens(w_g * max((V_t-R)^2,(Vclip_t-R)^2))
          / global_unweighted_valid_token_count
```

Weights are computed and normalized over the full accepted round, before DP
sharding and mini-batch splitting. Empty-mask samples are excluded from the
statistics. The weights scale the critic-loss numerator only; rewards and
actor advantages retain their values. The STD floor is 0.075, the variance
exponent is 0.5, and there is no additional weight cap.

Actor-only truncation filtering preserves truncated episodes for critic
training and return statistics, then zeros their actor masks. A critic-only
round can contain all truncated episodes; an actor round requires valid actor
tokens. Loss reduction is per token. Critic explained variance uses global
token moments.

The actor LR is `1e-6`; critic LR is `5e-6`. Both use Adam-family optimizers with
betas `(.9,.999)`, epsilon `1e-8`, weight decay `.01`, constant LR, and global
gradient-norm clipping at 1. Dropout is zero.

## Critic memory and token alignment

**`actor-summary`** reads the actor's actual context for each segment: the task,
current summary, and local conversation. Attention KV, recurrent state, and
convolution state are reset between segments.

**`full_detach`** retains the critic's own attention KV, recurrent state, and
convolution state across segments. All three are detached at the segment
boundary. Later segments can use earlier information without backpropagating
through earlier segments. The backbone and value head remain trainable within
each segment.

After explicit actor compaction, the critic's chronological stream uses a
short continuation instead of a second system/task/summary resume prompt.
The summary's original generation is already in history, so its copied resume
text is not added again. Newly observed tool output is preserved. A non-prefix
boundary without an explicit compaction record is rejected in `full_detach`.
Actor tokens, masks, and sampler probabilities retain their original layout.

Completed assistant messages use EOS-anchored token-in: when the next canonical
prompt has the exact recorded completed prefix, the actual sampled IDs replace
its re-encoded copy. Template whitespace, observations, and generation glue
after EOS remain intact. Explicit compaction clears the anchor.

Generated action and summary tokens have mask 1. Tool outputs, template glue,
and summary copies in subsequent inputs have mask 0. The trace stores exact
response spans and candidate rows; masked gaps use `-1/-inf` sentinels.
Candidates are stored in `easyppo-topk.npz` alongside the trajectory JSON.

## Context, summaries, and generation budget

The actor's per-call prompt+generation window is **32768 tokens**. An action
can generate at most 8192 tokens and a summary at most 2048. With current prompt
length `L` and summary-request overhead `O`, compaction triggers at:

```text
L >= min(24576,32768-max(8192,2048+O))
```

The actor then resumes from the original task plus the sampled progress
summary. Each call's limit is further bounded by its remaining context and
remaining episode generation budget.

The **120000-token cumulative generation cap** includes actions, thinking,
EOS, and summaries. It excludes prompts and tool outputs. Other limits are
300 action-loop steps, 120 seconds per shell command, and 16000 characters
per tool output. Harbor's task-specific agent/verifier timeouts also apply.

The **262144-token training packing budget** checks the critic's history and
the total actor-segment length for an episode. Tool outputs and repeated actor
contexts count toward training length. Segments are packed into the same actor
forward; full-detach memory also grows with episode history. Over-budget
training episodes raise an explicit error.

At K128, the candidate IDs and FP32 log-probs occupy about 1 KiB per response
position. A batch with 512 episodes generating 120000 tokens each needs about
58.6 GiB for those generated-position arrays, before tool gaps, transport,
and logs. Plan shared storage and host memory for the actual response length.

## GPU allocation and offload

| Recipe / allocation | Actor training | Rollout |
|---|---|---|
| Async default: 8 nodes × 8 B200 | 16 GPUs, TP4 × DP4, PP=CP=1 | 48 GPUs, 12 TP4 engines |
| Sync default: 4 nodes × 8 B200 | 8 GPUs, TP4 × DP2, PP=CP=1 | 24 GPUs, 6 TP4 engines |

Each recipe uses disjoint training and rollout pools. Join the corresponding
8-node async or 4-node sync Ray allocation before launch. Async can also use the
32-GPU layout with `--num-nodes 4 --trainer-num-nodes 1`; sync can request the
64-GPU layout with `--num-nodes 8 --trainer-num-nodes 2`.

The global actor batch remains 512 episodes and each critic mini-batch remains
128 episodes in both defaults. Async DP4 assigns 128 actor episodes per
logical DP shard per round and 32 per critic step. Sync DP2 assigns 256 actor
episodes per logical DP shard per round and 64 per critic step. TP siblings
replicate the shard's data.

The independent HF critic runs sequentially on the training GPUs. Each rank
holds a full BF16 forward model and gradients; `A_log` stays FP32 from load.
FP32 master weights and Adam moments are sharded on CPU. Gradient reduction
uses bounded FP32 chunks, followed by global clipping, owner updates, and
parameter broadcasts. Actor TP siblings duplicate the critic forward;
gradient and token-count normalization account for that duplication.

Actor training uses sequence parallelism, a distributed optimizer, and
full-layer recomputation. Actor/critic CPU offload alternates their GPU
residency. Each training node should budget roughly 1.5–2 TB host RAM.
SGLang uses a static memory fraction of .7 and concurrency 128 per engine.
Docker sandboxes and judges consume additional CPU/RAM on rollout hosts.

## Rewards, validation, and checkpoints

Native rewards are the default. FrontierSmith/FCS use checker partial score
scaled from `[0,100]` to `[0,1]`; TMAX/TB use their task reward. Missing solutions
and contestant compilation failures score zero. Infrastructure and verifier
service failures propagate as trial errors. Optional `test-count` mode reads
`passing_assertions/20`; a missing/unparseable count can fall back only to a
valid binary result, with the reason saved in metadata.

Validation defaults to the full evaluation set (`eval_max_tasks=0`) with five
episodes per task (`eval_samples_per_prompt=5`), sampled at `1/1/-1`. Each point
runs **179 × 5 = 895 Frontier-CS trials** or **89 × 5 = 445 TB2.1 trials** and uses
result names `fcs` / `tb21`.

A fresh 120-round run evaluates initially and at rounds 10 through 120: 13
points, **11,635 Frontier-CS trials** or **5,785 TB2.1 trials per experiment**.
Checkpoints are saved at rounds 10 through 120, giving 12 points.
Epoch-triggered extra eval/save is disabled.

For a quick subset, set `--eval-max-tasks N` with positive N. Selection and order
are fixed by seed 42, and names become `fcs_quick` / `tb21_quick`.
`--eval-samples-per-prompt N` independently controls trials per task.
Smoke runs use `--eval-max-tasks 1 --eval-samples-per-prompt 1`.

Evaluation logs include `eval/<name>/episode_raw_reward`, the mean over numeric
returned episodes, and `eval/<name>/num_training_samples`, the number of physical
segments. The top-level `eval/<name>` is a row mean, and
`eval/<name>-none_reward_ratio` records missing row rewards.

Actor and critic checkpoints are separate. Native critic checkpoints include
model/head, FP32 CPU master/Adam state, RNG, outer/update counters, and context
mode/topology. Resume uses the same run ID plus its explicit `--critic-load`,
with unchanged trainer world size, actor TP, and critic mode. A fresh run can
load critic weights with `--critic-weights-only`; this refreshes master weights
and starts new optimizer/step state. See the [resume commands](../../README.md#6-evaluate-and-resume).

## Validation status

CPU regression exercises memory/value alignment, four-step critic scheduling,
CPU optimizer/checkpoint behavior, terminal cleanup, data conversion, and the
official SC probability/loss pipeline. Long-budget tests exercise the real
terminal loop with stub decoding. Full 27B GPU forward, target-cluster judges,
B200 long-sequence capacity, offload timing, and throughput remain cluster
validation tasks.
