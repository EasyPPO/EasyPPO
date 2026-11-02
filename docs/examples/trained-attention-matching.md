# Native Miles attention-matching critic

AM trains through the same Miles/Megatron critic lifecycle as full-history and
actor-summary: the optimizer, prompt weights, shuffled episode mini-batches,
value clipping, checkpointing and resident-model validation are retained.
The previous separate HF AM optimizer is rejected. HF is a frozen reference only.

## Official algorithm and attention backend

The pinned upstream HighestAttnKeys demo configuration is used without replacing
its methods: RMS, NNLS refinement=2, weight bounds 0.05/20, C2 least squares and
ridge_lambda=0. Native memory reads call Megatron/TE with post-scale beta and
bottom-right causal visibility. There is no 128-query Python SDPA loop. AM selects
`--attention-backend auto`; Flash-only is rejected because it cannot express beta.
HF is a frozen dense additive-mask reference, not an independent AM trainer.

Repeated history requires converting upstream NNLS incremental beta to absolute
beta before upstream C2 fitting. This is a documented cache adapter, not an
unqualified claim that the entire official public wrapper is copied unchanged.

See [implementation audit](critic-am-official-audit.md) for removed deviations,
remaining task-specific choices, and GPU validation limits. Use a fresh output
directory; `problem_am_resume_v2` identifies this corrected implementation.

## Context contract

A resumed segment's full-attention read is:

```text
[current problem and shared template prefix]
[detached AM history]
[original resume suffix: summary preamble + summary + continuation instruction + assistant template]
[current segment responses and feedback, causally]
```

The problem prefix is the exact token prefix shared by the original initial and
resume prompts. BPE may leave a boundary delimiter in the suffix; the entire
suffix is retained verbatim, with no decoded-text search or retokenization.
No actor tokens, response lengths, masks, rewards or value positions change.

Each segment recomputes its problem prefix once, without attending to old
memory. Its suffix attends to that fresh problem, AM memory and its own causal
history. A whole-segment RoPE offset advances logical history while preserving
all relative positions inside the actor's original prompt. The fresh problem
replaces the previous problem physically; it is never added to compressed memory.

At a boundary, the original generated summary queries probe the history block.
AM compresses previous AM memory plus current presummary tokens after the problem.
The generated summary's full KV is discarded after its own values are evaluated.
The next segment supplies its summary exactly once via its actual resume prompt.
Older resume summaries may subsequently be represented inside compressed history;
there is no additional uncompressed copy of the current generated summary.
Source biases participate in selection, mass fitting and official C2 least squares.
The former custom V0 ridge and selector monkeypatch are removed from native AM.

**Linear attention follows actor-summary.** Every original segment remains an
independent packed sequence with convolution and recurrent state reset. No HF
cross-segment cache monkeypatch participates in native training.

The three-block layout applies only to full-attention reads, not to model input
tokens. Linear layers receive the original problem/resume/response tokens and
never consume AM K/V as synthetic tokens. A later linear layer can receive history
indirectly through hidden states produced by an earlier full-attention layer.
Resetting GDN state preserves the actor-summary boundary policy; it does not
reconstruct the old linear state or prove that this policy is optimal. Carrying
old state while replaying the question and summary would be a different policy.

AM history fitting is detached. Ratio 1 is a detached-history control with this
new context layout, not the old fully differentiable append-only full baseline.

## Existing data

Reuse the original `replay/rollout_*.pt`, the frozen validation plan and original
events. There is no new actor collection or judge run. Older replay without
`summary_span` requires its original `episodes.jsonl`; tokens and masks must match
exactly. Validation recovers offsets from its own cohort source, not training events.

## Run

Initialize the pinned dependency:

```bash
git submodule update --init third_party/compaction
```

Use the existing Miles runtime and converted model paths:

```bash
python scripts/agentic_critic/miles_train.py \
  --model "$MODEL" --model-dir "$CONVERTED_MODEL_DIR" \
  --rollouts "$REPLAY_DIR" --prompt-data "$REPLAY_DIR/prompts.jsonl" \
  --output "$NEW_AM_OUTPUT" --mode attention_matching \
  --validation-plan "$VALIDATION_PLAN" \
  --critic-context-events "$REPLAY_DIR/episodes.jsonl" \
  --critic-am-ratio 0.1 --steps 50 --gpus 4 --tensor-parallel 1 \
  --problems-per-batch 32 --episodes-per-problem 16 --mini-batches 4
```

The native model spec is `miles_plugins.models.qwen3_5 get_qwen3_5_spec`.
Initial supported topology is TP=PP=CP=1 with arbitrary data-parallel replicas,
THD dynamic complete-episode batches, no MTP, zero attention dropout and standard
split-half RoPE. Unsupported configurations fail at launch. Each native core call
rebuilds local memory from its immutable packed metadata; checkpoint recomputation
does not depend on mutable caches from an earlier call. Actor calls have no AM
side channel and continue through the original TE core.

`miles_run.json` records `problem_am_resume_v2` and per-segment linear resets.
AM validation dumps identify the context mode rather than labeling it actor-summary.
Use a fresh output directory: earlier HF results used a different context and backend.

## Validation status

CPU tests cover original-token preservation, inclusion of the continuation
instruction, exclusion of raw problem/summary KV from history, independent problem
reads, repeated fitting, logical positions, immutable memory, detached gradients
and checkpoint output/gradient equivalence. Native GPU Megatron/Transformer Engine
training must still be smoke-tested in the pinned production environment; CPU tests
do not establish production throughput, CUDA numerics or recovered 9B EV.

See [task decisions](critic-context-task-notes.md) for the user's requirements and
why the earlier HF experiment is not a controlled Miles comparison.
