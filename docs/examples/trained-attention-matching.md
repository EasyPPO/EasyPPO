# Native Miles attention-matching critic

AM trains through the same Miles/Megatron critic lifecycle as full-history and
actor-summary: the optimizer, prompt weights, shuffled episode mini-batches,
value clipping, checkpointing and resident-model validation are retained.
The previous separate HF AM optimizer is rejected. HF is a frozen reference only.

## Official algorithm and native reader

Uses pinned upstream HighestAttnKeys methods without monkeypatches or custom V0
ridge. Memory reads delegate to native Megatron/TE with additive beta and
bottom-right causal visibility; no 128-query Python reader remains. AM selects
`--attention-backend auto`; Flash-only is rejected. The corrected implementation
uses `problem_am_resume_v2`, and must use fresh output directories.

See [implementation audit](critic-am-official-audit.md) for exact upstream settings,
necessary task adapters, repeated-history beta conversion and remaining risks.

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
Source biases participate in official selection, mass fitting and C2 least squares; custom V0 ridge is removed.

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

## Enable on the miles branch

Initialize the pinned algorithm with `git submodule update --init third_party/compaction`.
Use the existing native Miles launcher and converted checkpoints. For example,
append these flags through its existing `--extra-args` option:

```text
--critic-context-mode attention_matching --critic-am-ratio 0.1
```

The model spec must be `miles_plugins.models.qwen3_5 get_qwen3_5_spec`.
Use TP=PP=CP=1, THD, dynamic batches, zero attention dropout, standard split-half
RoPE and synchronous training without MTP. Whole episodes stay on one DP rank
and in one forward micro-batch; critic optimizer mini-batches shuffle episodes.
This initial AM sharder requires exactly one complete logical batch per rollout.

Frontier-CS summary rollouts emit exact `summary_span` metadata. Legacy replay
must supply `--critic-context-events /path/to/original/episodes.jsonl`; token IDs,
masks and summary offsets are verified against those original events. Other
rollout adapters must provide complete episode/summary metadata before using AM;
this patch does not enable AM on the terminal adapter's distinct trace format.

The miles branch does not contain the offline experiment pipeline or its resident
validation launcher. They are not restored by this fix. Its existing shared and
full-history routes remain intact; AM value dumps include the new context schema.
HF reference readers reject AM training; native Miles owns the optimizer.

## Validation status

CPU tests cover original-token preservation, inclusion of the continuation
instruction, exclusion of raw problem/summary KV from history, independent problem
reads, repeated fitting, logical positions, immutable memory, detached gradients
and checkpoint output/gradient equivalence. Native GPU Megatron/Transformer Engine
training must still be smoke-tested in the pinned production environment; CPU tests
do not establish production throughput, CUDA numerics or recovered 9B EV.

See [task decisions](critic-context-task-notes.md) for the user's requirements and
why the earlier HF experiment is not a controlled Miles comparison.
