# Validation record: problem / AM history / resume context

Date: 2026-11-02. Base commit: `3c10130bdf220653d361a839b033b65b30385fee`.

## Result

357 selected CPU tests passed; 3 environment-dependent validation tests were
excluded after their imports failed. Static undefined-name/import checks and
`git diff --check` passed. Frozen HF evaluation also ran through its CLI for all
three modes, wrote paired outputs and rendered figures using a tiny model.

Environment: Python 3.13, PyTorch 2.13.0+cpu, Transformers 5.19.0. The upstream
compaction checkout was pinned at `2020f6d37940fe013e7e7f1ebf409e5d107c5670`.

The new context tests include:

- Exact preservation of actor tokens, target masks and resume continuation tokens.
- No-compaction attention equivalence with actor-summary.
- A fresh problem prefix independent of old AM memory.
- Exclusion of problem and generated-summary full KV from persistent history.
- Repeated biased memory fitting and immutable input memory.
- Whole-segment logical RoPE shifts preserving local relative positions.
- FP32 and BF16 checkpoint output/gradient equivalence and detached past gradients.
- Original-event recovery for legacy replay, with token/mask checks.
- Explicit native critic metadata, actor TE fallback and the TE call signature.
- Hybrid reference forwards resetting every linear-attention state per segment.
- Refusal of the retired standalone HF AM optimizer.

Legacy KV-compactor math tests now explicitly exercise the legacy `kv` ablation
mode. They do not assert that the new native AM context preserves raw summary KV
or that ratio 1 equals the old append-only full history. Frozen comparison tests
use the new context contract. HF baseline/reference cache helpers also include
the Transformers 5.19 convolution-return compatibility fix.

## Selected test command

```bash
KV_COMPACTION_REPO=/path/to/pinned/compaction OMP_NUM_THREADS=2 python -m pytest \
  --confcutdir=tests/agentic_critic -q \
  tests/agentic_critic/test_context_memory.py \
  tests/agentic_critic/test_critic_modes.py \
  tests/agentic_critic/test_critic_train.py \
  tests/agentic_critic/test_attention_matching.py \
  tests/agentic_critic/test_frozen_am_validation.py \
  tests/agentic_critic/test_trained_am_comparison.py \
  tests/agentic_critic/test_fcs_env.py \
  tests/agentic_critic/test_critic_input.py \
  tests/agentic_critic/test_miles_samples.py \
  tests/agentic_critic/test_miles_fcs.py \
  tests/agentic_critic/test_validation.py \
  -k 'not validation_packs_keep_whole_episodes and not validation_driver_finishes_all_chunks'
```

## Remaining validation

The three excluded tests are the two shared/full variants of
`test_validation_packs_keep_whole_episodes_across_eight_ranks` and
`test_validation_driver_finishes_all_chunks_before_logging`. The required launcher
suite (`tests/fast/launch_scripts tests/manual/launch_scripts`) and
`test_validation_failures.py` also could not collect in this environment.
Initially SGLang was absent; installing its pinned package in an isolated directory
revealed another missing transitive dependency, IPython. These are recorded as
blocked checks, not passes.

The native TE wrapper ingress tests use a CPU stand-in for TE and real tensor
memory operations. They do not instantiate a CUDA Megatron model. A production
smoke test must still verify the actual pinned Megatron/TE/FLA stack, checkpoint
save/load, resident critic validation, multi-rank DP and memory cost. No recovered
9B EV, production throughput, or completed native GPU training is claimed.

Use a fresh output directory and record `problem_am_resume_v1`; historical HF
AM training and older frozen AM inputs used a different context and backend.

## Publication review, 2026-11-03

Fixed a missing transport field: `_package_shards` must copy per-sample metadata
along with token rows. Added 1/2/4-rank sharder/packager tests and an optimizer
schedule test; these execute the actual function bodies with CPU-only runtime
stand-ins. They verify metadata survives, episodes stay ordered and complete,
and optimizer steps do not separate sibling segments.
