# Validation record: problem / AM history / resume context

Date: 2026-11-03. Base commit: `3c10130bdf220653d361a839b033b65b30385fee`.

## Result

361 selected CPU tests passed; 3 environment-dependent validation tests were
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
  tests/agentic_critic/test_native_am_gpu.py \
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

Use a fresh output directory and record `problem_am_resume_v2`; historical HF
AM training and older frozen AM inputs used a different context and backend.

## Publication review, 2026-11-03

Fixed a missing transport field: `_package_shards` must copy per-sample metadata
along with token rows. Added 1/2/4-rank sharder/packager tests and an optimizer
schedule test; these execute the actual function bodies with CPU-only runtime
stand-ins. They verify metadata survives, episodes stay ordered and complete,
and optimizer steps do not separate sibling segments.

## Official-alignment revision, 2026-11-03

The AM path now calls unmodified upstream HighestAttnKeys selection/NNLS and C2
least squares. The V0 ridge, selector monkeypatch, fixed per-boundary seed,
forced-FP32 inputs and 128-query reader are removed from AM. Native reads use TE
post-scale bias and bottom-right causal attention; HF remains a dense frozen
reference. The implementation schema is `problem_am_resume_v2`.

New CPU tests compare FP32/BF16 with upstream public results for unbiased history,
independently check absolute mass for repeated biased history, exercise native
BSHD/beta/causal ingress and gradient mapping, and check scoped HF registration.
`test_native_am_gpu.py` was skipped because this environment has CPU-only PyTorch.
Its real Megatron/TE CUDA forward/backward is still required before claiming
production correctness, peak memory, speed or 9B convergence.

See [remaining implementation audit](critic-am-official-audit.md). Existing legacy
KV ablation math tests still exercise `mode=kv`, not the restored official AM path.

The selected AM suite reports **361 passed, 1 CUDA test skipped, 3 dependency-blocked
checks deselected**. The affected Frontier-CS launcher tests and default snapshots
report **9 passed** (including the new AM backend selection case).

The unmodified root launcher command still fails collection because IPython is
missing through the SGLang fixture dependency. Running with a narrower conftest
boundary allows collection, but the broad suite has **66 failures and 52 errors**
(including /root/models permission failures). An identical run on the prior tree
has exactly the same failing/error test identities: 455 pass on the prior tree,
456 pass on the updated tree. No new launcher failure/error was observed. This is
not a clean launcher-suite pass.
