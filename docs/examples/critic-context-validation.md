# Miles AM bug fix validation

Updated 2026-11-03. This branch ports the native AM context fix without restoring
the separate offline experiment pipeline from actor-summary-compact.

## Passed

318 selected CPU tests passed. They cover native memory reads/fits, checkpoint
output and gradient agreement, HF frozen hybrid reset semantics, legacy cache
compatibility, exact summary offsets, actor value alignment, Frontier-CS rollouts,
and terminal dataset preparation.

New transport tests execute the actual sharder and packager with 1/2/4 DP ranks:
layout metadata must survive and each micro-batch must contain a complete ordered
episode. The actual critic schedule is exercised with CPU parallel-state stand-ins
to verify episode shuffle, optimizer-step boundaries and token-budget refusal.
These tests do not instantiate a real process group or CUDA Megatron model.

```bash
KV_COMPACTION_REPO=/path/to/pinned/compaction OMP_NUM_THREADS=2 python -m pytest \
  --confcutdir=tests/agentic_critic -q \
  tests/agentic_critic/test_context_memory.py \
  tests/agentic_critic/test_native_am_gpu.py \
  tests/agentic_critic/test_critic_modes.py \
  tests/agentic_critic/test_critic_train.py \
  tests/agentic_critic/test_fcs_env.py \
  tests/agentic_critic/test_critic_input.py \
  tests/agentic_critic/test_miles_samples.py \
  tests/agentic_critic/test_miles_fcs.py \
  tests/agentic_critic/test_terminal_data.py
```

## Remaining checks

Terminal agent tests could not collect because minisweagent is absent (two
optional terminal modules skip; test_terminal_thinking fails import). The native
full-history sharder suite requires unavailable SGLang transitive dependencies.
Neither failure is a test pass. Existing terminal and full-history implementations
were retained. AM is not enabled for the terminal adapter's separate trace format.

The production Megatron/Transformer Engine/FLA forward/backward, checkpoint
save/load, multi-rank DP and 9B convergence remain unverified. No training effect,
production memory budget or throughput claim follows from these CPU checks.

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

The selected suite reports **318 passed, 1 CUDA test skipped**. The omitted terminal
agent tests still require unavailable minisweagent. Shared and full-history paths
were retained; the AM native backend choice is the only launcher behavior change.

Affected Frontier-CS launcher checks: **5 passed**, including AM auto backend selection.

## Post-alignment bias-layout correction

TE fused C++ forwards raw bias data and shape without PyTorch strides; its
cuDNN descriptor computes dense strides from the actual shape. Replaced the
expanded zero-stride [1,H,Lq,Lk] bias view by a real contiguous [1,H,1,Lk] tensor.
The wrapper test now checks this native boundary contract as well as outputs and
gradients. The selected context suite: 22 passed; CUDA smoke: 1 skipped. This is
source/CPU evidence, not a completed native CUDA compatibility or throughput test.
