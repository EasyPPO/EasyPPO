# Miles AM bug fix validation

Updated 2026-11-03. This branch ports the native AM context fix without restoring
the separate offline experiment pipeline from actor-summary-compact.

## Passed

314 selected CPU tests passed. They cover native memory reads/fits, checkpoint
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
