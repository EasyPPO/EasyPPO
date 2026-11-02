# Critic context: task decisions

Updated 2026-11-02. This records the user's decisions and the evidence that led to this change.

- Train the AM critic with the existing Miles/Megatron optimizer, loss, episode shuffle, checkpoint and validation lifecycle. A separate HF optimizer loop is not an equivalent third experiment.
- At the start of a resumed segment, full attention sees `[problem and template] + [AM history] + [resume summary, continuation instruction and assistant template]`. Keep the continuation instruction.
- Read the summary from the actual next actor resume prompt. Do not additionally preserve the previous summary completion as an uncompressed KV suffix.
- Linear attention follows actor-summary: reset convolution and recurrent state at each original segment boundary. Do not carry detached historical hybrid states across segments.
- The user explicitly confirmed that the continuation instruction must be retained and that linear states must match actor-summary. Recompute each segment's problem once; do not keep an older raw problem KV block alongside it. One common whole-segment RoPE offset preserves the original prompt's internal relative positions.
- Preserve original actor tokens, response lengths, masks, rewards and value alignment. Identify the common problem prefix using exact prompt token IDs, not decoded text or regenerated summaries. The original resume suffix remains intact.
- AM history is detached; full versus AM therefore also needs a full-detach gradient control for algorithm attribution.
- The previous run used e13c557 plus 19 Horizon patches. Its HF read-only convolution stub violated the Transformers 5.19 return contract; CPU FP32/BF16 reproductions confirmed this. Native Miles segment packing avoids that cross-segment HF cache path.
- Frozen same-checkpoint evaluation remains useful. Existing trained HF AM EV cannot establish AM versus Miles algorithm quality.
- Implementation and CPU tests do not establish 9B CUDA/Transformer Engine correctness or recovered EV. Report those limits explicitly.
