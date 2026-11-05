"""Load the HF critic's BF16 Qwen3.5 backbone while preserving FP32 A_log.

The precision override participates in HF checkpoint loading itself. Casting an
already-rounded BF16 A_log back to float would not restore the checkpoint value.
"""

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


class _MemoryQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    # Match the Qwen3.5 parameter contract used by the Megatron bridge and
    # SGLang. HF 5.12.1 does not declare this strict precision requirement.
    _keep_in_fp32_modules_strict = set(Qwen3_5ForCausalLM._keep_in_fp32_modules_strict or ()) | {"A_log"}


def load_memory_backbone(checkpoint: str):
    language_model = _MemoryQwen3_5ForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # Dropping the causal-LM wrapper also drops the unused vocabulary head.
    return language_model.model
