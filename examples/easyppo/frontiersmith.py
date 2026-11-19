"""Generation defaults for FrontierSmith training and Frontier-CS evaluation."""

from dataclasses import dataclass

DEFAULT_SGLANG_SERVER_CONCURRENCY = 128
SYNC_EVAL_SAMPLES_PER_PROMPT = 3
SYNC_TERMINAL_ENVIRONMENT = "shared-judge"
DOCKER_JUDGE_URL = "http://host.docker.internal:8081"


@dataclass
class GenerationArgs:
    model_name: str = "Qwen3.8-27B"
    megatron_model_type: str = "qwen3.8-27B"
    hf_repo: str = "Qwen/Qwen3.8-27B"
    hf_revision: str = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
    chat_template_path: str = ""
    max_context_tokens: int = 131072
    compact_threshold: int = 12000
    max_new_tokens: int = 24576
    tensor_model_parallel_size: int = 4
    max_total_tokens: int = 120000
    summary_tokens: int = 6144
    continue_on_length: bool = True
    sglang_server_concurrency: int = DEFAULT_SGLANG_SERVER_CONCURRENCY


@dataclass
class Generation9BArgs(GenerationArgs):
    model_name: str = "Qwen3.5-9B"
    megatron_model_type: str = "qwen3.5-9B"
    hf_repo: str = "Qwen/Qwen3.5-9B"
    hf_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    tensor_model_parallel_size: int = 2
    max_total_tokens: int = 150000
    summary_tokens: int = 8192
    fcs_summary_thinking: bool = False
    sglang_server_concurrency: int = 4
