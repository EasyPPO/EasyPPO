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
    summary_tokens: int = 6144
    continue_on_length: bool = True
