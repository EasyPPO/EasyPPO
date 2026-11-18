"""Generation defaults for FrontierSmith training and Frontier-CS evaluation."""

from dataclasses import dataclass

DEFAULT_SGLANG_SERVER_CONCURRENCY = 128
SYNC_EVAL_SAMPLES_PER_PROMPT = 3
SYNC_TERMINAL_ENVIRONMENT = "shared-judge"
DOCKER_JUDGE_URL = "http://host.docker.internal:8081"


@dataclass
class GenerationArgs:
    max_context_tokens: int = 131072
    compact_threshold: int = 12000
    max_new_tokens: int = 24576
    summary_tokens: int = 4096
    continue_on_length: bool = True
