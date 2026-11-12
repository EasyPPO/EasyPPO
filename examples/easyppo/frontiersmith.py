"""Generation defaults for FrontierSmith training and Frontier-CS evaluation."""

from dataclasses import dataclass


@dataclass
class GenerationArgs:
    max_context_tokens: int = 131072
    compact_threshold: int = 12000
    max_new_tokens: int = 24576
    summary_tokens: int = 3072
    continue_on_length: bool = True
