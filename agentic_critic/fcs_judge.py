"""Submission feedback shared by the Frontier-CS judge adapters."""

from dataclasses import dataclass, field


@dataclass
class CaseResult:
    name: str
    ratio: float
    status: str  # Accepted | Wrong Answer | Time Limit Exceeded | Runtime Error | Output Limit Exceeded
    time_s: float
    msg: str = ""


@dataclass
class JudgeResult:
    score: float  # 0..100
    status: str  # done | Compile Error | Judge Error
    compile_msg: str = ""
    cases: list[CaseResult] = field(default_factory=list)
