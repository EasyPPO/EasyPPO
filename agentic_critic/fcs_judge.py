"""Convert an official Frontier-CS verdict into feedback for the next action."""

from dataclasses import dataclass, field

from scripts.frontiersmith_verifier import _reward


@dataclass
class JudgeResult:
    score: float
    status: str
    compile_msg: str = ""
    case_statuses: list[str] = field(default_factory=list)


def judge_feedback(response: dict) -> JudgeResult:
    score = 100 * _reward(response)
    if response["status"] == "error":
        message = str(response.get("error", ""))
        if message.startswith("Error: compile failed: "):
            return JudgeResult(score, "Compile Error", compile_msg=message)
        return JudgeResult(score, "Judge Error")
    statuses = [
        str(case.get("status") or ("Accepted" if case.get("scoreRatio") == 1 else "Wrong Answer"))
        for case in response.get("cases") or []
    ]
    return JudgeResult(score, "done", case_statuses=statuses)
