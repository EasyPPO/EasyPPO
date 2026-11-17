"""Harbor agent exposing only C++ submissions to the shared Frontier-CS judge."""

import json
from pathlib import Path

from harbor.utils.env import resolve_env_vars

from agentic_critic.fcs_judge import CaseResult, JudgeResult
from agentic_critic.fcs_terminal import FcsSubmissionAgent
from agentic_critic.harbor_terminal import MiniSweTerminalAgent
from scripts.frontiersmith_verifier import _reward


class HarborFcsAgent(MiniSweTerminalAgent):
    @staticmethod
    def name() -> str:
        return "easyppo-fcs-submit"

    def version(self) -> str:
        return "easyppo-fcs-submit-1"

    def bind_judge(self, *, verifier_env: dict, max_submissions: int, max_turns: int) -> None:
        self.judge_env = resolve_env_vars(verifier_env)
        if self.judge_env.get("FCS_ITERATIVE_SUBMISSIONS") != "1":
            raise ValueError("Reprepare FCS tasks for iterative submissions before training")
        self.max_submissions, self.max_turns = max_submissions, max_turns

    async def setup(self, environment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ledger = self.logs_dir / "initial-submissions.json"
        ledger.write_text("[]", encoding="utf-8")
        await environment.upload_file(ledger, "/app/fcs-submissions.json")
        client = Path(__file__).resolve().parents[1] / "scripts/frontiersmith_verifier.py"
        await environment.upload_file(client, "/app/easyppo-fcs-submit.py")

    def _make_agent(self, environment):
        async def submit(code: str) -> JudgeResult:
            source = self.logs_dir / "solution.cpp"
            source.write_text(code, encoding="utf-8")
            await environment.upload_file(source, "/app/solution.cpp")
            result = await environment.exec(
                command="python3 /app/easyppo-fcs-submit.py --submit",
                env=self.judge_env,
                timeout_sec=float(self.judge_env.get("MAX_POLL_TIME", "600")) + 90,
            )
            if result.return_code != 0:
                raise RuntimeError(f"FCS submission failed: {(result.stderr or result.stdout or '')[-2000:]}")
            response = json.loads(result.stdout)
            score = 100 * _reward(response)
            if response["status"] == "error":
                message = str(response.get("error", ""))
                if message.startswith("Error: compile failed: "):
                    return JudgeResult(score, "Compile Error", compile_msg=message)
                return JudgeResult(score, "Judge Error")
            cases = [
                CaseResult(
                    name=str(i),
                    ratio=float(case.get("scoreRatio") or 0),
                    status="Accepted" if case.get("scoreRatio") == 1 else "Wrong Answer",
                    time_s=float(case.get("time") or 0) / 1e9,
                )
                for i, case in enumerate(response.get("cases") or [], 1)
            ]
            return JudgeResult(score, "done", cases=cases)

        return FcsSubmissionAgent(
            tokenizer=self.tokenizer,
            decode=self.decode,
            submit=submit,
            config=self.episode_config,
            max_submissions=self.max_submissions,
            max_turns=self.max_turns,
        )
