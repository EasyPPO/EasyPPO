"""Harbor lifecycle adapter for the mini-swe-agent EasyPPO harness.

Requires the optional terminal dependencies. A fresh adapter is bound to one
in-process episode; no global registry or external agent server is needed.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from agentic_critic.terminal_env import TerminalAgent, TerminalTrace


class MiniSweTerminalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "easyppo-mini-swe-agent"

    def version(self) -> str:
        return "mini-swe-agent-1.17.1/easyppo-1"

    async def setup(self, environment) -> None:
        # The harness runs on the rollout worker. Only bash runs in the sandbox.
        pass

    def bind(self, *, tokenizer, decode, config) -> None:
        self.tokenizer = tokenizer
        self.decode = decode
        self.episode_config = config
        self.trace = TerminalTrace()

    async def run(self, instruction, environment, context) -> None:
        async def execute(command: str, timeout: int) -> dict:
            try:
                result = await environment.exec(command=command, timeout_sec=timeout)
            except RuntimeError as exc:
                # Harbor 0.24's Docker/Podman adapter wraps its asyncio timeout.
                # Normalize only that wrapper; infrastructure failures still propagate.
                if str(exc) == f"Command timed out after {timeout} seconds" and isinstance(
                    exc.__context__, TimeoutError
                ):
                    raise TimeoutError(str(exc)) from exc
                raise
            return {"output": (result.stdout or "") + (result.stderr or ""), "returncode": result.return_code}

        agent = TerminalAgent(
            tokenizer=self.tokenizer, decode=self.decode, execute=execute, config=self.episode_config
        )
        self.trace = agent.trace
        try:
            await agent.run_async(instruction)
        except asyncio.CancelledError:
            self.trace.exit_status = "Cancelled"
            self.trace.truncated = True
            raise
        except Exception:
            self.trace.exit_status = "AgentError"
            raise
        finally:
            # Persist partial traces too (including cancellation and Harbor timeouts).
            context.n_input_tokens = sum(c["prompt_tokens"] for c in self.trace.calls)
            context.n_output_tokens = self.trace.generated_tokens
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "easyppo-trajectory.json").write_text(
                json.dumps({"trace": asdict(self.trace), "messages": agent.messages}, ensure_ascii=False)
            )


async def run_trial(*, task_path: Path, trials_dir: Path, environment: str, tokenizer, decode, config):
    """Run native Harbor setup -> agent -> verifier -> teardown, without retries.

    Infrastructure/verifier failures are errors, not zero-reward training data.
    Harbor handles cancellation cleanup in Trial.run().
    """
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_path),
            trials_dir=trials_dir,
            agent=AgentConfig(import_path="agentic_critic.harbor_terminal:MiniSweTerminalAgent"),
            environment=EnvironmentConfig(type=environment),
        )
    )
    trial.agent.bind(tokenizer=tokenizer, decode=decode, config=config)
    result = await trial.run()
    trace = trial.agent.trace
    if result.exception_info is not None:
        error = result.exception_info
        # Harbor still verifies after an agent timeout. Preserve that outcome,
        # but do not silently treat setup or verifier failures as task failures.
        if error.exception_type == "AgentTimeoutError" and result.verifier_result is not None:
            trace.exit_status = "AgentTimeoutError"
            trace.truncated = True
        else:
            raise RuntimeError(
                f"Harbor trial {trial.config.trial_name}: {error.exception_type}: {error.exception_message}"
            )
    if result.verifier_result is None or result.verifier_result.rewards is None:
        raise RuntimeError(f"Harbor trial {trial.config.trial_name} returned no verifier reward")
    return trace, result.verifier_result.rewards, str(trial.paths.trial_dir)
