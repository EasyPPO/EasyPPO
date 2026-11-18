"""Harbor lifecycle and trajectory persistence for EasyPPO agents.

Requires the optional terminal dependencies. A fresh adapter is bound to one
in-process episode; no global registry or external agent server is needed.
"""

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from agentic_critic.fcs_env import DEFAULT_MAX_SUBMISSIONS, DEFAULT_MAX_TURNS
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

    def _make_agent(self, environment):
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

        return TerminalAgent(tokenizer=self.tokenizer, decode=self.decode, execute=execute, config=self.episode_config)

    async def run(self, instruction, environment, context) -> None:
        agent = self._make_agent(environment)
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
            self.trace.write_trajectory(agent.messages, self.logs_dir)


async def run_trial(
    *,
    task_path: Path,
    trials_dir: Path,
    environment: str,
    tokenizer,
    decode,
    config,
    memory_mb: int | None = None,
    docker_compose: tuple[str, ...] = (),
    judge_url: str | None = None,
    agent_protocol: str = "terminal",
    fcs_max_submissions: int = DEFAULT_MAX_SUBMISSIONS,
    fcs_max_turns: int = DEFAULT_MAX_TURNS,
):
    """Run one episode with the selected environment and return its verified reward."""
    if environment == "shared-judge":
        if memory_mb is not None or docker_compose:
            raise ValueError("shared-judge does not use --terminal-memory-mb or --terminal-docker-compose")
        if agent_protocol != "fcs-submit":
            raise ValueError("shared-judge requires the fcs-submit agent protocol")
        # Only direct FCS submissions need the Miles HTTP client.
        from agentic_critic.shared_fcs import run_shared_judge_episode

        return await run_shared_judge_episode(
            task_path=task_path,
            trials_dir=trials_dir,
            judge_url=judge_url,
            tokenizer=tokenizer,
            decode=decode,
            config=config,
            max_submissions=fcs_max_submissions,
            max_turns=fcs_max_turns,
        )

    agents = {
        "terminal": "agentic_critic.harbor_terminal:MiniSweTerminalAgent",
        "fcs-submit": "agentic_critic.harbor_fcs:HarborFcsAgent",
    }
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_path),
            trials_dir=trials_dir,
            agent=AgentConfig(import_path=agents[agent_protocol]),
            environment=EnvironmentConfig(
                type=environment,
                override_memory_mb=memory_mb,
                extra_docker_compose=[Path(path) for path in docker_compose],
            ),
        )
    )
    trial.agent.bind(tokenizer=tokenizer, decode=decode, config=config)
    if agent_protocol == "fcs-submit":
        trial.agent.bind_judge(
            verifier_env=trial.task.config.verifier.env,
            max_submissions=fcs_max_submissions,
            max_turns=fcs_max_turns,
        )
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
