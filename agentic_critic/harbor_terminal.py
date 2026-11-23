"""Harbor lifecycle and trajectory persistence for EasyPPO agents.

Requires the optional terminal dependencies. A fresh adapter is bound to one
in-process episode; no global registry or external agent server is needed.
"""

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from agentic_critic.fcs_env import DEFAULT_MAX_SUBMISSIONS, DEFAULT_MAX_TURNS
from agentic_critic.task_integrity import protected_digests
from agentic_critic.terminal_env import TerminalAgent, TerminalTrace


class SandboxInfrastructureFailure(RuntimeError):
    def __init__(self, failure_type, message, *, retry_start=False):
        super().__init__(message)
        self.failure_type = failure_type
        self.retry_start = retry_start


def _modal_infrastructure_failure(result, trial_dir):
    error = result.exception_info
    kind, message = error.exception_type, error.exception_message
    starting = result.environment_setup is not None and result.agent_setup is None and result.agent_execution is None
    setup = result.agent_setup is not None and result.agent_execution is None
    # Harbor serializes exceptions; distinguish the SDK's errors from application errors of the same name.
    provider_error = kind in {
        "ConnectionError", "InternalError", "InternalFailure", "ResourceExhaustedError",
        "ServiceError", "SandboxTerminatedError", "SandboxTimeoutError", "TimeoutError",
        "MountUploadTimeoutError", "VolumeUploadTimeoutError",
    } and f"modal.exception.{kind}:" in getattr(error, "exception_traceback", "")
    retry_start = False
    if provider_error or (starting and kind == "EnvironmentStartTimeoutError"):
        retry_start = starting
    elif starting and kind == "ImageBuildError":
        retry_start = any(text in message.lower() for text in (
            "temporary failure resolving", "connection timed out", "connection reset",
            "service unavailable", "too many requests", "gateway timeout",
        ))
    elif not (
        kind in {"SandboxLikelyOutOfMemoryError", "VerifierTimeoutError", "DownloadVerifierDirError"}
        or (setup and kind == "AgentSetupTimeoutError")
        or (setup and kind == "RuntimeError" and message.startswith("Failed to start tmux session"))
        or (starting and kind == "RuntimeError" and message.startswith((
            "Docker daemon not ready after ", "Main container not running after ",
            "docker compose build failed:", "docker compose up failed:",
        )))
    ):
        return None
    return SandboxInfrastructureFailure(kind, f"{trial_dir}: {message}", retry_start=retry_start)


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
    task_metadata: dict | None = None,
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
    terminus = agent_protocol in ("terminus-2", "terminus-2-etl")
    if terminus:
        agent = AgentConfig(
            import_path="agentic_critic.harbor_terminus:MilesTerminus2",
            model_name=getattr(tokenizer, "name_or_path", "Qwen/Qwen3.5-9B"),
            kwargs=dict(
                parser_name="xml", max_turns=config.max_steps, suppress_max_turns_warning=True,
                record_terminal_session=False, enable_summarize=False,
            ),
        )
    else:
        agent = AgentConfig(import_path=agents[agent_protocol])
    metadata = task_metadata or {}
    environment_kwargs = {}
    if environment == "modal" and metadata.get("modal_task_entrypoint"):
        keepalive = [metadata["modal_task_entrypoint"], "sleep", "infinity"]
        if "agentpick_workdir" in metadata:
            keepalive = ["sh", "-c", 'cd "$1" 2>/dev/null || cd /; shift; exec "$@"',
                         "entrypoint", metadata["agentpick_workdir"], *keepalive]
        environment_kwargs["keepalive"] = keepalive
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_path),
            trials_dir=trials_dir,
            agent=agent,
            verifier=VerifierConfig(
                import_path="agentic_critic.agentpick_verifier:AgentPickVerifier",
                kwargs={"workdir": metadata["agentpick_workdir"]},
            ) if "agentpick_workdir" in metadata else VerifierConfig(),
            environment=EnvironmentConfig(
                type=environment,
                override_memory_mb=memory_mb,
                kwargs=environment_kwargs,
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
    if terminus:
        async def grade_submissions_only(_event):
            trial.config.verifier.disable = trial.agent.trace.exit_status != "Submitted"

        trial.add_hook(TrialEvent.AGENT_END, grade_submissions_only)
    if metadata.get("protected_paths") or metadata.get("protected_cmds"):
        baseline = None

        async def capture_baseline(_event):
            nonlocal baseline
            baseline = await protected_digests(trial.agent_environment, metadata)

        async def check_baseline(_event):
            if trial.config.verifier.disable:
                return
            if baseline is None:
                raise RuntimeError("Missing pre-agent integrity baseline")
            if await protected_digests(trial.agent_environment, metadata) != baseline:
                trial.config.verifier.disable = True
                trial.result.verifier_result = VerifierResult(rewards={"reward": 0.0})

        trial.add_hook(TrialEvent.AGENT_START, capture_baseline)
        trial.add_hook(TrialEvent.AGENT_END, check_baseline)
    result = await trial.run()
    trace = trial.agent.trace
    if result.exception_info is not None:
        error = result.exception_info
        if environment == "modal":
            failure = _modal_infrastructure_failure(result, trial.paths.trial_dir)
            if failure is not None:
                raise failure
        # Harbor still verifies after an agent timeout. Preserve that outcome,
        # but do not silently treat setup or verifier failures as task failures.
        if error.exception_type == "AgentTimeoutError" and (terminus or result.verifier_result is not None):
            trace.exit_status = "AgentTimeoutError"
            trace.truncated = True
        else:
            raise RuntimeError(
                f"Harbor trial {trial.config.trial_name}: {error.exception_type}: {error.exception_message}"
            )
    if terminus and trace.exit_status != "Submitted":
        return trace, {}, str(trial.paths.trial_dir)
    if result.verifier_result is None or result.verifier_result.rewards is None:
        raise RuntimeError(f"Harbor trial {trial.config.trial_name} returned no verifier reward")
    return trace, result.verifier_result.rewards, str(trial.paths.trial_dir)
