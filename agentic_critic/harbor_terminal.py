"""Harbor lifecycle and trajectory persistence for EasyPPO agents.

Requires the optional terminal dependencies. A fresh adapter is bound to one
in-process episode; no global registry or external agent server is needed.
"""

import asyncio
import logging
import os
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.trial.result import ExceptionInfo
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from agentic_critic.fcs_env import DEFAULT_MAX_SUBMISSIONS, DEFAULT_MAX_TURNS
from agentic_critic.task_integrity import TaskIntegrityCheckError, protected_digests
from agentic_critic.terminal_env import TerminalAgent, TerminalTrace


class SandboxInfrastructureFailure(RuntimeError):
    def __init__(self, failure_type, message, *, retry_start=False):
        super().__init__(message)
        self.failure_type = failure_type
        self.retry_start = retry_start


_MODEL_ERROR_NOTE = "Miles model decode callback failed (outside sandbox isolation)"


def _modal_trial_failure(error, location, *, starting):
    kind = error.exception_type
    message = error.exception_message
    trace = getattr(error, "exception_traceback", "")
    # Global failures must reach the owner even if Harbor serialized or wrapped them.
    if _MODEL_ERROR_NOTE in trace or kind in {
        "AuthError", "PermissionDeniedError", "MissingExtraError", "ModuleNotFoundError",
        "VersionError", "CancelledError", "KeyboardInterrupt", "SystemExit",
    } or any(f"modal.exception.{name}:" in trace for name in ("AuthError", "PermissionDeniedError")):
        return None
    # The allowlist controls only whether a fresh sandbox is worth trying.
    transient = kind == "EnvironmentStartTimeoutError" or (
        kind in {"ConnectionError", "InternalError", "InternalFailure", "ResourceExhaustedError",
                 "ServiceError", "SandboxTerminatedError", "SandboxTimeoutError", "TimeoutError",
                 "MountUploadTimeoutError", "VolumeUploadTimeoutError"}
        and f"modal.exception.{kind}:" in trace
    )
    if kind == "ImageBuildError":
        transient = any(text in message.lower() for text in (
            "temporary failure resolving", "connection timed out", "connection reset",
            "service unavailable", "too many requests", "gateway timeout",
        ))
    return SandboxInfrastructureFailure(kind, f"{location}: {message}", retry_start=starting and transient)


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
    if environment == "modal":
        environment_kwargs["modal_vm_runtime"] = True
        if "EASYPPO_TERMINAL_MODAL_APP" in os.environ:
            environment_kwargs["app_name"] = os.environ["EASYPPO_TERMINAL_MODAL_APP"]
    trial_config = TrialConfig(
        task=TaskConfig(path=task_path),
        trials_dir=trials_dir,
        agent=agent,
        verifier=VerifierConfig(
            import_path="agentic_critic.agentpick_verifier:AgentPickVerifier",
            kwargs={"workdir": metadata["agentpick_workdir"]},
        ) if "agentpick_workdir" in metadata else VerifierConfig(),
        # Concurrent Modal VM provisioning needs headroom beyond local builds.
        # Keep the agent and verifier execution budgets at the task defaults.
        environment_build_timeout_multiplier=6.0 if environment == "modal" else None,
        environment=EnvironmentConfig(
            type=environment if environment != "modal" else None,
            import_path="agentic_critic.harbor_modal:TerminalModalEnvironment" if environment == "modal" else None,
            # Keep the optional Modal backend's validated DinD transport.
            env={"COMPOSE_BAKE": "false"} if environment == "modal" else {},
            override_memory_mb=memory_mb,
            kwargs=environment_kwargs,
            extra_docker_compose=[Path(path) for path in docker_compose],
        ),
    )
    try:
        trial = await Trial.create(trial_config)
    except Exception as exc:
        if environment == "modal":
            failure = _modal_trial_failure(ExceptionInfo.from_exception(exc), task_path, starting=False)
            if failure is not None:
                logging.getLogger(__name__).exception("Could not initialize Modal trial for %s", task_path)
                raise failure from exc
        raise

    async def model_decode(*args, **kwargs):
        try:
            return await decode(*args, **kwargs)
        except Exception as exc:
            exc.add_note(_MODEL_ERROR_NOTE)
            raise

    trial.agent.bind(tokenizer=tokenizer, decode=model_decode if environment == "modal" else decode, config=config)
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
                raise TaskIntegrityCheckError("Missing pre-agent integrity baseline")
            if await protected_digests(trial.agent_environment, metadata) != baseline:
                trial.config.verifier.disable = True
                trial.result.verifier_result = VerifierResult(rewards={"reward": 0.0})

        trial.add_hook(TrialEvent.AGENT_START, capture_baseline)
        trial.add_hook(TrialEvent.AGENT_END, check_baseline)
    try:
        result = await trial.run()
    except Exception as exc:
        if environment == "modal":
            failure = _modal_trial_failure(ExceptionInfo.from_exception(exc), trial.paths.trial_dir, starting=False)
            if failure is not None:
                logging.getLogger(__name__).exception("Modal trial failed at %s", trial.paths.trial_dir)
                raise failure from exc
        raise
    trace = trial.agent.trace
    if result.exception_info is not None:
        error = result.exception_info
        if error.exception_type == "AgentTimeoutError" and (terminus or result.verifier_result is not None):
            trace.exit_status = "AgentTimeoutError"
            trace.truncated = True
        else:
            if environment == "modal":
                starting = result.environment_setup is not None and result.agent_setup is None and result.agent_execution is None
                failure = _modal_trial_failure(error, trial.paths.trial_dir, starting=starting)
                if failure is not None:
                    raise failure
            raise RuntimeError(
                f"Harbor trial {trial.config.trial_name}: {error.exception_type}: {error.exception_message}"
            )
    if terminus and trace.exit_status != "Submitted":
        return trace, {}, str(trial.paths.trial_dir)
    if result.verifier_result is None or result.verifier_result.rewards is None:
        message = f"Harbor trial {trial.paths.trial_dir} returned no verifier reward"
        if environment == "modal":
            raise SandboxInfrastructureFailure("MissingVerifierReward", message)
        raise RuntimeError(message)
    return trace, result.verifier_result.rewards, str(trial.paths.trial_dir)
