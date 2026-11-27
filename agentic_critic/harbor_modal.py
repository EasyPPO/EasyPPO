"""Modal task isolation and output handling for the pinned Harbor 0.24 adapter.

Task programs can emit arbitrary bytes. Harbor's default text-mode SDK reads
raise UnicodeDecodeError before returning the command's output and exit code.
Read bytes at that boundary and replace invalid UTF-8 only for display.
Reserve VM memory for Docker separately from the task's enforced memory limit.
Send long commands through stdin instead of Modal's bounded argument vector.
"""

from harbor.environments.base import ExecResult
from harbor.environments.modal import ModalEnvironment, _cached_env_secret

_DIND_MEMORY_OVERHEAD_MB = 2048
_COMMAND_ARG_BYTES = 60 * 1024
# Read to EOF before eval so child commands cannot consume the script itself.
# The sentinel preserves trailing newlines stripped by command substitution.
_STDIN_COMMAND = '_easyppo_command=$(cat; printf .); eval "${_easyppo_command%.}"'


class TerminalModalEnvironment(ModalEnvironment):
    def _memory_config(self):
        memory = super()._memory_config()
        if memory is None or not self._compose_mode:
            return memory
        # The Docker task retains its configured limit. Reserve separate RAM
        # for the VM/kernel/daemon so a task OOM cannot exhaust the outer VM.
        if isinstance(memory, tuple):
            return tuple(value + _DIND_MEMORY_OVERHEAD_MB for value in memory)
        return memory + _DIND_MEMORY_OVERHEAD_MB

    async def _sdk_exec(self, command, cwd=None, env=None, timeout_sec=None, shell="bash", login=False):
        # Keep Harbor's environment, timeout and unexpected-exit handling. This
        # override is tied to Harbor 0.24; no command is retried on decode errors.
        env = self._merge_env(env)
        if not self._sandbox:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        command_bytes = command.encode("utf-8")
        use_stdin = len(command_bytes) > _COMMAND_ARG_BYTES
        try:
            process = await self._sandbox.exec.aio(
                shell,
                "-lc" if login else "-c",
                _STDIN_COMMAND if use_stdin else command,
                workdir=cwd,
                secrets=[_cached_env_secret(env)] if env else [],
                timeout=timeout_sec,
                text=False,
            )
            if use_stdin:
                for offset in range(0, len(command_bytes), _COMMAND_ARG_BYTES):
                    process.stdin.write(command_bytes[offset : offset + _COMMAND_ARG_BYTES])
                    await process.stdin.drain.aio()
                process.stdin.write_eof()
                await process.stdin.drain.aio()
            stdout = await process.stdout.read.aio()
            stderr = await process.stderr.read.aio()
            return_code = await process.wait.aio()
        except Exception as exc:
            await self._raise_if_unexpected_exit_137(cause=exc)
            raise
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=return_code,
        )
