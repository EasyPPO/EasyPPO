"""Harbor 0.24 Modal execution compatibility for terminal tasks."""

import asyncio
import shlex
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.environments.base import ExecResult
from harbor.environments.modal import ModalEnvironment, _cached_env_secret

# Leave space for Harbor's compose/user wrappers below Modal's 64 KiB argv cap.
_INLINE_COMMAND_BYTES = 16 * 1024


async def _stage_command(command, upload):
    if len(command.encode("utf-8")) < _INLINE_COMMAND_BYTES:
        return command
    remote = f"/tmp/easyppo-exec-{uuid4().hex}.sh"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "command.sh"
        path.write_text(command, encoding="utf-8")
        path.chmod(0o644)
        await upload(path, remote)
    quoted = shlex.quote(remote)
    return f"trap 'rm -f {quoted}' EXIT; . {quoted}"


class TerminalModalEnvironment(ModalEnvironment):
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        # Stage inside the task container before DinD quotes the command into
        # docker compose exec. Its VM-level wrapper is staged separately below.
        command = await _stage_command(command, self.upload_file)
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def _sdk_exec(self, command, cwd=None, env=None, timeout_sec=None, shell="bash", login=False):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        env = self._merge_env(env)
        try:
            command = await _stage_command(command, self._sdk_upload_file)
            process = await self._sandbox.exec.aio(
                shell,
                "-lc" if login else "-c",
                command,
                workdir=cwd,
                secrets=[_cached_env_secret(env)] if env else [],
                timeout=timeout_sec,
                text=False,
            )
            # Read bytes so arbitrary command output cannot abort a trial on
            # invalid UTF-8. Keep both pipes draining while the process runs.
            readers = [
                asyncio.create_task(call)
                for call in (process.stdout.read.aio(), process.stderr.read.aio(), process.wait.aio())
            ]
            try:
                stdout, stderr, return_code = await asyncio.gather(*readers)
            finally:
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
        except Exception as exc:
            await self._raise_if_unexpected_exit_137(cause=exc)
            raise
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=return_code,
        )
