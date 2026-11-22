"""Record AgentPick's protected files and command outputs inside the trial sandbox."""

import re
import shlex


async def protected_digests(environment, metadata):
    commands = []
    for path in metadata.get("protected_paths", []):
        quoted = shlex.quote(path)
        # The pinned AgentPick release protects regular files at absolute paths.
        commands.append(
            f"if [ -f {quoted} ]; then sha256sum < {quoted} | cut -d' ' -f1; "
            f"elif [ -e {quoted} ]; then echo OTHER; else echo ABSENT; fi"
        )
    for cmd in metadata.get("protected_cmds", []):
        commands.append(
            "if output=$(env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LC_ALL=C "
            f"bash -c {shlex.quote(cmd)} </dev/null 2>/dev/null); then "
            "printf '%s' \"$output\" | sha256sum | cut -d' ' -f1; else echo FAIL; fi"
        )
    result = await environment.exec(
        command="bash -c " + shlex.quote("set -eo pipefail\n" + "\n".join(commands)),
        user="root",
        timeout_sec=120,
    )
    digests = (result.stdout or "").splitlines()
    if result.return_code or len(digests) != len(commands) or not all(
        re.fullmatch(r"[0-9a-f]{64}|ABSENT|OTHER|FAIL", value) for value in digests
    ):
        raise RuntimeError("Could not measure AgentPick integrity baseline")
    return digests
