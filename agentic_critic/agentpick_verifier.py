"""Harbor verifier for the published AgentPick reward.txt contract."""

import shlex
import uuid

from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import RewardFileNotFoundError, Verifier


class AgentPickVerifier(Verifier):
    def __init__(self, *, workdir, **kwargs):
        super().__init__(**kwargs)
        self.workdir = workdir

    async def verify(self):
        nonce = f"tmax-sentinel-{uuid.uuid4().hex}"
        command = (
            f"mkdir -p {shlex.quote(self.workdir)} /logs/verifier && cd /logs/verifier || exit 1\n"
            "chattr -i . reward.txt ctrf.json 2>/dev/null || true\n"
            "rm -f reward.txt ctrf.json && "
            f"printf %s {nonce} > reward.txt && test \"$(cat reward.txt)\" = {nonce}"
        )
        prepared = await self.environment.exec(command=command, user="root", timeout_sec=60)
        if prepared.return_code:
            return VerifierResult(rewards={"reward": 0.0})
        try:
            return await super().verify()
        except RewardFileNotFoundError:
            return VerifierResult(rewards={"reward": 0.0})

    def _parse_reward_text(self):
        try:
            with self.trial_paths.reward_text_path.open() as stream:
                reward = float(stream.readline().strip())
        except (FileNotFoundError, ValueError):
            reward = 0.0
        return {"reward": max(0.0, min(1.0, reward))}

    def _parse_reward_json(self):
        # The published grader reads only reward.txt, even when other outputs coexist.
        return self._parse_reward_text()
