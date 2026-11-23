from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from agentic_critic.agentpick_verifier import AgentPickVerifier
from harbor.verifier.verifier import Verifier


@pytest.mark.parametrize(("content", "expected"), [(None, 0), ("", 0), ("invalid", 0), ("3.5", 1), ("-1", 0), ("0.75\nextra info", .75)])
def test_published_text_reward_has_precedence(tmp_path, content, expected):
    verifier = object.__new__(AgentPickVerifier)
    verifier.trial_paths = SimpleNamespace(reward_text_path=tmp_path / "reward.txt")
    if content is not None:
        verifier.trial_paths.reward_text_path.write_text(content)
    assert verifier._parse_reward_text() == verifier._parse_reward_json() == {"reward": expected}


@pytest.mark.asyncio
async def test_failed_pregrade_does_not_accept_old_output(monkeypatch):
    verifier = object.__new__(AgentPickVerifier)
    verifier.workdir = "/app"
    verifier.environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1)))
    parent = AsyncMock()
    monkeypatch.setattr(Verifier, "verify", parent)
    assert (await verifier.verify()).rewards == {"reward": 0.0}
    parent.assert_not_awaited()
