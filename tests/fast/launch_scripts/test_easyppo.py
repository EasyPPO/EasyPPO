import pytest

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.fixture
def launcher(monkeypatch):
    freeze_environment(monkeypatch)
    return import_launch_script(REPO_ROOT / "examples/easyppo/run_qwen3_5_9b_frontier_cs.py")


def test_default_recipe_requires_a_task_and_reward(launcher):
    with pytest.raises(ValueError, match="prompt-data"):
        launcher._require_frontier_cs_inputs(launcher.ScriptArgs())


def test_agentic_recipe_requires_a_dataset(launcher):
    args = launcher.ScriptArgs(agentic_generate_function_path="agentic_critic.miles_fcs.generate")
    with pytest.raises(ValueError, match="prompt-data"):
        launcher._require_frontier_cs_inputs(args)


@pytest.mark.parametrize("mode", ["shared", "full_history"])
def test_agentic_recipe_routes_context_and_reward(launcher, monkeypatch, tmp_path, mode):
    recording = install_command_recorder(monkeypatch)
    call_entrypoint(
        launcher,
        "execute",
        {
            "prompt_data": str(tmp_path / "problems.jsonl"),
            "agentic_generate_function_path": "agentic_critic.miles_fcs.generate",
            "critic_context_mode": mode,
            "num_gpus_per_node": 1,
            "tensor_model_parallel_size": 1,
        },
        sandbox=tmp_path,
    )
    command = " ".join(recording.commands)
    assert "--custom-generate-function-path agentic_critic.miles_fcs.generate" in command
    assert f"--critic-context-mode {mode}" in command
    assert "--critic-num-steps-per-rollout 4" in command
    assert "--filter-truncated-in-actor-loss" in command
    assert "--apply-chat-template" not in command
    assert "--custom-rm-path" not in command
    assert "--use-dynamic-global-batch-size" not in command
