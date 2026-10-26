"""Check the terminal launcher reuses EasyPPO and retains thinking."""

import shlex
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.fixture
def train():
    return import_launch_script(REPO_ROOT / "examples/easyppo/run_qwen3_5_9b_terminal.py")


@pytest.mark.parametrize("critic_mode", ["shared", "full_history"])
def test_training_request(train, monkeypatch, tmp_path, critic_mode):
    captured = {}

    class Backend:
        def execute_train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train.ScriptArgs, "create_backend", lambda self: Backend())
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = train.ScriptArgs(
        prompt_data=str(tmp_path / "data with spaces.jsonl"),
        tasks_dir=str(tmp_path / "harbor tasks"),
        model_dir=str(tmp_path / "model dir"),
        data_dir=str(tmp_path / "data dir"),
        output_dir=str(tmp_path / "output dir"),
        critic_context_mode=critic_mode,
        eval_interval=5,
    )
    train.execute(args)
    argv = shlex.split(captured["train_args"])

    def value(flag):
        return argv[argv.index(flag) + 1]

    assert value("--custom-generate-function-path") == "agentic_critic.miles_terminal.generate"
    assert value("--prompt-data") == args.prompt_data
    assert value("--terminal-tasks-dir") == args.tasks_dir
    assert Path(value("--hf-checkpoint")) == Path(args.model_dir) / args.model_name
    assert value("--critic-context-mode") == critic_mode
    assert "--terminal-enable-thinking" in argv
    assert "--no-terminal-enable-thinking" not in argv
    assert Path(value("--chat-template-path")).is_file()
    assert value("--terminal-compact-threshold") == "12000"
    assert value("--terminal-max-new-tokens") == "8192"
    assert value("--rollout-top-p") == "1" and value("--rollout-top-k") == "-1"
    assert "--use-dynamic-global-batch-size" not in argv
    assert value("--eval-prompt-data") == "tb21"
    assert argv[argv.index("--eval-prompt-data") + 2] == args.tb21_data
    assert argv.count("--eval-prompt-data") == 1
    assert value("--eval-interval") == "5"
    assert value("--n-samples-per-eval-prompt") == "1"
    assert value("--eval-top-p") == "1" and value("--eval-top-k") == "-1"
    assert value("--eval-max-response-len") == "8192"
    assert "tmax_val" not in argv
    assert captured["extra_env_vars"]["PYTHONPATH"] == args.megatron_path


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"prompt_data": "data"},
        {"tasks_dir": "tasks"},
        {"prompt_data": "data", "tasks_dir": "tasks", "custom_rm_path": "other.reward"},
    ],
)
def test_invalid_launcher_inputs(train, overrides):
    with pytest.raises(ValueError):
        train._validate(train.ScriptArgs(**overrides))


def test_preparation_builds_validation_before_model(train, monkeypatch, tmp_path):
    calls = []

    class Backend:
        def exec_command_cpu(self, command):
            calls.append(shlex.split(command))

    monkeypatch.setattr(train.ScriptArgs, "create_backend", lambda self: Backend())
    monkeypatch.setattr(train._base, "prepare", lambda args: calls.append("model"))
    args = train.ScriptArgs(prompt_data="train.jsonl", tasks_dir=str(tmp_path / "tasks with spaces"))
    train.prepare(args)
    assert calls == [
        [
            "python",
            str(REPO_ROOT / "scripts/make_miles_terminal_data.py"),
            "--dataset",
            "terminal-bench-2.1",
            "--out",
            args.data_dir,
            "--tasks-dir",
            args.tasks_dir,
        ],
        "model",
    ]
