"""Preparation and execution helpers, imported inside the training image."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from miles.utils.file_utils import atomic_write_text


def prepare_data(config):
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.generation.hf_repo,
        revision=config.generation.hf_revision,
        local_dir=f"{config.models}/{config.generation.model_name}",
    )
    tasks = Path(config.data, "harbor")
    if not Path(config.prompt_data).exists():
        subprocess.run(
            ["python3", "scripts/make_miles_agentpick_data.py", "--published-release", "--tasks-dir", str(tasks)],
            check=True,
        )
    subprocess.run(
        [
            "python3",
            "scripts/make_miles_terminal_data.py",
            "--dataset",
            "terminal-bench-2.1",
            "--tasks-dir",
            str(tasks),
            "--out",
            config.data,
        ],
        check=True,
    )
    check_data(config)


def check_data(config):
    from harbor.models.task.task import Task, strip_canary

    tasks = []
    for relative, count in ((config.prompt_data, 802), ("tb21.eval.jsonl", 89)):
        rows = [json.loads(line) for line in Path(config.data, relative).read_text().splitlines()]
        if len(rows) != count:
            raise ValueError(f"Expected {count} tasks in {relative}, found {len(rows)}")
        tasks.extend(rows)

    def validate(row):
        path = Path(config.data, "harbor", row["metadata"]["task_path"])
        try:
            task = Task(path)
            if task.instruction != strip_canary(row["prompt"]):
                raise ValueError("Prepared prompt differs from task instruction")
        except Exception as exc:
            return f"{path}: {type(exc).__name__}: {exc}"
        return None

    # Harbor walks every input tree. Overlap independent metadata reads on v2
    # storage while retaining schema, path-safety and prompt checks for every task.
    with ThreadPoolExecutor(max_workers=16) as pool:
        failures = [error for error in pool.map(validate, tasks) if error is not None]
    if failures:
        raise ValueError("Invalid prepared tasks:\n" + "\n".join(failures))
    print("Validated all 802 training and 89 evaluation task schemas and prompts", flush=True)


def check_manifest(config):
    path = Path(config.output, "modal_run.json")
    expected = config.manifest()
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError("Existing run has different settings; use a new run ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(expected, indent=2) + "\n")


def train(config, _judge_url, _judge_token, *, run_command):
    # Named secrets are inherited by Ray; no credentials enter the printed argv.
    os.environ.update(WANDB_RUN_ID=config.run_id, WANDB_RESUME="allow", EASYPPO_TERMINAL_MODAL_APP=config.sandbox_app)
    check_manifest(config)
    run_command(config.command(), check=True)
    atomic_write_text(Path(config.output, "training_complete.json"), '{"completed": true}\n')


async def sandbox_smoke(config):
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    rows = [json.loads(line) for line in Path(config.data, "tb21.eval.jsonl").read_text().splitlines()]
    task = Path(config.data, "harbor", rows[0]["metadata"]["task_path"])
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task),
            trials_dir=Path(config.output, "sandbox-smoke"),
            agent=AgentConfig(
                import_path="agentic_critic.harbor_terminus:MilesTerminus2", model_name=config.generation.hf_repo
            ),
            environment=EnvironmentConfig(
                import_path="agentic_critic.harbor_modal:TerminalModalEnvironment",
                env={"COMPOSE_BAKE": "false"},
                kwargs={"modal_vm_runtime": True, "app_name": config.sandbox_app},
            ),
        )
    )
    environment = trial.agent_environment
    try:
        await environment.start(force_build=False)
        result = await environment.exec(
            command="tmux -V && tmux new-session -d -s easyppo-smoke 'sleep 60' "
            "&& tmux has-session -t easyppo-smoke && tmux kill-session -t easyppo-smoke "
            "&& printf EASYPPO_SANDBOX_OK",
            timeout_sec=30,
        )
        if result.return_code != 0 or "EASYPPO_SANDBOX_OK" not in result.stdout:
            raise RuntimeError("Terminal Bench sandbox command failed")
        print("Real TB2.1 sandbox creation, execution and cleanup passed", flush=True)
    finally:
        await environment.stop(delete=True)
        trial._close_logger_handler()
