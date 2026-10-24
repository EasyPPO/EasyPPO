"""Exercise the unchanged converter with the requested full 50-batch shape."""

import json
import shlex
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace

import torch

from agentic_critic_pipeline.data import training_quotas
from agentic_critic_pipeline.miles_episodes import export_episodes
from agentic_critic_pipeline.validation import prepare
from miles import critic_recipe
from scripts.agentic_critic import miles_train
from tests.agentic_critic.cpu_validation_harness import ROOT
from tests.agentic_critic.test_validation import _episode, _write


def test_128_problem_export_produces_exactly_50_complete_batches(monkeypatch, tmp_path):
    problems = [{"problem_id": f"frontiersmith:{i}"} for i in range(128)]
    quotas = training_quotas(problems, 1234)
    source = tmp_path / "train.jsonl"
    _write(
        source,
        (
            _episode(f"train:{problem}:{i}", split="train", problem=problem, lengths=(3,) * (1 + i % 3))
            for problem, count in quotas.items()
            for i in range(count)
        ),
    )
    export_episodes(source, tmp_path / "episodes.jsonl")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/agentic_critic/episodes_to_rollouts.py"),
            "--episodes",
            str(tmp_path / "episodes.jsonl"),
            "--out",
            str(tmp_path),
            "--steps",
            "50",
            "--problems-per-step",
            "32",
            "--episodes-per-problem",
            "16",
            "--heldout-problems",
            "0",
            "--seed",
            "1234",
            "--rounds",
            "--require-exhaustion",
        ],
        check=True,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["steps"]) == 50
    assert len(list(tmp_path.glob("rollout_*.pt"))) == 50
    seen = set()
    counts = Counter()
    for index in range(50):
        samples = torch.load(tmp_path / f"rollout_{index}.pt", weights_only=False)["samples"]
        episodes = {}
        segments = Counter()
        for sample in samples:
            identity = sample["metadata"]["sample"]
            assert identity not in seen
            episodes[identity] = sample["metadata"]["problem"]
            segments[identity] += 1
        assert len(episodes) == 512
        assert Counter(Counter(episodes.values()).values()) == {16: 32}
        assert all(n == 1 + int(identity.rsplit(":", 1)[1]) % 3 for identity, n in segments.items())
        seen.update(episodes)
        counts.update(episodes.values())
    assert len(seen) == 25600
    assert counts == quotas
    _check_validation_and_launch(monkeypatch, tmp_path, source, quotas)


def _check_validation_and_launch(monkeypatch, directory, source, quotas):
    validation = directory / "validation.jsonl"
    _write(
        validation,
        (_episode(f"validation:{problem}:{i}", problem=problem, lengths=(3,)) for problem in quotas for i in range(32)),
    )
    plan = prepare(validation, source, directory / "validation_replay")
    assert len(plan["episodes"]) == 4096
    assert len(plan["replay"]) == 128
    model = directory / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    prompts = directory / "prompts.jsonl"
    _write(prompts, [dict(prompt=problem) for problem in quotas])
    captured = []
    backend = SimpleNamespace(
        convert_checkpoint=lambda **kw: captured.append(("convert", kw)),
        execute_train=lambda **kw: captured.append(("train", kw)),
    )
    monkeypatch.setattr(critic_recipe.ScriptArgs, "create_backend", lambda args: backend)
    monkeypatch.setenv("MEGATRON_PATH", str(directory / "megatron"))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    for mode in ("full_history", "actor_summary"):
        output = directory / mode
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "miles_train.py",
                "--model",
                str(model),
                "--rollouts",
                str(directory),
                "--output",
                str(output),
                "--model-dir",
                str(directory / "models"),
                "--prompt-data",
                str(prompts),
                "--mode",
                mode,
                "--validation-plan",
                str(directory / "validation_replay/validation_plan.json"),
            ],
        )
        miles_train.main()  # Real CLI defaults and recipe; only external submission is replaced.
        assert [kind for kind, _ in captured[-2:]] == ["convert", "train"]
        argv = shlex.split(captured[-1][1]["train_args"])
        assert "--save-debug-train-data" not in argv
        for flag, expected in {
            "--global-batch-size": "512",
            "--rollout-batch-size": "32",
            "--n-samples-per-prompt": "16",
            "--num-rollout": "50",
            "--num-critic-only-steps": "50",
            "--critic-num-steps-per-rollout": "4",
            "--critic-validation-interval": "5",
            "--gamma": "1.0",
            "--lambd": "1.0",
            "--critic-context-mode": "full_history" if mode == "full_history" else "shared",
        }.items():
            assert argv.count(flag) == 1
            assert argv[argv.index(flag) + 1] == expected
        metadata = json.loads((output / "miles_run.json").read_text())
        assert metadata["validation_plan_hash"] == plan["plan_hash"]
        assert metadata["optimizer_updates"] == 200
        assert metadata["dump_dir"] is None
