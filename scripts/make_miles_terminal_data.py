"""Prepare full TMAX training data and official Terminal-Bench 2.1 validation.

Run from a Miles environment with the terminal extra installed:
    python scripts/make_miles_terminal_data.py --tasks-dir /data/harbor --out /data/prepared

Use --dataset terminal-bench-2.1 to prepare validation only. Missing task datasets
are downloaded at fixed versions; the complete task root is needed by workers.
"""

import argparse
import errno
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from copy import deepcopy
from pathlib import Path

import tomlkit

import agentic_critic

_DATASETS = {
    "tmax": (
        "TMax-15K-Harbor",
        "tmax/TMax-15K-Harbor@sha256:167d0c6c7e34db1efb81ea6cfed9e26cf20a78fa4d0b1bce29778aafae760dbc",
    ),
    "terminal-bench-2.1": (
        "terminal-bench-2-1",
        "terminal-bench/terminal-bench-2-1@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a",
    ),
}


# csk4gby2 data release, torchtitan 9403e73.
TERMINAL_RUNTIME_LAYER = """
# harbor-agent-runtime: tmux is required by the terminal agent
RUN ( if command -v tmux >/dev/null 2>&1; then \\
        exit 0; \\
      elif command -v apt-get >/dev/null 2>&1; then \\
        apt-get update && apt-get install -y tmux \\
        && rm -rf /var/lib/apt/lists/*; \\
      elif command -v yum >/dev/null 2>&1; then \\
        yum install -y tmux && yum clean all; \\
      elif command -v dnf >/dev/null 2>&1; then \\
        dnf install -y tmux && dnf clean all; \\
      elif command -v microdnf >/dev/null 2>&1; then \\
        microdnf install -y tmux && microdnf clean all; \\
      elif command -v apk >/dev/null 2>&1; then \\
        apk add --no-cache tmux; \\
      elif command -v pacman >/dev/null 2>&1; then \\
        pacman -Sy --noconfirm tmux; \\
      elif command -v zypper >/dev/null 2>&1; then \\
        zypper install -y tmux; \\
      else \\
        echo 'ERROR: no supported package manager to install tmux' >&2; exit 1; \\
      fi ) \\
    && tmux -V
"""


def task_records(
    tasks_dir: Path, *, dataset: str, split: str, limit: int | None = None, path_root: Path | None = None
) -> list[dict]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    root = tasks_dir.resolve(strict=True)
    path_root = path_root.resolve(strict=True) if path_root is not None else root
    configs = sorted(root.rglob("task.toml"))
    if not configs:
        raise ValueError(f"No Harbor task.toml files under {root}")
    records = []
    for config in configs[:limit]:
        task = config.parent
        instruction = task / "instruction.md"
        if not instruction.is_file() or not instruction.read_text().strip():
            raise ValueError(f"Missing or empty instruction.md in {task}")
        if not (task / "environment").is_dir() or not (task / "tests").is_dir():
            raise ValueError(f"Expected Harbor environment/ and tests/ in {task}")
        path = task.relative_to(path_root).as_posix()
        records.append(
            {
                "prompt": instruction.read_text(),
                "metadata": {
                    "task_path": path,
                    "task_id": f"{dataset}/{path}",
                    "dataset": dataset,
                    "split": split,
                    "agent_name": "mini-swe-agent",
                },
            }
        )
    return records


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prompt_key(text):
    return digest(" ".join(unicodedata.normalize("NFKC", text).split()).encode())


def audit_task(row, tokenizer, task_root):
    # These optional dependencies are only needed for full training-data checks.
    from harbor.models.task.task import Task
    from agentic_critic.terminal_env import _SYSTEM

    path = task_root / row["metadata"]["task_path"]

    task = Task(path)
    assert row["prompt"] == (path / "instruction.md").read_text()
    assert path.resolve().is_relative_to(task_root.resolve())
    assert (path / "tests/test.sh").is_file()
    env = path / "environment"
    image = task.config.environment.docker_image
    assert image or any(
        (env / name).is_file()
        for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": task.instruction}]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=True
    )
    assert tokens and all(type(t) is int for t in tokens)
    file_hashes = {}
    for file in sorted(path.rglob("*")):
        if file.is_file():
            if file.is_symlink():
                assert file.resolve().is_relative_to(task_root.resolve()), f"External symlink: {file}"
            file_hashes[file.relative_to(path).as_posix()] = digest(file.read_bytes())
    # Group byte-identical executable fixtures; task names/metadata do not identify fixtures.
    core = {k: v for k, v in file_hashes.items() if k == "instruction.md" or k.startswith(("environment/", "tests/"))}
    return {
        "task_id": row["metadata"]["task_id"],
        "task_path": row["metadata"]["task_path"],
        "prompt_group": prompt_key(task.instruction),
        "fixture_hash": digest(json.dumps(core, sort_keys=True).encode()),
        "fits_initial_action": len(tokens) + 8192 <= 32768,
    }


def training_records(rows, audits, eval_audits):
    eval_groups = {a["prompt_group"] for a in eval_audits}
    eval_fixtures = {a["fixture_hash"] for a in eval_audits}
    seen = set()
    training = []
    excluded = []
    for row, audit in zip(rows, audits, strict=True):
        reason = None
        if audit["prompt_group"] in eval_groups or audit["fixture_hash"] in eval_fixtures:
            reason = "benchmark_overlap"
        elif not audit["fits_initial_action"]:
            reason = "initial_context_exceeds_action_budget"
        elif audit["fixture_hash"] in seen:
            reason = "identical_instruction_environment_tests"
        if reason:
            excluded.append({"task_id": audit["task_id"], "reason": reason})
            continue
        seen.add(audit["fixture_hash"])
        record = deepcopy(row)
        record["metadata"]["split"] = "train"
        training.append(record)
    training.sort(key=lambda row: row["metadata"]["task_id"])
    if not training:
        raise ValueError("No eligible TMAX training tasks")
    return training, excluded


def _download(dataset: str, tasks_dir: Path) -> Path:
    directory, reference = _DATASETS[dataset]
    source = tasks_dir / directory
    if not source.is_dir():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "harbor.cli.main",
                "datasets",
                "download",
                reference,
                "--output-dir",
                str(tasks_dir),
            ],
            check=True,
        )
    return source


def prepare_tb21(tasks_dir: Path, out: Path) -> list[dict]:
    source = _download("terminal-bench-2.1", tasks_dir)
    rows = task_records(source, dataset="terminal-bench-2.1", split="test", path_root=tasks_dir)
    if len(rows) != 89:
        raise ValueError(f"TB2.1 validation requires all 89 tasks; found {len(rows)} under {source}")
    # Keep the registry download intact. Harbor normally prefers docker_image to the local Dockerfile.
    prepared = tasks_dir / "terminal-bench-2-1-tmux"
    if not prepared.exists():
        with tempfile.TemporaryDirectory(dir=tasks_dir) as staging:
            staged = Path(staging) / prepared.name
            shutil.copytree(source, staged)
            for row in rows:
                relative = (tasks_dir / row["metadata"]["task_path"]).relative_to(source)
                task = staged / relative
                config_path = task / "task.toml"
                config = tomlkit.parse(config_path.read_text())
                dockerfile = task / "environment/Dockerfile"
                image = config.get("environment", {}).pop("docker_image", None)
                if image:
                    recipe = f"FROM {image}"
                    config_path.write_text(tomlkit.dumps(config))
                else:
                    recipe = dockerfile.read_text().rstrip()
                dockerfile.write_text(recipe + "\n" + TERMINAL_RUNTIME_LAYER)
            try:
                staged.rename(prepared)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not prepared.is_dir():
                    raise

    for row in rows:
        relative = (tasks_dir / row["metadata"]["task_path"]).relative_to(source)
        row["metadata"]["task_path"] = (Path(prepared.name) / relative).as_posix()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as staging:
        manifest = Path(staging) / "tb21.eval.jsonl"
        write_jsonl(manifest, rows)
        manifest.replace(out / manifest.name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("tmax", "terminal-bench-2.1"), default="tmax")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()
    evaluation = prepare_tb21(args.tasks_dir, args.out)
    print(f"Prepared {len(evaluation)} TB2.1 validation tasks", flush=True)
    if args.dataset == "terminal-bench-2.1":
        return

    # Validation-only preparation needs neither a model tokenizer nor the agent loop.
    from transformers import AutoTokenizer

    source = _download("tmax", args.tasks_dir)
    rows = task_records(source, dataset="tmax", split="train", path_root=args.tasks_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.chat_template = Path(agentic_critic.__file__).with_name("qwen3_5_interleaved.jinja").read_text()
    eval_audits = [audit_task(row, tokenizer, args.tasks_dir) for row in evaluation]
    audits = []
    for index, row in enumerate(rows):
        audits.append(audit_task(row, tokenizer, args.tasks_dir))
        if (index + 1) % 1000 == 0:
            print(f"Checked {index + 1}/{len(rows)} TMAX tasks", flush=True)
    training, excluded = training_records(rows, audits, eval_audits)
    write_jsonl(args.out / "tmax.train.jsonl", training)
    # Remove the obsolete generated holdout when reusing an earlier output directory.
    (args.out / "tmax.val.jsonl").unlink(missing_ok=True)
    print(f"Prepared {len(training)} TMAX training tasks; excluded {len(excluded)}; no TMAX holdout")


if __name__ == "__main__":
    main()
