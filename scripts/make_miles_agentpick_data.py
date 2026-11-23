"""Prepare AgentPick Harbor tasks from the historical subset or the published runbook release."""

import argparse
import base64
import csv
import hashlib
import json
import random
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import tomlkit
from huggingface_hub import hf_hub_download

from make_miles_terminal_data import TERMINAL_RUNTIME_LAYER, task_records, write_jsonl

_DATASET = "Fzz1/tb-training-mix-agentpick"
_REVISION = "6e3e6c8a01d84f06b9f18f61782170f4801d5330"
_ARCHIVES = {
    "clean": "4f705d011b5d8e47174d9176a951f3b496a6d3b6aa45c352acccd64e8d3d9298",
    "calibforge_clean": "57d806342919d0a8444392edc79c88657210048f310cccff81e3ac76e7bc40c8",
}

_RELEASE_REPO = "andylizf/TerminalWorld-Seeds-Clean"
_RELEASE_REVISION = "05ae8e94386dacbda45d94c876c4510e1b34fe3b"
_RELEASE_ID = "3c67194ede7c8a88dee4e21b42ad39c9b5365bcf70868c60f599e32d717c8876"
_RELEASE_TAR_SHA256 = "1b432b47d59fca9625a2497c93edfcd7723dae467b5e7eab09613463ee2e9cdd"


def selected_tasks():
    """List the historical training population from upstream split metadata."""
    path = hf_hub_download(_DATASET, "metadata/split_assignment.tsv", repo_type="dataset", revision=_REVISION)
    with open(path, encoding="utf-8", newline="") as stream:
        rows = [
            {key: row[key] for key in ("task_id", "corpus", "split")}
            for row in csv.DictReader(stream, delimiter="\t")
            if row["split"] in _ARCHIVES and row["corpus"] != "SMITH" and row["task_id"] != "tw_14495"
        ]
    if len(rows) != 543 or len({row["task_id"] for row in rows}) != 543:
        raise ValueError("Expected 543 unique tasks in the pinned AgentPick selection")
    return rows


def list_tasks(seed):
    rows = selected_tasks()
    random.Random(seed).shuffle(rows)
    for row in rows:
        print(json.dumps(row))


def _extract(root, split, ids, archive_dir):
    filename = f"tasks-{split}.tar"
    archive = (
        archive_dir / filename
        if archive_dir
        else Path(hf_hub_download(_DATASET, f"data/{filename}", repo_type="dataset", revision=_REVISION))
    )
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != _ARCHIVES[split]:
            raise ValueError(f"Unexpected archive digest for {filename}")
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = Path(member.name).parts
            if len(parts) >= 3 and parts[0] == "tasks" and parts[1] in ids:
                tar.extract(member, root, filter="data")


def _configure_task(task, resources):
    # Harbor reads this staged file at rollout time; task_records also uses it for the JSONL prompt.
    instruction = task / "instruction.md"
    raw_instruction = instruction.read_text()
    prompt, removed = re.subn(r"^.*harbor-canary.*$\n?", "", raw_instruction, flags=re.MULTILINE)
    if removed:
        instruction.write_text(prompt)

    dockerfile = task / "environment/Dockerfile"
    image_recipe = dockerfile.read_text()
    if task.name == "tw_12315":
        # Buster's pip ignores Modal's HTTP index. This version supports the task's Python 3.7.
        image_recipe = image_recipe.replace(
            "RUN pip install --no-cache-dir msgpack\n",
            "RUN pip install --index-url https://pypi.org/simple --no-cache-dir msgpack==1.0.5\n",
        )
    dockerfile.write_text(image_recipe.rstrip("\n") + "\n" + TERMINAL_RUNTIME_LAYER)

    _configure_harbor(task, resources)


def _configure_harbor(task, resources, *, run_as_root=False):
    config = task / "task.toml"
    document = tomlkit.parse(config.read_text() if config.exists() else "")
    # SWE-Rebench packages publish resource fields at the TOML root.
    if "cpus" in document and "environment" not in document:
        document["environment"] = tomlkit.table()
        for key in ("cpus", "memory", "storage", "docker_image"):
            if key in document:
                document["environment"][key] = document.pop(key)
    if "environment" not in document:
        document["environment"] = tomlkit.table()
    env = document["environment"]
    # Harbor must build the derived image even when the upstream task names a prebuilt image.
    env.pop("docker_image", None)

    # The pinned split publishes the final allocations. Harbor rejects conflicting legacy units.
    env.pop("memory", None)
    env.pop("storage", None)
    env["cpus"] = int(resources["req_cpus"])
    env["memory_mb"] = int(resources["req_memory_mb"])
    env["storage_mb"] = int(resources["est_disk_mb"])
    # csk4gby2 disabled per-task agent budgets at release build time. Neither
    # pinned parquet has validated_test_timeout_s, so the verifier uses the run default too.
    for phase, seconds in (("agent", 2400), ("verifier", 600)):
        if phase not in document:
            document[phase] = tomlkit.table()
        document[phase]["timeout_sec"] = seconds
        if run_as_root:
            document[phase]["user"] = "root"
    # Modal needs its outbound filter enabled when the sandbox is created to change it between phases.
    if env.get("allow_internet") is False or env.get("network_mode") == "no-network":
        env.pop("allow_internet", None)
        env["network_mode"] = "public"
        document["agent"]["network_mode"] = "no-network"
    document["verifier"]["network_mode"] = "public"
    config.write_text(tomlkit.dumps(document))


def prepare(tasks_dir, archive_dir):
    selected = selected_tasks()
    tasks_dir.mkdir(parents=True, exist_ok=True)
    dest = tasks_dir / "agentpick-csk4gby2"
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    with tempfile.TemporaryDirectory(dir=tasks_dir) as staging:
        root = Path(staging) / dest.name
        resources = {}
        for split in _ARCHIVES:
            path = hf_hub_download(_DATASET, f"data/{split}.parquet", repo_type="dataset", revision=_REVISION)
            table = pq.read_table(path, columns=["task_id", "req_cpus", "req_memory_mb", "est_disk_mb", "protected_paths", "protected_cmds"])
            resources.update({(split, row["task_id"]): row for row in table.to_pylist()})
            _extract(root, split, {row["task_id"] for row in selected if row["split"] == split}, archive_dir)
        records = []
        for index, row in enumerate(selected):
            task = root / "tasks" / row["task_id"]
            settings = resources[row["split"], task.name]
            _configure_task(task, settings)
            (record,) = task_records(task, dataset=_DATASET, split="train", path_root=Path(staging))
            record["metadata"].update(
                source_index=index,
                source_split=row["split"],
                corpus=row["corpus"],
                source_task_id=row["task_id"],
                source_revision=_REVISION,
            )
            for key in ("protected_paths", "protected_cmds"):
                entries = json.loads(settings[key] or "[]")
                if entries:
                    record["metadata"][key] = entries
            # Both service tasks in the pinned archives exec their argument after setup.
            dockerfile = (task / "environment/Dockerfile").read_text()
            if 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile.splitlines():
                record["metadata"]["modal_task_entrypoint"] = "/entrypoint.sh"
            records.append(record)
        write_jsonl(root / "train.jsonl", records)
        root.rename(dest)
    print(f"Prepared {len(records)} tasks in {dest}")


def prepare_release(tasks_dir, archive_dir):
    archive = archive_dir / "release.tar" if archive_dir else Path(hf_hub_download(
        _RELEASE_REPO, f"releases/{_RELEASE_ID}/release.tar", revision=_RELEASE_REVISION, repo_type="dataset"
    ))
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != _RELEASE_TAR_SHA256:
            raise ValueError("Unexpected runbook release archive digest")
    dest = tasks_dir / "agentpick-802"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    with tempfile.TemporaryDirectory(dir=tasks_dir) as staging:
        root = Path(staging) / dest.name
        release = root / "release"
        with tarfile.open(archive) as tar:
            tar.extractall(release, filter="data")
        manifest = json.loads((release / "manifest.json").read_text())
        if manifest["release_sha256"] != _RELEASE_ID:
            raise ValueError("Unexpected AgentPick release")
        records = []
        for index, line in enumerate((release / "mix.jsonl").read_text().splitlines()):
            row = json.loads(line)
            metadata = row["metadata"]
            task = root / "tasks" / row["label"]
            source = release / "sources" / metadata["corpus"] / "tasks" / row["label"]
            shutil.copytree(source, task, ignore=lambda directory, _: {"environment"} if Path(directory) == source else set())
            environment = task / "environment"
            environment.mkdir()
            for relative_path, encoded in metadata.get("build_context", {}).items():
                target = environment / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(encoded, validate=True))
            (task / "environment/Dockerfile").write_text(metadata["dockerfile"])
            (task / "instruction.md").write_text(row["prompt"])
            _configure_harbor(task, metadata["resource_inputs"], run_as_root=True)
            (record,) = task_records(task, dataset=_RELEASE_REPO, split="train", path_root=Path(staging))
            record["metadata"].update(
                source_index=index,
                source_task_id=row["label"],
                corpus=metadata["corpus"],
                source_revision=_RELEASE_REVISION,
                agentpick_workdir=metadata["workdir"],
            )
            for key in ("protected_paths", "protected_cmds"):
                if entries := metadata["tmax"].get(key):
                    record["metadata"][key] = entries
            if metadata.get("entrypoint"):
                record["metadata"]["modal_task_entrypoint"] = metadata["entrypoint"].split()[0]
            records.append(record)
        if len(records) != manifest["rows"]:
            raise ValueError("Published mix length differs from its manifest")
        write_jsonl(root / "train.jsonl", records)
        root.rename(dest)
    print(f"Prepared {len(records)} tasks in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tasks-dir", type=Path)
    mode.add_argument("--list-tasks", action="store_true", help="Print the 543 historical task metadata rows as JSONL.")
    parser.add_argument("--published-release", action="store_true", help="Prepare the complete 802-task runbook release.")
    parser.add_argument("--seed", type=int, help="Shuffle seed for --list-tasks.")
    parser.add_argument(
        "--archive-dir", type=Path, help="Read the selected release tar or the two historical split archives from this directory."
    )
    args = parser.parse_args()
    if args.list_tasks:
        if args.archive_dir or args.published_release:
            parser.error("--archive-dir and --published-release require --tasks-dir")
        list_tasks(args.seed)
    else:
        if args.seed is not None:
            parser.error("--seed requires --list-tasks")
        if args.published_release:
            prepare_release(args.tasks_dir, args.archive_dir)
        else:
            prepare(args.tasks_dir, args.archive_dir)
