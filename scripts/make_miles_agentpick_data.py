"""Prepare the selected csk4gby2 Harbor tasks under a shared terminal tasks directory."""

import argparse
import csv
import hashlib
import json
import random
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from make_miles_terminal_data import task_records, write_jsonl

_DATASET = "Fzz1/tb-training-mix-agentpick"
_REVISION = "6e3e6c8a01d84f06b9f18f61782170f4801d5330"
_ARCHIVES = {
    "clean": "4f705d011b5d8e47174d9176a951f3b496a6d3b6aa45c352acccd64e8d3d9298",
    "calibforge_clean": "57d806342919d0a8444392edc79c88657210048f310cccff81e3ac76e7bc40c8",
}


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


def _configure_task(task, row, tmax_resources):
    config = task / "task.toml"
    if row["corpus"] == "TMAX":
        # TMAX publishes a prebuilt image with setup already applied.
        first_line = (task / "environment/Dockerfile").read_text().splitlines()[0]
        directive, image = first_line.split()
        if directive != "FROM":
            raise ValueError(f"Expected a prebuilt image for {task.name}")
        resources = tmax_resources[task.name]
        config.write_text(
            f"[environment]\ndocker_image = {json.dumps(image)}\ncpus = {int(resources['req_cpus'])}\nmemory_mb = {int(resources['req_memory_mb'])}\nstorage_mb = {int(resources['est_disk_mb'])}\n"
        )
    elif row["corpus"] == "REBENCH":
        # The release puts these Harbor environment fields at the TOML root.
        raw = config.read_text()
        config.write_text("[environment]\n" + raw)


def prepare(tasks_dir, archive_dir):
    selected = selected_tasks()
    path = hf_hub_download(_DATASET, "data/clean.parquet", repo_type="dataset", revision=_REVISION)
    resources = pq.read_table(path, columns=["task_id", "req_cpus", "req_memory_mb", "est_disk_mb"])
    tmax_resources = {row["task_id"]: row for row in resources.to_pylist()}
    tasks_dir.mkdir(parents=True, exist_ok=True)
    dest = tasks_dir / "agentpick-csk4gby2"
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    with tempfile.TemporaryDirectory(dir=tasks_dir) as staging:
        root = Path(staging) / dest.name
        for split in _ARCHIVES:
            _extract(root, split, {row["task_id"] for row in selected if row["split"] == split}, archive_dir)
        records = []
        for index, row in enumerate(selected):
            task = root / "tasks" / row["task_id"]
            _configure_task(task, row, tmax_resources)
            (record,) = task_records(task, dataset=_DATASET, split="train", path_root=Path(staging))
            record["metadata"].update(
                source_index=index,
                source_split=row["split"],
                corpus=row["corpus"],
                source_task_id=row["task_id"],
                source_revision=_REVISION,
            )
            records.append(record)
        write_jsonl(root / "train.jsonl", records)
        root.rename(dest)
    print(f"Prepared {len(records)} tasks in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tasks-dir", type=Path)
    mode.add_argument("--list-tasks", action="store_true", help="Print the 543 historical task metadata rows as JSONL.")
    parser.add_argument("--seed", type=int, help="Shuffle seed for --list-tasks.")
    parser.add_argument(
        "--archive-dir", type=Path, help="Read the two pinned task archives from this local directory."
    )
    args = parser.parse_args()
    if args.list_tasks:
        if args.archive_dir:
            parser.error("--archive-dir requires --tasks-dir")
        list_tasks(args.seed)
    else:
        if args.seed is not None:
            parser.error("--seed requires --list-tasks")
        prepare(args.tasks_dir, args.archive_dir)
