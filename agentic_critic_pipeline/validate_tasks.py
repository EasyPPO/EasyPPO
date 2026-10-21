"""Probe pinned task checkers with empty contestant output before selecting problems.

This detects checker compilation errors and input/answer failures encountered by
that execution path. It does not prove that every input field was read: some
checkers reject empty contestant output before reading the complete input.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .data import inventory, verify_problem
from .environment import LocalJudge, _run

LIMITATIONS = (
    "Empty output exercises only one checker path. A normal WA/PE or score verdict does not prove "
    "complete input validity: a checker may read contestant output before the entire input/answer. "
    "Runtime checker failures remain fatal; this preflight never converts them to zero rewards."
)


def validate_task(job):
    problem, testlib, cache, *limits = job
    cpu_seconds, memory_bytes = limits if limits else (60, 256 << 20)
    if cpu_seconds < 1 or memory_bytes < 1:
        raise ValueError("Probe CPU seconds and memory bytes must be positive")
    root = Path(problem["path"])
    import yaml

    config = yaml.safe_load((root / "config.yaml").read_text())
    source = root / config.get("checker", "chk.cc")
    result = {
        "status": "passed",
        "reasons": [],
        "cases": [],
        "files_sha256": problem["files_sha256"],
    }

    def fail(stage, case, returncode, message):
        result["status"] = "failed"
        result["reasons"].append(dict(stage=stage, case=case, returncode=returncode, message=message))

    try:
        judge = LocalJudge(cache, testlib)
        fingerprint = hashlib.sha256(source.read_bytes() + (Path(testlib) / "testlib.h").read_bytes()).hexdigest()
        checker = Path(cache) / fingerprint
        if not checker.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                binary = Path(temporary) / "checker"
                ok, message = judge.compile(source, binary, testlib)
                if not ok:
                    resource_failure = any(
                        marker in message.lower()
                        for marker in (
                            "killed signal",
                            "out of memory",
                            "cannot allocate memory",
                            "no space left",
                            "disk quota",
                            "permission denied",
                            "resource temporarily unavailable",
                            "cannot execute",
                            "internal compiler error",
                            "unable to execute",
                        )
                    )
                    fail("infrastructure" if resource_failure else "compile", None, None, message)
                    return problem["judge_id"], result
                os.replace(binary, checker)
        inputs = sorted((root / "testdata").glob("*.in"))
        with tempfile.TemporaryDirectory(prefix="task-preflight-") as temporary:
            work = Path(temporary)
            output = work / "empty-output"
            output.touch()
            for inp in inputs:
                with (work / "stdout").open("wb") as stdout, (work / "stderr").open("wb") as stderr:
                    code, _ = _run(
                        [str(checker), str(inp), str(output), str(inp.with_suffix(".ans"))],
                        work,
                        cpu_seconds,
                        memory_bytes,
                        subprocess.DEVNULL,
                        stdout,
                        stderr,
                    )
                message = (
                    (work / "stdout").read_text(errors="replace") + (work / "stderr").read_text(errors="replace")
                )[-2000:]
                verdict = {
                    0: "ok",
                    1: "wrong_answer",
                    2: "presentation_error",
                    3: "checker_fail",
                    7: "partial_score",
                }.get(code, "abnormal_exit")
                result["cases"].append(dict(case=inp.name, returncode=code, verdict=verdict, message=message))
                if code == 3:
                    fail("checker_probe", inp.name, code, message)
                elif code not in (0, 1, 2, 7):
                    fail("infrastructure", inp.name, code, message or "Checker exited abnormally")
        verify_problem(problem)
    except Exception as error:
        fail("infrastructure", None, None, repr(error))
    return problem["judge_id"], result


def validate_tasks(
    root, dataset, testlib, cache, output, workers=16, probe_cpu_seconds=60, probe_memory_bytes=256 << 20
):
    if workers < 1 or probe_cpu_seconds < 1 or probe_memory_bytes < 1:
        raise ValueError("workers and probe resource limits must be positive")
    root, testlib, cache, output = [Path(p).resolve() for p in (root, testlib, cache, output)]
    if not (testlib / "testlib.h").is_file():
        raise FileNotFoundError(testlib / "testlib.h")
    problems = inventory(root, dataset)
    if not problems:
        raise ValueError("No eligible noninteractive tasks found")
    cache.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "dataset": dataset,
        "root": str(root),
        "source_revision": problems[0]["source_revision"],
        "scope": "empty_output_checker_preflight",
        "limitations": LIMITATIONS,
        "testlib_sha256": hashlib.sha256((testlib / "testlib.h").read_bytes()).hexdigest(),
        "probe_limits": {"cpu_seconds": probe_cpu_seconds, "address_space_bytes": probe_memory_bytes},
        "tasks": {},
    }
    jobs = [(p, str(testlib), str(cache), probe_cpu_seconds, probe_memory_bytes) for p in problems]
    # Spawned processes each have one main thread; _run's POSIX preexec resource
    # limits are never invoked from ThreadPoolExecutor workers.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for identity, row in pool.map(validate_task, jobs):
            report["tasks"][identity] = row
            print(json.dumps({"task": identity, "status": row["status"], "reasons": row["reasons"]}), flush=True)
    report["fatal_errors"] = [
        {"task": identity, **reason}
        for identity, row in report["tasks"].items()
        for reason in row["reasons"]
        if reason["stage"] == "infrastructure"
    ]
    report["status"] = "infrastructure_failed" if report["fatal_errors"] else "complete"
    report["summary"] = {
        "tasks": len(problems),
        "passed": sum(row["status"] == "passed" for row in report["tasks"].values()),
        "failed": sum(row["status"] == "failed" for row in report["tasks"].values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    if report["fatal_errors"]:
        raise RuntimeError(f"Task preflight infrastructure failure; inspect {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--testlib", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--probe-cpu-seconds", type=int, default=60)
    parser.add_argument("--probe-memory-bytes", type=int, default=256 << 20)
    args = parser.parse_args()
    report = validate_tasks(**vars(args))
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
