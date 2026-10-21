"""Explicit Frontier-CS judge clients. Local execution is for disposable smoke hosts.

Scoring follows FrontierCS/Frontier-CS algorithmic/judge/src/judge_engine.js,
including testlib's partial-score exit code 7. No test contents reach the actor.
"""

import hashlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml


def request_json(url, payload=None, timeout=120):
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class HTTPJudge:
    def __init__(self, url, timeout=900):
        self.url, self.timeout = url.rstrip("/"), timeout
        request_json(self.url + "/health")

    def __call__(self, problem, code):
        sid = request_json(self.url + "/submit", {"pid": problem["judge_id"], "lang": "cpp", "code": code})["sid"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = request_json(f"{self.url}/result/{sid}")
            if result["status"] == "error":
                error = str(result.get("error", ""))
                # Checker compilation and service failures must not become reward=0.
                compile_failure = error.startswith("Error: compile failed:") or error.startswith("compile failed:")
                diagnostic = error.split("compile failed:", 1)[-1].strip()
                infrastructure = diagnostic in {"Internal Error", "File Error"} or any(
                    marker in diagnostic.lower()
                    for marker in (
                        "failed to get environment",
                        "disk full",
                        "no space left",
                        "connection refused",
                    )
                )
                if compile_failure and not infrastructure:
                    return {"score": 0.0, "status": "Compile Error", "detail": error[-500:], "cases": []}
                raise RuntimeError(f"Judge infrastructure failure for submission {sid}: {error}")
            if result["status"] == "done":
                score = float(result["score"])
                if not math.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError(f"Invalid bounded judge score: {score}")
                cases = [{"status": c["status"], "ratio": c["scoreRatio"]} for c in result["cases"]]
                return {"score": score, "status": "done", "cases": cases, "submission_id": sid}
            time.sleep(1)
        raise TimeoutError(f"Judge submission {sid} timed out")


def quantity(value, units):
    match = re.fullmatch(r"\s*([\d.]+)\s*([a-z]*)\s*", str(value).lower())
    if match is None or match[2] not in units:
        raise ValueError(f"Unsupported limit {value!r}")
    return float(match[1]) * units[match[2]]


def _limits(cpu, memory):
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(cpu), math.ceil(cpu) + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run(command, cwd, cpu, memory, stdin, stdout, stderr):
    start = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        preexec_fn=lambda: _limits(cpu, memory),
        env={"PATH": "/usr/bin:/bin"},
    )
    try:
        proc.wait(timeout=cpu * 2 + 1)
        return proc.returncode, time.monotonic() - start
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return -signal.SIGXCPU, time.monotonic() - start


class LocalJudge:
    """C++17/testlib execution with rlimits, without filesystem/network isolation.

    Requires explicit --allow-local-execution. Sequential only: do not call from
    thread pools because resource limits use POSIX preexec_fn.
    """

    def __init__(self, cache_dir, testlib_dir):
        self.cache = Path(cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.testlib = Path(testlib_dir).resolve()
        if not (self.testlib / "testlib.h").is_file():
            raise FileNotFoundError(self.testlib / "testlib.h")

    def compile(self, source, binary, include=None):
        cmd = ["g++", "-O2", "-pipe", "-std=gnu++17", str(source), "-o", str(binary)]
        if include:
            cmd += ["-I", str(include)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return result.returncode == 0, result.stderr[-500:]

    def __call__(self, problem, code):
        root = Path(problem["path"]).resolve()
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        if cfg.get("type", "default") != "default":
            raise ValueError("LocalJudge only implements type: default; use official HTTP judge for interactive")
        if any(set(s) - {"score", "n_cases"} for s in cfg.get("subtasks", [])):
            raise ValueError("LocalJudge requires uniform case limits; use official HTTP judge for subtask overrides")
        source = root / cfg.get("checker", "chk.cc")
        digest = hashlib.sha256(source.read_bytes() + (self.testlib / "testlib.h").read_bytes()).hexdigest()
        checker = self.cache / digest
        if not checker.exists():
            with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
                binary = Path(tmp) / "checker"
                ok, error = self.compile(source, binary, self.testlib)
                if not ok:
                    raise RuntimeError(f"Checker compilation failed: {error}")
                os.replace(binary, checker)
        cpu = quantity(cfg.get("time_limit", cfg.get("time", "1s")), {"": 1, "s": 1, "ms": 0.001})
        memory = int(
            quantity(
                cfg.get("memory_limit", cfg.get("memory", "256m")),
                {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "m": 1 << 20, "mb": 1 << 20, "g": 1 << 30, "gb": 1 << 30},
            )
        )
        inputs = sorted((root / "testdata").glob("*.in"))
        if not inputs or any(not x.with_suffix(".ans").exists() for x in inputs):
            raise ValueError(f"Missing judge cases/answers at {root}")
        cases = []
        with tempfile.TemporaryDirectory(prefix="agentic-judge-") as tmp:
            work = Path(tmp)
            (work / "solution.cpp").write_text(code)
            ok, error = self.compile(work / "solution.cpp", work / "solution")
            if not ok:
                return {"score": 0.0, "status": "Compile Error", "detail": error, "cases": []}
            for inp in inputs:
                with inp.open("rb") as fin, (work / "output").open("wb") as fout, (work / "stderr").open("wb") as ferr:
                    status, elapsed = _run([str(work / "solution")], work, cpu, memory, fin, fout, ferr)
                if status:
                    cases.append(
                        {
                            "name": inp.stem,
                            "ratio": 0.0,
                            "status": "Time Limit Exceeded" if status in (-9, -24) else "Runtime Error",
                            "wall_seconds": elapsed,
                        }
                    )
                    continue
                with (work / "checker-out").open("wb") as out, (work / "checker-err").open("wb") as err:
                    checked, _ = _run(
                        [str(checker), str(inp), str(work / "output"), str(inp.with_suffix(".ans"))],
                        work,
                        10,
                        256 << 20,
                        subprocess.DEVNULL,
                        out,
                        err,
                    )
                # Only the checker's own deliberate scoring verdict may award credit.
                message = (work / "checker-out").read_text(errors="replace") or (work / "checker-err").read_text(
                    errors="replace"
                )
                if checked not in (0, 1, 2, 3, 7):
                    raise RuntimeError(f"Checker infrastructure failure {checked} for {root}/{inp.name}")
                if checked == 3:
                    raise RuntimeError(f"Checker _fail verdict for {root}/{inp.name}: {message[-500:]}")
                match = re.search(r"Ratio: ([\d.]+)", message) if checked in (0, 7) else None
                ratio = float(match[1]) if match else float(checked == 0)
                if not math.isfinite(ratio) or not 0 <= ratio <= 1:
                    raise ValueError(f"Checker returned invalid ratio {ratio}")
                cases.append(
                    {
                        "name": inp.stem,
                        "ratio": ratio,
                        "status": "Accepted" if ratio == 1 else "Wrong Answer",
                        "wall_seconds": elapsed,
                    }
                )
        return {"score": 100 * sum(c["ratio"] for c in cases) / len(cases), "status": "done", "cases": cases}
