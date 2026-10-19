"""Local Frontier-CS judge for `type: default` problems, without Docker.

Mirrors the scoring of the upstream go-judge based server (judge/src/judge_engine.js):
compile the C++17 solution, run it on every test case under the problem's time and memory
limits, run the testlib checker `chk in out ans`, and take per-case
`Ratio: x` from the checker message (1.0 if the checker accepts without a ratio, 0 on any
failure). The score is 100 * mean ratio.

Isolation is rlimits only (CPU, address space, file size, wall clock). There is no
filesystem or network sandbox, so only run it on a machine and account where that is acceptable.
"""

import hashlib
import os
import re
import resource
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CXX = ["/usr/bin/g++", "-O2", "-pipe", "-std=gnu++17"]
COMPILE_TIMEOUT_S = 60
CHECKER_CPU_S = 10
CHECKER_MEM_BYTES = 2 << 30  # address-space limit; 256 MiB is too small for testlib checkers
MAX_OUTPUT_BYTES = 128 << 20
MSG_CHARS = 2000


@dataclass
class CaseResult:
    name: str
    ratio: float
    status: str  # Accepted | Wrong Answer | Time Limit Exceeded | Runtime Error | Output Limit Exceeded
    time_s: float
    msg: str = ""


@dataclass
class JudgeResult:
    score: float  # 0..100
    status: str  # done | Compile Error
    compile_msg: str = ""
    cases: list[CaseResult] = field(default_factory=list)


def _to_seconds(v) -> float:
    m = re.fullmatch(r"\s*([\d.]+)\s*(ms|s)?\s*", str(v))
    return float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)


def _to_bytes(v) -> int:
    m = re.fullmatch(r"\s*([\d.]+)\s*([kmg]?)b?\s*", str(v).lower())
    return int(float(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2)])


def load_problem(problem_dir: str | Path) -> dict:
    problem_dir = Path(problem_dir)
    cfg = yaml.safe_load((problem_dir / "config.yaml").read_text()) or {}
    cases = sorted((problem_dir / "testdata").glob("*.in"), key=lambda p: (len(p.stem), p.stem))
    return {
        "dir": problem_dir,
        "type": cfg.get("type", "default"),
        "checker": problem_dir / cfg.get("checker", "chk.cc"),
        "time_s": _to_seconds(cfg.get("time", "1s")),
        "memory_bytes": _to_bytes(cfg.get("memory_limit") or cfg.get("memory") or "256m"),
        "cases": [(p, p.with_suffix(".ans")) for p in cases],
    }


def _limits(cpu_s: float, mem_bytes: int):
    def apply():
        cpu = int(cpu_s) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    return apply


def _compile(src: Path, out: Path, include: Path | None = None) -> tuple[bool, str]:
    cmd = CXX + ([f"-I{include}"] if include else []) + ["-o", str(out), str(src)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S, cwd=src.parent)
    except subprocess.TimeoutExpired:
        return False, "compilation timed out"
    return p.returncode == 0, p.stderr[-MSG_CHARS:]


def get_checker(problem: dict, testlib_dir: str | Path, cache_dir: str | Path) -> Path:
    """Compile the problem's checker once and cache the binary by source hash."""
    src = problem["checker"]
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out = Path(cache_dir) / f"chk_{digest}"
    if not out.exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            tmp_src = Path(tmp) / "chk.cc"
            shutil.copy(src, tmp_src)
            ok, msg = _compile(tmp_src, Path(tmp) / "chk", include=Path(testlib_dir))
            if not ok:
                raise RuntimeError(f"checker for {problem['dir'].name} failed to compile: {msg}")
            os.replace(Path(tmp) / "chk", out)
    return out


def _run_case(binary: Path, checker: Path, problem: dict, inp: Path, ans: Path, workdir: Path) -> CaseResult:
    name = inp.stem
    out_path, err_path = workdir / f"{name}.out", workdir / f"{name}.err"
    wall = problem["time_s"] * 2 + 1
    start = time.monotonic()
    try:
        with open(inp, "rb") as fin, open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
            p = subprocess.run(
                [str(binary)],
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                timeout=wall,
                cwd=workdir,
                env={"PATH": "/usr/bin:/bin"},
                preexec_fn=_limits(problem["time_s"], problem["memory_bytes"]),
            )
    except subprocess.TimeoutExpired:
        return CaseResult(name, 0.0, "Time Limit Exceeded", time.monotonic() - start)
    elapsed = time.monotonic() - start
    usage_exceeded = p.returncode in (-9, -24) or elapsed > problem["time_s"]  # SIGKILL / SIGXCPU
    if p.returncode != 0 or usage_exceeded:
        if p.returncode == -25:  # SIGXFSZ
            status = "Output Limit Exceeded"
        elif usage_exceeded:
            status = "Time Limit Exceeded"
        else:
            status = "Runtime Error"
        return CaseResult(name, 0.0, status, elapsed, err_path.read_bytes()[-MSG_CHARS:].decode(errors="replace"))

    try:
        c = subprocess.run(
            [str(checker), str(inp), str(out_path), str(ans)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=CHECKER_CPU_S * 2,
            cwd=workdir,
            env={"PATH": "/usr/bin:/bin"},
            preexec_fn=_limits(CHECKER_CPU_S, CHECKER_MEM_BYTES),
        )
    except subprocess.TimeoutExpired:
        return CaseResult(name, 0.0, "Wrong Answer", elapsed, "checker timed out")
    msg = (c.stdout or c.stderr or "")[-MSG_CHARS:]
    ok = c.returncode == 0
    m = re.search(r"Ratio: ([\d.]+)", msg)
    ratio = float(m.group(1)) if m else (1.0 if ok else 0.0)
    return CaseResult(name, ratio, "Accepted" if ratio == 1.0 else "Wrong Answer", elapsed, msg)


def judge(
    problem_dir: str | Path,
    code: str,
    *,
    testlib_dir: str | Path,
    cache_dir: str | Path,
    work_root: str | Path | None = None,
    parallel: int = 4,
) -> JudgeResult:
    problem = load_problem(problem_dir)
    if problem["type"] != "default":
        raise NotImplementedError(f"problem type {problem['type']!r} is not supported")
    checker = get_checker(problem, testlib_dir, cache_dir)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        workdir = Path(tmp)
        src = workdir / "main.cpp"
        src.write_text(code)
        ok, msg = _compile(src, workdir / "main")
        if not ok:
            return JudgeResult(0.0, "Compile Error", compile_msg=msg)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            cases = list(
                pool.map(lambda c: _run_case(workdir / "main", checker, problem, c[0], c[1], workdir), problem["cases"])
            )
    score = 100.0 * sum(c.ratio for c in cases) / len(cases) if cases else 0.0
    return JudgeResult(score, "done", cases=cases)
