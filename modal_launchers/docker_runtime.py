"""One local Docker daemon per Modal training container."""

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def docker_daemon(log_path: Path):
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        daemon = subprocess.Popen(
            ["dockerd", "--iptables=false", "--bridge=none", "--storage-driver=vfs"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if daemon.poll() is not None:
                    raise RuntimeError(f"Docker exited during startup; see {log_path}")
                if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"Docker did not become ready; see {log_path}")
            subprocess.run(["docker", "compose", "version"], check=True, timeout=15)
            yield
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=30)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
