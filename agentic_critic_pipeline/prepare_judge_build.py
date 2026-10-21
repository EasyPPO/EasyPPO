"""Apply the reviewed infrastructure-error patch in an isolated judge build tree."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

UPSTREAM_REVISION = "aec2aff05f3ece1b150c7c40827b0cea449e0b28"
ENGINE_SHA256 = "7067bb6df7b35e8be408b60a5fd127306607e66180751e986d8b1e2927e3f2d7"
CLIENT_SHA256 = "4498fac1eab4629322f69f0a15c28aa84c90b01f1337fde5a15687fa0b030162"
BUILD_FILES = ("Dockerfile", "package.json", "package-lock.json", "server.js", "entrypoint.sh", "judge")


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    patch = Path(__file__).resolve().parent.parent / "patches/agentic_critic/frontier_judge_infrastructure.patch"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Judge patch supports only the pinned Frontier-CS revision; review changes before updating")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", *BUILD_FILES], check=True)
    original = source / "judge/src/judge_engine.js"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ENGINE_SHA256:
        raise ValueError("Judge engine differs from the reviewed source")
    if hashlib.sha256((source / "judge/src/gojudge.js").read_bytes()).hexdigest() != CLIENT_SHA256:
        raise ValueError("Compiler client differs from the reviewed source")
    output.mkdir(parents=True, exist_ok=False)
    for name in BUILD_FILES:
        if (source / name).is_dir():
            shutil.copytree(source / name, output / name)
        else:
            shutil.copy2(source / name, output / name)
    # git apply only processes this exact checked-in patch, without fuzzy matching.
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=output, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=output, check=True)
    patched = output / "judge/src/judge_engine.js"
    report = {
        "upstream_revision": revision,
        "original_engine_sha256": ENGINE_SHA256,
        "original_client_sha256": CLIENT_SHA256,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "patched_engine_sha256": hashlib.sha256(patched.read_bytes()).hexdigest(),
        "patched_client_sha256": hashlib.sha256((output / "judge/src/gojudge.js").read_bytes()).hexdigest(),
        "source_checkout_modified": False,
        "difference": (
            "Internal Error/File Error during C++ compilation, contestant or checker execution "
            "abort the submission as infrastructure errors"
        ),
    }
    (output / "agentic_judge_patch.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Pinned Frontier-CS algorithmic directory")
    parser.add_argument("--output", required=True, help="New isolated Docker build directory")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)))


if __name__ == "__main__":
    main()
