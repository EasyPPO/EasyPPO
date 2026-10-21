"""Export a saved Miles critic as a Hugging Face checkpoint."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.training_output / "miles_run.json").read_text())
    root = Path(run["checkpoints"])
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise FileNotFoundError(f"Miles did not produce a completed critic checkpoint: {tracker}")
    iteration = tracker.read_text().strip()
    if not iteration.isdigit():
        raise ValueError(f"Unexpected Miles checkpoint iteration: {iteration!r}")
    checkpoint = root / f"iter_{int(iteration):07d}"
    converter = Path(__file__).resolve().parents[2] / "scripts/agentic_critic/critic_to_hf.py"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--input-dir",
            str(checkpoint),
            "--base-hf-dir",
            str(args.model),
            "--out",
            str(args.output),
            "--miles",
            os.environ["MILES_DIR"],
        ],
        check=True,
    )
    (args.output / "training.json").write_text(
        json.dumps(
            {
                **run,
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": int(iteration),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
