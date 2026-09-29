#!/usr/bin/env python3
"""CPU-only release checks; no Ray cluster, model, tokenizer or CUDA context."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

from check_easyppo_config import compose_config
from prepare_aime_data import validate_canonical

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "dapo-math-17k.parquet": (17917, "134671de0cd455477e3bbca80f125ea32094084ef1ae62e2fa3e1b066414ba4c"),
    "aime-2024.parquet": (30, "91f8ef5168ae2a7db9bc6860211b731de3b9358a993088c29a6d07cda90b0dd9"),
}
VERSIONS = {
    "torch": "2.11.0",
    "transformers": "5.9.0",
    "ray": "2.55.1",
    "tensordict": "0.10.0",
    "TransferQueue": "0.1.8",
    "hydra-core": "1.3.2",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Skip GPU-library checks in the local CPU test venv")
    args = parser.parse_args()
    import verl

    source_path = Path(verl.__file__).resolve().parent
    if source_path != ROOT / "verl":
        raise RuntimeError(f"Expected verl from {ROOT / 'verl'}, imported {source_path}")
    versions = dict(VERSIONS)
    if not args.local:
        versions.update({"vllm": "0.21.0", "flash-attn": "2.8.3"})
    for package, expected in versions.items():
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")
    for module in ["torch", "transformers", "ray", "transfer_queue", "verl.trainer.ppo.v1.trainer_base"]:
        importlib.import_module(module)
    if not args.local:
        importlib.import_module("flash_attn")
        importlib.import_module("vllm")
    for name, (rows, expected_hash) in DATA.items():
        path = ROOT / "data/aime" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Checksum mismatch: {name}")
        validate_canonical(path, rows)
    cfg = compose_config()
    assert cfg.data.filter_overlong_responses and not cfg.data.filter_overlong_responses_critic
    assert cfg.critic.variance_weight_beta == 0.5 and cfg.critic.variance_weight_min == 0.25
    assert cfg.data.train_batch_size // cfg.critic.ppo_mini_batch_size == 4
    from verl.utils.reward_score import default_compute_score

    correct = default_compute_score("math_dapo", "Answer: 42", "42")
    wrong = default_compute_score("math_dapo", "Answer: 41", "42")
    assert correct["score"] == 1 and wrong["score"] == -1, (correct, wrong)
    print(
        json.dumps({"status": "ok", "source_root": str(ROOT), "versions": versions, "models_loaded": False}, indent=2)
    )


if __name__ == "__main__":
    main()
