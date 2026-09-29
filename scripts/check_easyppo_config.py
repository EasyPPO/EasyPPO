#!/usr/bin/env python3
"""Compose and validate the release recipe without importing the training stack."""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def compose_config(overrides=()):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        config = compose(config_name="easyppo_aime", overrides=list(overrides))
    batch = config.data.train_batch_size
    n = config.actor_rollout_ref.rollout.n
    actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    critic = config.critic.ppo_mini_batch_size
    gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    if min(batch, n, actor, critic, gpus) < 1 or batch % actor or batch % critic:
        raise ValueError("Rollout batch must be divisible by positive actor/critic mini-batches")
    if (actor * n) % gpus or (critic * n) % gpus:
        raise ValueError("Actor and critic response mini-batches must be divisible by the GPU count")
    if gpus % config.actor_rollout_ref.rollout.tensor_model_parallel_size:
        raise ValueError("GPU count must be divisible by rollout tensor parallelism")
    fsdp_sizes = [
        config.actor_rollout_ref.actor.fsdp_config.fsdp_size,
        config.actor_rollout_ref.ref.fsdp_config.fsdp_size,
        config.critic.fsdp.fsdp_size,
    ]
    if any(size != -1 and (size < 1 or gpus % size) for size in fsdp_sizes):
        raise ValueError("Each FSDP group size must divide the total GPU count (or be -1)")
    return config


if __name__ == "__main__":
    print(OmegaConf.to_yaml(compose_config(sys.argv[1:]), resolve=True))
