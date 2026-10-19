"""Noise-normalised critic weights with compact rollouts: an episode counts once, however many segments it has."""

import torch

from miles.ray.rollout.train_data_conversion import _compute_critic_loss_weights


def test_segments_of_one_rollout_count_once():
    # prompt 0: episodes with rewards 0.0 (3 segments) and 1.0 (1 segment); prompt 1: 0.2 and 0.4 (1 segment each)
    rewards = [0.0, 0.0, 0.0, 1.0, 0.2, 0.4]
    rollout_ids = [10, 10, 10, 11, 12, 13]
    masks = [[1]] * 6
    groups = [[0, 1, 2, 3], [4, 5]]
    per_segment = _compute_critic_loss_weights(rewards=rewards, loss_masks=masks, prompt_groups=groups, std_floor=0.075)
    per_episode = _compute_critic_loss_weights(rewards=rewards, loss_masks=masks, prompt_groups=groups, std_floor=0.075, rollout_ids=rollout_ids)
    raw = torch.tensor([1 / 0.5, 1 / 0.1])  # stds over episodes: 0.5 and 0.1
    expected = (raw / raw.mean()).tolist()
    assert all(abs(w - expected[0]) < 1e-5 for w in per_episode[:4]) and all(abs(w - expected[1]) < 1e-5 for w in per_episode[4:])
    assert abs(per_segment[0] - per_episode[0]) > 1e-3  # counting segments gives another std for prompt 0
    # the same episodes as one sample each (what a full_history critic trains on) get the same weights
    whole = _compute_critic_loss_weights(rewards=[0.0, 1.0, 0.2, 0.4], loss_masks=[[1]] * 4, prompt_groups=[[0, 1], [2, 3]], std_floor=0.075,
                                         rollout_ids=[10, 11, 12, 13])  # fmt: skip
    assert abs(whole[0] - per_episode[0]) < 1e-6 and abs(whole[2] - per_episode[4]) < 1e-6


def test_unique_rollout_ids_change_nothing():
    rewards, masks, groups = [0.1, 0.5, 0.9, 0.3], [[1]] * 4, [[0, 1], [2, 3]]
    a = _compute_critic_loss_weights(rewards=rewards, loss_masks=masks, prompt_groups=groups, std_floor=0.075)
    b = _compute_critic_loss_weights(rewards=rewards, loss_masks=masks, prompt_groups=groups, std_floor=0.075, rollout_ids=[0, 1, 2, 3])
    assert a == b
