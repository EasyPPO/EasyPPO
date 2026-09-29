# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    compute_prompt_variance_loss_weights,
    compute_prompt_variance_topk_shares,
    compute_value_loss,
)


class TestVarianceWeightedCriticLoss(unittest.TestCase):
    def test_prompt_variance_weights_use_floor(self):
        rewards = torch.tensor([0.0, 1.0, 0.5, 0.5])
        uids = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"])

        weights, variances = compute_prompt_variance_loss_weights(rewards, uids, beta=1.0, w_min=0.01)

        torch.testing.assert_close(variances, torch.tensor([0.25, 0.25, 0.0, 0.0]))
        torch.testing.assert_close(weights, torch.tensor([4.0, 4.0, 100.0, 100.0]) / 52.0)
        self.assertAlmostEqual(weights.mean().item(), 1.0)

    def test_prompt_variance_topk_shares_deduplicate_rollouts(self):
        variances = torch.tensor([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0])
        uids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

        shares = compute_prompt_variance_topk_shares(variances, uids)

        torch.testing.assert_close(shares[1], torch.tensor(0.4))
        torch.testing.assert_close(shares[2], torch.tensor(0.7))
        torch.testing.assert_close(shares[3], torch.tensor(0.9))

    def test_prompt_variance_topk_shares_are_zero_when_all_variances_are_zero(self):
        variances = torch.zeros(4)
        uids = np.array(["a", "a", "b", "b"])

        shares = compute_prompt_variance_topk_shares(variances, uids)

        self.assertEqual({k: value.item() for k, value in shares.items()}, {1: 0.0, 2: 0.0, 3: 0.0})

    def test_weighted_value_loss_matches_prompt_weighted_sequence_mean(self):
        vpreds = torch.ones((4, 2))
        values = vpreds.clone()
        returns = torch.zeros_like(vpreds)
        response_mask = torch.ones_like(vpreds)
        loss_weights = torch.tensor([4.0, 4.0, 100.0, 100.0])

        loss, _ = compute_value_loss(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="seq-mean-token-mean",
            global_batch_size=4,
            loss_weights=loss_weights,
        )

        self.assertAlmostEqual(loss.item(), 26.0)

    def test_weighted_value_loss_is_microbatch_invariant(self):
        vpreds = torch.arange(1, 9, dtype=torch.float32).view(4, 2)
        values = vpreds.clone()
        returns = torch.zeros_like(vpreds)
        response_mask = torch.ones_like(vpreds)
        loss_weights = torch.tensor([4.0, 4.0, 100.0, 100.0])
        kwargs = {
            "cliprange_value": 0.2,
            "loss_agg_mode": "seq-mean-token-mean",
            "global_batch_size": 4,
        }

        whole_loss, _ = compute_value_loss(
            vpreds,
            returns,
            values,
            response_mask,
            loss_weights=loss_weights,
            **kwargs,
        )
        accumulated_loss = sum(
            compute_value_loss(
                vpreds[start : start + 2],
                returns[start : start + 2],
                values[start : start + 2],
                response_mask[start : start + 2],
                loss_weights=loss_weights[start : start + 2],
                **kwargs,
            )[0]
            for start in (0, 2)
        )

        torch.testing.assert_close(accumulated_loss, whole_loss)


if __name__ == "__main__":
    unittest.main()
