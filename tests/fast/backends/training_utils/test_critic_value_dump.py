from types import SimpleNamespace

import torch

from miles.backends.training_utils.metrics.train_dump import save_debug_critic_values


def test_dump_holds_the_critic_fields_on_cpu(tmp_path):
    args = SimpleNamespace(save_debug_train_data=str(tmp_path / "{rollout_id}_{rank}.pt"))
    data = {"values": [torch.tensor([0.1, 0.2]), torch.tensor([0.3])], "returns": [torch.ones(2), torch.ones(1)],
            "rewards": [1.0, 1.0], "sample_indices": [7, 9], "tokens": [[1, 2, 3], [4, 5]], "value_slices": [(0, 0, 2), (1, 0, 1)]}
    path = save_debug_critic_values(args, rollout_id=3, rollout_data=data, rank=0)
    assert path.name == "3_0.critic.pt"
    got = torch.load(path)
    assert got["rollout_id"] == 3 and got["sample_indices"] == [7, 9] and got["value_slices"] == [(0, 0, 2), (1, 0, 1)]
    assert "tokens" not in got
    assert torch.equal(got["values"][0], torch.tensor([0.1, 0.2])) and got["values"][1].device.type == "cpu"


def test_no_path_no_dump(tmp_path):
    assert save_debug_critic_values(SimpleNamespace(save_debug_train_data=None), rollout_id=0, rollout_data={}, rank=0) is None
