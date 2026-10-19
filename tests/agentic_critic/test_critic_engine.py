"""End-to-end smoke of scripts/critic_engine.py on a tiny random model: every setting trains and stores its values."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
REPO = Path(__file__).resolve().parents[2]
VOCAB = 512


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5TextConfig
    except ImportError:
        pytest.skip("this transformers version has no Qwen3.5")
    d = tmp_path_factory.mktemp("engine")
    cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=8, linear_value_head_dim=8,
                            vocab_size=VOCAB, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])  # fmt: skip
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d / "model")
    rng, index = random.Random(0), 0
    (d / "rollouts").mkdir()
    for step in range(2):
        samples = []
        for ep in range(8):  # 2 problems x 4 episodes
            reward = rng.random()
            for k in range(rng.randint(1, 3)):
                prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(3, 6))]
                resp = [rng.randrange(VOCAB) for _ in range(rng.randint(4, 8))]
                mask = [1] * (len(resp) - 1) + [0]
                samples.append({"index": index, "rollout_id": step * 100 + ep, "tokens": prompt + resp, "response_length": len(resp), "loss_mask": mask,
                                "reward": reward, "metadata": {"problem": f"p{step}_{ep // 4}", "sample": ep % 4, "segment": k}})  # fmt: skip
                index += 1
        torch.save({"rollout_id": step, "samples": samples}, d / "rollouts" / f"rollout_{step}.pt")
    return d


@pytest.mark.parametrize("mode", ["shared", "full", "full_detach", "kv"])
def test_engine_trains_and_stores_values(workdir, mode):
    out = workdir / f"out_{mode}"
    r = subprocess.run([sys.executable, str(REPO / "scripts/critic_engine.py"), "--model", str(workdir / "model"), "--rollouts", str(workdir / "rollouts"),
                        "--mode", mode, "--out", str(out), "--device", "cpu", "--no-autocast", "--lr", "1e-3", "--head-init-std", "0.5"],
                       capture_output=True, text=True)  # fmt: skip
    if mode == "kv" and "compaction" in r.stderr and "No such file" in r.stderr:
        pytest.skip("upstream compaction repo not available")
    assert r.returncode == 0, r.stderr[-2000:]
    log = json.loads((out / "log.json").read_text())
    assert [s["step"] for s in log["steps"]] == [0, 1] and all(len(s["updates"]) == 4 for s in log["steps"])
    assert all(u["grad_norm"] > 0 and u["value_loss"] > 0 for s in log["steps"] for u in s["updates"])
    check = log["self_check"]
    assert check["episodes"] >= 1 and check["first_segment"] < 1e-4 and check["train_forward"] < 1e-4
    assert (check["later_segments"] is None) == (mode == "shared") and (check["kv_keep_all"] is not None) == (mode == "kv")
    samples = torch.load(workdir / "rollouts" / "rollout_1.pt", weights_only=False)["samples"]
    dump = torch.load(out / "dump" / "1_0.critic.pt", weights_only=False)
    assert sorted(dump["sample_indices"]) == sorted(s["index"] for s in samples)
    by_index = {s["index"]: s for s in samples}
    for idx, v, m in zip(dump["sample_indices"], dump["values"], dump["loss_masks"], strict=True):
        assert len(v) == by_index[idx]["response_length"] and m.tolist() == by_index[idx]["loss_mask"] and torch.isfinite(v).all()


def test_engine_stops_at_the_end_marker_of_a_growing_directory(workdir):
    import shutil

    grow = workdir / "growing"
    grow.mkdir()
    shutil.copy(workdir / "rollouts" / "rollout_0.pt", grow / "rollout_0.pt")
    (grow / "END").write_text("")
    out = workdir / "out_growing"
    r = subprocess.run([sys.executable, str(REPO / "scripts/critic_engine.py"), "--model", str(workdir / "model"), "--rollouts", str(grow), "--mode", "shared",
                        "--out", str(out), "--device", "cpu", "--no-autocast", "--wait-for-end", "--self-check", "0"], capture_output=True, text=True, timeout=600)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
    assert [s["step"] for s in json.loads((out / "log.json").read_text())["steps"]] == [0]
