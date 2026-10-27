"""Real local HF checkpoint -> critic cache -> exact 64-token probe on CPU."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from agentic_critic_pipeline.compact_grad import _cached_pairs, _episodes, main
from agentic_critic_pipeline.records import data_sha256


@pytest.fixture
def artifacts(tmp_path):
    torch.manual_seed(52)
    config = LlamaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    actor = LlamaForCausalLM(config).eval()
    base = tmp_path / "actor"
    actor.save_pretrained(base)
    critic = tmp_path / "critic"
    critic.mkdir()
    backbone = {key: value.clone() for key, value in actor.state_dict().items() if key.startswith("model.")}
    save_file(backbone, critic / "critic.safetensors")
    (critic / "meta.json").write_text(json.dumps(dict(converted_parameters=len(backbone))))
    torch.save(torch.nn.Linear(16, 1).state_dict(), critic / "value_head.pt")
    episodes = []
    for i in range(2):
        segments = []
        for _ in range(2):
            mask = [1] * 150
            mask[4] = mask[80] = 0
            ids = torch.randint(1, 31, (153,)).tolist()
            segments.append(dict(input_ids=ids, response_start=3, loss_mask=mask))
        episodes.append(
            dict(
                episode_id=f"e{i}",
                problem_id=f"p{i}",
                split="validation",
                reward=0.3 + 0.4 * i,
                segments=segments,
                n_compactions=1,
                finish_reason="max_turns",
            )
        )
    source = tmp_path / "episodes.jsonl"
    source.write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    return base, critic, source


def _hashes(root):
    return {str(p): data_sha256(p) for p in Path(root).rglob("*") if p.is_file()}


def _cache(artifacts, tmp_path, mode="actor_summary"):
    base, critic, source = artifacts
    cache = tmp_path / "cache"
    main(
        [
            "cache-values",
            "--base-model",
            str(base),
            "--critic",
            str(critic),
            "--episodes",
            str(source),
            "--output",
            str(cache),
            "--device",
            "cpu",
            "--critic-mode",
            mode,
        ]
    )
    return cache


@pytest.mark.parametrize("mode", ["actor_summary", "full_history"])
def test_complete_cli_uses_actual_64_token_windows_without_modifying_checkpoints(artifacts, tmp_path, mode):
    base, critic, source = artifacts
    before = {**_hashes(base), **_hashes(critic)}
    cache = _cache(artifacts, tmp_path, mode)
    manifest = json.loads((cache / "manifest.json").read_text())
    raw = list(_cached_pairs(source, cache))
    advantages = torch.tensor(
        [ep["reward"] - v for ep, values in raw for segment in values for v in segment], dtype=torch.float64
    )
    assert manifest["advantage_moments"]["mean"] == pytest.approx(float(advantages.mean()))
    assert manifest["advantage_moments"]["variance"] == pytest.approx(float(advantages.var()))
    output = tmp_path / "result"
    common = [
        "measure",
        "--actor",
        str(base),
        "--episodes",
        str(source),
        "--values",
        str(cache),
        "--device",
        "cpu",
        "--controls",
        "2",
        "--bootstrap",
        "20",
        "--head-chunk-size",
        "64",
    ]
    main(common + ["--output", str(output)])
    rows = [json.loads(line) for line in (output / "boundaries.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["segment_index"] == 1 and row["loss_tokens"] == 148
        assert len(row["target_positions"]) == 64
        assert 7 not in row["target_positions"]  # response offset 4 is a tool token
        assert len(row["control_positions"]) == 2
        assert all(len(p) == 64 and not set(p) & set(row["target_positions"]) for p in row["control_positions"])
        assert row["parameter_count"] > 0 and row["full_norm"] > 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["measured_boundaries"] == 2 and summary["problems"] == 2
    assert (output / "manifest.json").exists()
    # Cached values and seeds are reusable; layer checkpointing changes no result.
    second = tmp_path / "no_checkpoint"
    main(common + ["--output", str(second), "--no-checkpoint-layers"])
    other = [json.loads(line) for line in (second / "boundaries.jsonl").read_text().splitlines()]
    for a, b in zip(rows, other, strict=True):
        assert a["control_positions"] == b["control_positions"]
        assert a["ratio_to_control_mean"] == pytest.approx(b["ratio_to_control_mean"], rel=2e-6)
    assert before == {**_hashes(base), **_hashes(critic)}
    with pytest.raises(FileExistsError):
        main(common + ["--output", str(output)])


def test_rejects_stale_tampered_and_misaligned_cache(artifacts, tmp_path):
    base, _, source = artifacts
    cache = _cache(artifacts, tmp_path)
    args = [
        "measure",
        "--actor",
        str(base),
        "--episodes",
        str(source),
        "--values",
        str(cache),
        "--output",
        str(tmp_path / "result"),
        "--device",
        "cpu",
    ]
    original = source.read_text()
    source.write_text(original + "\n")
    with pytest.raises(ValueError, match="different episode"):
        main(args)
    source.write_text(original)
    path = cache / "values.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["values"][0].pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="checksum"):
        main(args)
    with pytest.raises(ValueError, match="misaligned"):
        list(_cached_pairs(source, cache))


def test_rejects_duplicate_or_incomplete_episodes(artifacts):
    _, _, source = artifacts
    original = source.read_text()
    source.write_text(original + original)
    with pytest.raises(ValueError, match="Duplicate"):
        list(_episodes(source))
    row = json.loads(original.splitlines()[0])
    row["finish_reason"] = "context_full"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="naturally completed"):
        list(_episodes(source))


@pytest.mark.parametrize("flag", ["--window-size", "--controls", "--head-chunk-size", "--bootstrap", "--max-tokens"])
def test_invalid_flags_fail_before_model_loading(flag):
    with pytest.raises(SystemExit):
        main(
            [
                "measure",
                "--actor",
                "missing",
                "--episodes",
                "missing",
                "--values",
                "missing",
                "--output",
                "missing",
                flag,
                "0",
            ]
        )
