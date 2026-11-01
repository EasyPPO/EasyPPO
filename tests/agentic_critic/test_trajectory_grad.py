"""Exact gradient algebra, independent critics, paired worker output and figures."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from agentic_critic_pipeline.gradient_models import actor_loss_function
from agentic_critic_pipeline.records import data_sha256
from agentic_critic_pipeline.trajectory_grad import measure_trajectory
from agentic_critic_pipeline.trajectory_grad_comparison import main, merge_results, prepare
from agentic_critic_pipeline.trajectory_grad_figures import render, summarize


@pytest.fixture
def experiment(tmp_path):
    torch.manual_seed(78)
    config = LlamaConfig(
        vocab_size=23,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    actor = LlamaForCausalLM(config).eval()
    base = tmp_path / "actor"
    actor.save_pretrained(base)
    critics = []
    for i in range(2):
        critic = tmp_path / f"critic_{i}"
        critic.mkdir()
        state = {k: v.clone() + 0.001 * i for k, v in actor.state_dict().items() if k.startswith("model.")}
        save_file(state, critic / "critic.safetensors")
        head = torch.nn.Linear(8, 1)
        torch.save(head.state_dict(), critic / "value_head.pt")
        (critic / "meta.json").write_text(json.dumps(dict(converted_parameters=len(state))))
        critics.append(critic)
    episodes = []
    for i in range(3):
        segments = []
        for j in range(i + 1):
            mask = [1] * (4 + 2 * j)
            mask[1] = 0  # Tool/context token contributes no loss, but remains in the context.
            if i == 2 and j == 1:
                mask = [0] * len(mask)  # Empty-loss segment is retained as a zero contribution.
            segments.append(
                dict(
                    input_ids=torch.randint(1, 23, (3 + len(mask),)).tolist(),
                    response_start=3,
                    loss_mask=mask,
                )
            )
        episodes.append(
            dict(
                episode_id=f"e{i}",
                problem_id=f"p{i % 2}",
                split="validation",
                reward=0.3 + 0.1 * i,
                segments=segments,
                n_compactions=i,
                finish_reason="max_turns",
            )
        )
    source = tmp_path / "episodes.jsonl"
    source.write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    return actor, base, critics, source, episodes


@pytest.mark.parametrize("normalization", ["none", "cohort"])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_two_pass_matches_direct_whole_trajectory_gradient(experiment, normalization, checkpoint):
    model, _, _, _, episodes = experiment
    episode = episodes[2]
    values = [[0.02 * i for i in range(sum(s["loss_mask"]))] for s in episode["segments"]]
    moments = dict(mean=0.13, variance=0.23)
    parameters = tuple(model.parameters())
    # Verify user gradients/flags are preserved, and tied parameters count once.
    model.train()
    for p in parameters:
        p.grad = torch.ones_like(p)
    original = {name: p.clone() for name, p in model.named_parameters()}
    trajectory, rows = measure_trajectory(
        model,
        episode,
        values,
        moments=moments,
        normalization=normalization,
        checkpoint=checkpoint,
        head_chunk_size=3,
    )
    assert model.training and all(torch.equal(p.grad, torch.ones_like(p)) for p in parameters)
    assert all(torch.equal(original[name], p) for name, p in model.named_parameters())
    assert trajectory["parameter_count"] == sum(p.numel() for p in parameters)
    model.eval()
    total = sum(sum(s["loss_mask"]) for s in episode["segments"])
    losses = []
    for s, v in zip(episode["segments"], values, strict=True):
        if not v:
            losses.append(None)
            continue
        advantages = torch.tensor([episode["reward"] - x for x in v])
        if normalization == "cohort":
            advantages = (advantages - moments["mean"]) / (moments["variance"] + 1e-8) ** 0.5
        losses.append(actor_loss_function(model, s, advantages, head_chunk_size=3)().sum() / total)
    full = torch.autograd.grad(sum(loss for loss in losses if loss is not None), parameters, retain_graph=True)
    flat = torch.cat([g.flatten() for g in full]).double()
    assert trajectory["trajectory_grad_norm"] == pytest.approx(float(flat.norm()), rel=2e-6)
    assert trajectory["projection_sum"] == pytest.approx(1, abs=2e-6)
    for i, (loss, row) in enumerate(zip(losses, rows, strict=True)):
        assert row["prior_compactions"] == i and row["total_compactions"] == 2
        if loss is None:
            assert row["segment_grad_norm"] == 0 and row["cosine_with_trajectory"] is None
            assert row["segment_mean_grad_norm"] is None
            continue
        grad = torch.autograd.grad(loss, parameters, retain_graph=True)
        g = torch.cat([x.flatten() for x in grad]).double()
        assert row["segment_grad_norm"] == pytest.approx(float(g.norm()), rel=2e-6)
        assert row["cosine_with_trajectory"] == pytest.approx(
            float(torch.dot(g, flat) / (g.norm() * flat.norm())), abs=2e-6
        )
        assert row["norm_ratio"] == pytest.approx(float(g.norm() / flat.norm()), rel=2e-6)
        assert row["segment_mean_norm_ratio"] == pytest.approx(row["norm_ratio"] / row["token_fraction"])


def test_zero_gradients_remain_null_instead_of_nan(experiment):
    model, _, _, _, episodes = experiment
    episode = episodes[0]
    values = [[episode["reward"]] * sum(episode["segments"][0]["loss_mask"])]
    trajectory, rows = measure_trajectory(model, episode, values, moments={})
    assert trajectory["trajectory_grad_norm"] == rows[0]["segment_grad_norm"] == 0
    assert rows[0]["norm_ratio"] is None and rows[0]["cosine_with_trajectory"] is None
    json.dumps([trajectory, rows], allow_nan=False)


def test_cancelling_segments_have_negative_cosine_and_ratio_above_one(monkeypatch):
    model = torch.nn.Linear(1, 1)

    def scalar_losses(model, segment, advantages, **kwargs):
        return lambda: model.weight.flatten() * advantages

    monkeypatch.setattr("agentic_critic_pipeline.trajectory_grad.actor_loss_function", scalar_losses)
    segment = dict(input_ids=[1, 2], response_start=1, loss_mask=[1])
    episode = dict(reward=1, segments=[segment, segment])
    trajectory, rows = measure_trajectory(model, episode, [[-3], [3]], moments={}, checkpoint=False)
    assert trajectory["trajectory_grad_norm"] == 1
    assert trajectory["projection_sum"] == 1
    assert [row["norm_ratio"] for row in rows] == [2, 1]
    assert [row["cosine_with_trajectory"] for row in rows] == [1, -1]
    # The unused bias is represented by zero, without changing .grad.
    assert model.bias.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen35_hybrid_actor_matches_ordinary_hf_trajectory_backward(dtype):
    torch.manual_seed(34)
    config = Qwen3_5TextConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config).to(dtype).eval()
    segments = [
        dict(input_ids=[1, 2, 4, 3, 7, 9], response_start=2, loss_mask=[1, 0, 1, 1]),
        dict(input_ids=[1, 3, 2, 8], response_start=2, loss_mask=[1, 1]),
    ]
    episode = dict(reward=0.7, segments=segments)
    values = [[0.1, 0.3, 0.8], [0.6, 0.2]]
    trajectory, rows = measure_trajectory(model, episode, values, moments={})
    loss = 0
    for s, v in zip(segments, values, strict=True):
        ids = torch.tensor([s["input_ids"]])
        positions = [s["response_start"] + i - 1 for i, flag in enumerate(s["loss_mask"]) if flag]
        logits = model(ids, use_cache=False).logits[0, positions].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[0, torch.tensor(positions) + 1], reduction="none")
        loss = loss + (nll * torch.tensor([0.7 - x for x in v])).sum() / 5
    full = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
    norm = sum(float(g.double().square().sum()) for g in full if g is not None) ** 0.5
    assert trajectory["trajectory_grad_norm"] == pytest.approx(norm, rel=0.02 if dtype == torch.bfloat16 else 3e-6)
    assert trajectory["projection_sum"] == pytest.approx(1, abs=3e-6)
    assert all(row["cosine_with_trajectory"] is not None for row in rows)


def test_sampling_keeps_complete_histories_and_fails_on_overlength(experiment, tmp_path):
    _, _, _, source, _ = experiment
    args = Namespace(episodes=str(source), seed=4, trajectories=2, workers=2, max_tokens=120000)
    first, second = tmp_path / "first", tmp_path / "second"
    a, b = prepare(args, first), prepare(args, second)
    assert a == b and (first / "cohort.jsonl").read_bytes() == (second / "cohort.jsonl").read_bytes()
    for line in (first / "cohort.jsonl").read_text().splitlines():
        episode = json.loads(line)
        assert len(episode["segments"]) == episode["n_compactions"] + 1
    args.trajectories = 4
    with pytest.raises(ValueError, match="more trajectories"):
        prepare(args, tmp_path / "too_many")
    args.trajectories, args.max_tokens = None, 1
    with pytest.raises(ValueError, match="truncation"):
        prepare(args, tmp_path / "overlength")


def _args(base, critics, source, root, workers):
    return [
        "run",
        "--actor",
        str(base),
        "--actor-summary-critic",
        str(critics[0]),
        "--full-history-critic",
        str(critics[1]),
        "--episodes",
        str(source),
        "--output",
        str(root),
        "--workers",
        str(workers),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--bootstrap",
        "20",
        "--head-chunk-size",
        "3",
        "--advantage-normalization",
        "cohort",
    ]


def test_two_worker_experiment_pairs_all_segments_and_saves_reproducible_figures(experiment, tmp_path):
    _, base, critics, source, episodes = experiment
    before = {str(p): data_sha256(p) for directory in [base, *critics] for p in directory.rglob("*") if p.is_file()}
    source_hash = data_sha256(source)
    root = tmp_path / "parallel"
    main(_args(base, critics, source, root, 2))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["selection"]["trajectories"] == 3
    report = root / "gradient_report"
    assert len(list(report.glob("*.png"))) == len(list(report.glob("*.pdf"))) == 5
    plot_path = report / "plot_data.json"
    plot_hash = data_sha256(plot_path)
    summary = render(root, bootstrap=20)
    assert data_sha256(plot_path) == plot_hash
    for mode in ("actor_summary", "full_history"):
        assert summary[mode]["trajectories"] == 3 and summary[mode]["segments"] == 6
        assert summary[mode]["metrics"]["cosine_with_trajectory"]["undefined"] == 1
        rows = [json.loads(line) for line in (root / mode / "trajectories.jsonl").read_text().splitlines()]
        assert all(row["projection_sum"] == pytest.approx(1, abs=3e-6) for row in rows)
        segment_rows = [json.loads(line) for line in (root / mode / "segments.jsonl").read_text().splitlines()]
        assert len({(r["episode_id"], r["segment_index"]) for r in segment_rows}) == sum(
            len(e["segments"]) for e in episodes
        )
        assert segment_rows[0]["norm_ratio"] == pytest.approx(1)
        assert segment_rows[0]["cosine_with_trajectory"] == pytest.approx(1)
    assert source_hash == data_sha256(source)
    assert before == {name: data_sha256(name) for name in before}
    # A serial execution must use the same whitening moments and measurements.
    serial = tmp_path / "serial"
    main(_args(base, critics, source, serial, 1))
    for mode in ("actor_summary", "full_history"):
        a = [json.loads(s) for s in (root / mode / "segments.jsonl").read_text().splitlines()]
        b = [json.loads(s) for s in (serial / mode / "segments.jsonl").read_text().splitlines()]
        for left, right in zip(a, b, strict=True):
            for metric in ("segment_grad_norm", "trajectory_grad_norm", "norm_ratio", "cosine_with_trajectory"):
                assert left[metric] == pytest.approx(right[metric], rel=3e-6, abs=3e-7)
    with pytest.raises(FileExistsError):
        main(_args(base, critics, source, root, 1))
    # Completed files cannot be silently edited and re-rendered.
    path = root / "actor_summary/segments.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        render(root, bootstrap=20)
    # Worker duplicate rows cannot be promoted into complete results.
    directory = root / "actor_summary/measure_0"
    path = directory / "segments.jsonl"
    path.write_text(path.read_text() * 2)
    worker_manifest = json.loads((directory / "manifest.json").read_text())
    worker_manifest["segments_sha256"] = data_sha256(path)
    (directory / "manifest.json").write_text(json.dumps(worker_manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        merge_results(
            root / "actor_summary",
            manifest["selection"],
            "actor_summary",
            manifest["actor_checkpoint"],
            Namespace(**manifest["arguments"]),
        )


def test_identical_critic_checkpoint_is_rejected(experiment, tmp_path):
    _, base, critics, source, _ = experiment
    with pytest.raises(ValueError, match="independently pretrained"):
        main(_args(base, [critics[0], critics[0]], source, tmp_path / "bad", 1))


def test_summary_balances_episodes_then_problems_and_excludes_only_undefined():
    rows = [dict(episode_id="a", problem_id="p", x=0)] * 8
    rows += [dict(episode_id="b", problem_id="p", x=10), dict(episode_id="c", problem_id="q", x=20)]
    rows += [dict(episode_id="d", problem_id="q", x=None)]
    result = summarize(rows, "x", seed=1, bootstrap=20)
    assert result["mean"] == 12.5 and result["undefined"] == 1 and result["problems"] == 2
    assert result["trajectories"] == 3 and result["ci95"] is not None


@pytest.mark.parametrize("flag", ["--workers", "--trajectories", "--bootstrap", "--max-tokens", "--head-chunk-size"])
def test_invalid_flags_fail_before_loading(flag):
    args = _args(Path("missing"), [Path("a"), Path("b")], Path("missing"), Path("missing"), 1)
    with pytest.raises(SystemExit):
        main(args + [flag, "0"])
