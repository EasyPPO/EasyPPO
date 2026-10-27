"""Independent gradient oracles, alignment, determinism, and frozen-state checks."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import Qwen3_5TextConfig

from agentic_critic_pipeline.gradient_models import (
    actor_loss_function,
    checkpoint_layers,
    critic_values,
    token_positions,
)
from agentic_critic_pipeline.gradient_probe import measure_gradients, probe_mode, select_windows, summarize


@pytest.fixture(params=["llama", "qwen35"])
def model(request):
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
        attention_dropout=0.2,
    )
    if request.param == "llama":
        return LlamaForCausalLM(config).float().eval()
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
    return AutoModelForCausalLM.from_config(config).float().eval()


def _segment():
    return dict(
        input_ids=[1, 7, 2, 6, 4, 9, 3, 5, 8, 2, 11, 4], response_start=3, loss_mask=[1, 1, 0, 1, 1, 0, 1, 1, 1]
    )


def _flat_grad(loss, model):
    gradients = torch.autograd.grad(loss, model.parameters(), allow_unused=True, retain_graph=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten().double()
            for p, g in zip(model.parameters(), gradients, strict=True)
        ]
    )


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_matches_independent_per_token_and_full_backward(model, checkpointed, chunk_size):
    segment = _segment()
    advantages = torch.tensor([0.3, -0.5, 1.2, 0.7, -0.8, 0.4, 0.1])
    windows = [[0, 1], [2, 3], [5, 6]]
    # Independent ordinary HF causal-LM forward with all logits, shifted labels.
    ids = torch.tensor([segment["input_ids"]])
    positions = [2, 3, 5, 6, 8, 9, 10]
    logits = model(ids, use_cache=False).logits[0, positions]
    nll = F.cross_entropy(logits, ids[0, torch.tensor(positions) + 1], reduction="none")
    losses = nll * advantages
    per_token = [_flat_grad(loss, model) / len(losses) for loss in losses]
    full = _flat_grad(losses.mean(), model)
    assert torch.allclose(torch.stack(per_token).sum(0), full, atol=2e-7, rtol=1e-5)
    before = copy.deepcopy(model.state_dict())
    with probe_mode(model), checkpoint_layers(model, checkpointed):
        callback = actor_loss_function(model, segment, advantages, head_chunk_size=chunk_size)
        result = measure_gradients(callback, model.parameters(), windows)
    assert result["full_norm"] == pytest.approx(float(full.norm()), rel=2e-6)
    for window, row in zip(windows, [result["target"]] + result["controls"], strict=True):
        g = torch.stack([per_token[i] for i in window]).sum(0)
        remaining = [i for i in range(len(losses)) if i not in window]
        rest = _flat_grad(losses[remaining].sum() / len(losses), model)
        assert row["norm"] == pytest.approx(float(g.norm()), rel=3e-6)
        assert row["projection_on_rest"] == pytest.approx(float(g @ rest / (rest @ rest)), abs=2e-6)
        assert row["cosine_with_rest"] == pytest.approx(float(g @ rest / (g.norm() * rest.norm())), abs=2e-6)
        assert row["rest_norm"] == pytest.approx(float(rest.norm()), rel=3e-6)
        assert row["norm_over_rest"] == pytest.approx(float(g.norm() / rest.norm()), rel=3e-6)
    assert result["ratio_to_control_mean"] == pytest.approx(
        result["target"]["norm"] / (sum(row["norm"] for row in result["controls"]) / 2)
    )
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert all("forward" not in layer.__dict__ for layer in model.model.layers)
    assert result["parameter_count"] == sum(p.numel() for p in model.parameters())


def test_signed_projection_and_cancellation_are_not_norm_shares():
    p = torch.nn.Parameter(torch.tensor([0.2, 0.4], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.zeros(3))
    losses = torch.stack([p[0] * 3, -p[0] * 2, p[1] * 4])
    result = measure_gradients(losses, [p, unused], [[0], [1], [2]])
    assert result["controls"][0]["projection_on_rest"] < 0
    assert result["parameter_count"] == 5 and result["used_parameter_count"] == 2
    assert sum(row["norm"] for row in [result["target"]] + result["controls"]) > result["full_norm"]


def test_zero_gradients_are_explicit_not_epsilon_inflated_ratios():
    p = torch.nn.Parameter(torch.ones(2))
    result = measure_gradients(p * 0, [p], [[0], [1]])
    assert result["ratio_to_control_mean"] is None
    assert result["target"]["norm"] == 0
    assert result["target"]["projection_on_rest"] is None
    assert result["target"]["cosine_with_rest"] is None


def test_cancelling_nonzero_windows_have_zero_full_gradient():
    p = torch.nn.Parameter(torch.tensor(1.0))
    result = measure_gradients(torch.stack([p, -p]), [p], [[0], [1]])
    assert result["full_norm"] == 0 and result["target"]["norm"] == 0.5
    assert result["ratio_to_control_mean"] == 1
    assert result["target"]["norm_over_full"] is None
    assert result["target"]["rest_norm"] == 0.5
    assert result["target"]["cosine_with_rest"] == -1


def test_self_alignment_does_not_hide_conflict_with_remaining_tokens():
    p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    result = measure_gradients(torch.stack([10 * p, -9 * p]), [p], [[0], [1]])
    assert result["full_norm"] == 0.5  # Full and target point in the SAME direction.
    assert result["target"]["cosine_with_rest"] == -1
    assert result["target"]["projection_on_rest"] == pytest.approx(-10 / 9)


def test_zero_remaining_gradient_and_window_unused_parameters():
    p = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

    def losses(indices):
        if indices is None:
            return torch.stack([p, q])
        return torch.stack([p if i == 0 else q for i in indices])

    result = measure_gradients(losses, [p, q], [[0], [1]])
    assert result["target"]["rest_norm"] == 0.5  # q is unused in this window, used in full.
    assert result["target"]["cosine_with_rest"] == 0
    result = measure_gradients(torch.stack([p, p * 0]), [p], [[0], [1]])
    assert result["target"]["rest_norm"] == 0
    assert result["target"]["cosine_with_rest"] is None
    assert result["target"]["projection_on_rest"] is None


def test_rest_subtraction_is_stable_when_window_dominates():
    p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    result = measure_gradients(torch.stack([1e12 * p, p]), [p], [[0], [1]])
    assert result["target"]["rest_norm"] == 0.5
    assert result["target"]["cosine_with_rest"] == 1


def test_windows_are_fixed_size_exclude_target_and_are_reproducible():
    windows, skip = select_windows(300, controls=12, seed="episode:1")
    assert skip is None and windows == select_windows(300, controls=12, seed="episode:1")[0]
    assert windows[0] == list(range(64))
    assert len({window[0] for window in windows[1:]}) == 12
    assert all(len(w) == 64 and w == list(range(w[0], w[0] + 64)) for w in windows)
    assert all(not set(windows[0]) & set(w) for w in windows[1:])
    assert select_windows(63)[1] == "short_target"
    assert select_windows(127)[1] == "no_control_window"
    assert len(select_windows(128)[0]) == 2
    with pytest.raises(ValueError):
        select_windows(128, size=0)


def test_probe_restores_flags_existing_gradients_and_checkpoint_wrappers_on_error(model):
    model.train()
    model.model.layers[0].eval()
    next(model.parameters()).requires_grad_(False)
    flags = [(m, m.training) for m in model.modules()]
    params = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError), probe_mode(model), checkpoint_layers(model, True):
        assert not model.training and all(p.requires_grad for p in model.parameters())
        raise RuntimeError("injected")
    assert all(m.training == flag for m, flag in flags)
    assert all(p.requires_grad == flag and torch.equal(p.grad, torch.ones_like(p)) for p, flag in params)
    assert all("forward" not in layer.__dict__ for layer in model.model.layers)


@pytest.mark.parametrize("mode", ["actor_summary", "full_history"])
def test_critic_alignment_matches_independent_t_minus_one_forward(model, mode):
    torch.manual_seed(7)
    head = torch.nn.Linear(16, 1)
    segment = _segment()
    episode = dict(
        episode_id="ep",
        problem_id="p",
        split="validation",
        reward=0.5,
        segments=[segment, dict(segment, input_ids=list(reversed(segment["input_ids"])))],
    )
    actual = critic_values(model, lambda h: head(h).squeeze(-1), episode, mode=mode, max_tokens=100)
    offset = 0
    with torch.no_grad():
        if mode == "full_history":
            tokens = [token for s in episode["segments"] for token in s["input_ids"]]
            hidden = model.model(torch.tensor([tokens]), use_cache=False).last_hidden_state[0]
        for s, values in zip(episode["segments"], actual, strict=True):
            if mode == "actor_summary":
                hidden = model.model(torch.tensor([s["input_ids"]]), use_cache=False).last_hidden_state[0]
            expected = head(hidden[[offset + p for p in token_positions(s)]]).squeeze(-1)
            assert torch.allclose(torch.tensor(values), expected)
            if mode == "full_history":
                offset += len(s["input_ids"])
    assert all(p.grad is None for p in list(model.parameters()) + list(head.parameters()))
    with pytest.raises(ValueError, match="truncation"):
        critic_values(model, head, episode, mode=mode, max_tokens=3)


def test_problem_cluster_summary_and_zero_ratio_reporting():
    rows = [dict(problem_id="a", ratio_to_control_mean=2.0)] * 4
    rows += [
        dict(problem_id="b", ratio_to_control_mean=8.0),
        dict(problem_id="c", ratio_to_control_mean=0.0),
        dict(skip="short_target"),
    ]
    result = summarize(rows, bootstrap=100, seed=7)
    assert result == summarize(rows, bootstrap=100, seed=7)
    assert result["problem_balanced_geometric_ratio"] == pytest.approx(4.0)
    assert result["problems"] == 2 and result["measured_boundaries"] == 6
    assert result["excluded_zero_or_undefined_ratios"] == 1
    assert result["problem_cluster_bootstrap_95ci"] == pytest.approx([2.0, 8.0])
    assert summarize([], bootstrap=2)["problem_balanced_geometric_ratio"] is None


def test_fixed_policy_surrogate_matches_ppo_gradient_at_ratio_one():
    logits = torch.tensor([[0.4, -0.8], [0.2, 0.1]], requires_grad=True)
    logp = logits.log_softmax(-1)[:, 0]
    advantages = torch.tensor([-0.5, 0.7])
    ratio = (logp - logp.detach()).exp()
    ppo = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages).mean()
    surrogate = -(logp * advantages).mean()
    actual = torch.autograd.grad(ppo, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(surrogate, logits)[0]
    assert torch.allclose(actual, expected)


def test_head_evaluates_only_selected_window_after_full_reference(model):
    calls = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, inputs: calls.append(inputs[0].shape[0]))
    try:
        callback = actor_loss_function(model, _segment(), torch.ones(7), head_chunk_size=128)
        callback([0, 1]).sum().backward()
    finally:
        handle.remove()
    assert calls and set(calls) == {2}


def test_finite_difference_direction_matches_full_parameter_gradient(model):
    segment = _segment()
    advantages = torch.tensor([0.3, -0.5, 1.2, 0.7, -0.8, 0.4, 0.1])
    model.double()
    loss = actor_loss_function(model, segment, advantages)([0, 1]).sum() / 7
    grads = torch.autograd.grad(loss, model.parameters())
    directions = [torch.randn_like(p) for p in model.parameters()]
    scale = math.sqrt(sum(float(d.square().sum()) for d in directions))
    directions = [d / scale for d in directions]
    analytical = sum(float((g * d).sum()) for g, d in zip(grads, directions, strict=True))
    epsilon = 1e-3  # LM logits intentionally reduce in fp32, as in production.
    values = []
    original = [p.detach().clone() for p in model.parameters()]
    for sign in (1, -1):
        with torch.no_grad():
            for p, initial, d in zip(model.parameters(), original, directions, strict=True):
                p.copy_(initial + sign * epsilon * d)
        values.append(float(actor_loss_function(model, segment, advantages)([0, 1]).sum().detach()) / 7)
    assert (values[0] - values[1]) / (2 * epsilon) == pytest.approx(analytical, abs=2e-5, rel=0.01)
