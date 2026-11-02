"""The bounded-memory episode reader gives the values and the gradients of the plain one-graph forward."""

import pytest
import torch

from agentic_critic.critic_modes import episode_values, value_loss
from agentic_critic.critic_train import read_episode
from tests.agentic_critic.test_critic_modes import KV_UPSTREAM, make_segments
from tests.agentic_critic.test_critic_modes import head, model  # noqa: F401  (fixtures)

SETTINGS = [("shared", None), ("state", None), ("full_detach", None), ("full", None)]
KV_SETTINGS = [("kv", {"ratio": 0.5, "method": "evict", "queries": "hindsight"}), ("kv", {"ratio": 0.5, "method": "am", "queries": "hindsight"}),
               ("kv", {"ratio": 0.5, "method": "am_cv", "queries": "causal"})]  # fmt: skip


def collect(model, head, segs, mode, kv, train, use_checkpoint=True):  # noqa: F811
    got = {}

    def on_values(pairs):
        if train:
            total = sum(((v[torch.tensor(segs[i].loss_mask, dtype=torch.bool)] - 0.3) ** 2).sum() for i, v in pairs)
            total.backward()
        for i, v in pairs:
            got[i] = v.detach().clone()

    model.zero_grad()
    head.zero_grad()
    if train:
        read_episode(model, head, segs, mode, "cpu", on_values, train=True, kv=kv, use_checkpoint=use_checkpoint)
    else:
        with torch.no_grad():
            read_episode(model, head, segs, mode, "cpu", on_values, train=False, kv=kv)
    grads = {n: p.grad.clone() for n, p in list(model.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()] if p.grad is not None}
    return [got[i] for i in range(len(segs))], grads


def reference(model, head, segs, mode, kv):  # noqa: F811
    model.zero_grad()
    head.zero_grad()
    vals = episode_values(model, head, segs, mode, "cpu", kv=kv)
    loss, _ = value_loss(vals, segs, 0.3)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in list(model.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()] if p.grad is not None}
    return [v.detach() for v in vals], grads


def check(model, head, mode, kv, seed):  # noqa: F811
    segs = make_segments(seed, 3)
    ref_vals, ref_grads = reference(model, head, segs, mode, kv)
    for train in (False, True):
        vals, grads = collect(model, head, segs, mode, kv, train)
        for a, b in zip(ref_vals, vals, strict=True):
            assert torch.allclose(a, b, atol=1e-5), (mode, train)
        if train:
            assert set(grads) == set(ref_grads)
            worst = max(float((grads[n] - ref_grads[n]).abs().max() / (ref_grads[n].abs().max() + 1e-12)) for n in grads)
            assert worst < 1e-3, (mode, worst)


@pytest.mark.parametrize("mode,kv", SETTINGS)
@pytest.mark.parametrize("seed", range(2))
def test_values_and_gradients_match_the_plain_forward(model, head, mode, kv, seed):  # noqa: F811
    check(model, head, mode, kv, seed)


@KV_UPSTREAM
@pytest.mark.parametrize("mode,kv", KV_SETTINGS)
def test_values_and_gradients_match_the_plain_forward_with_a_compacted_memory(model, head, mode, kv):  # noqa: F811
    check(model, head, mode, kv, 4)


def test_checkpointing_does_not_change_the_gradients(model, head):  # noqa: F811
    segs = make_segments(1, 3)
    for mode in ("full_detach", "full"):
        _, with_cp = collect(model, head, segs, mode, None, True, use_checkpoint=True)
        _, without = collect(model, head, segs, mode, None, True, use_checkpoint=False)
        assert all(torch.allclose(with_cp[n], without[n], atol=1e-6, rtol=1e-4) for n in without), mode


def test_the_layers_are_left_as_they_were(model, head):  # noqa: F811
    segs = make_segments(0, 2)
    collect(model, head, segs, "full_detach", None, True)
    assert all("forward" not in vars(layer) for layer in model.model.layers)


def test_problem_weights_follow_the_easyppo_formula():
    from agentic_critic.critic_train import problem_weights

    w = problem_weights({"a": [0.0, 0.4], "b": [0.5, 0.5, 0.5], "c": [0.1, 0.9]}, std_floor=0.075)
    raw = {"a": 1 / 0.2, "b": 1 / 0.075, "c": 1 / 0.4}
    mean = sum(raw.values()) / 3
    assert all(abs(w[p] - raw[p] / mean) < 1e-5 for p in raw)
    assert abs(sum(w.values()) / 3 - 1) < 1e-6


def test_clipped_value_loss():
    from agentic_critic.critic_train import clipped_value_loss

    old = torch.tensor([0.0, 0.0, 0.0])
    new = torch.tensor([0.1, 0.5, -0.5])
    loss, binds = clipped_value_loss(new, old, target=1.0, clip=0.2)
    # moving towards the target beyond the clip is not rewarded; moving away is fully penalised
    assert torch.allclose(loss, torch.tensor([0.81, 0.64, 2.25])) and binds.tolist() == [False, True, True]


def test_half_precision_moments_follow_adamw():
    from agentic_critic.critic_train import AdamWHalfMoments

    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(50))
    b = torch.nn.Parameter(a.detach().clone())
    ours = AdamWHalfMoments([a], lr=1e-2, weight_decay=0.01, moment_dtype=torch.float32)
    ref = torch.optim.AdamW([b], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    half_p = torch.nn.Parameter(a.detach().clone())
    half = AdamWHalfMoments([half_p], lr=1e-2, weight_decay=0.01)
    for step in range(20):
        g = torch.randn(50, generator=torch.Generator().manual_seed(step))
        for p, opt in ((a, ours), (b, ref), (half_p, half)):
            p.grad = g.clone()
            opt.step()
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(half_p, b, atol=2e-3) and half.state[half_p]["m"].dtype == torch.bfloat16


@KV_UPSTREAM
def test_plain_memory_reads_take_the_fast_attention_path(model, head):  # noqa: F811
    from agentic_critic import critic_train

    segs = make_segments(2, 3)
    for mode, kv, expect in [("full_detach", None, True), ("kv", {"ratio": 0.5, "method": "evict", "queries": "hindsight"}, True),
                             ("shared", None, False), ("state", None, False)]:  # fmt: skip
        critic_train.MEMORY_ATTENTION_CALLS[0] = 0
        collect(model, head, segs, mode, kv, True)
        assert (critic_train.MEMORY_ATTENTION_CALLS[0] > 0) == expect, (mode, kv)
    assert model.config._attn_implementation != "memory_sdpa"
