"""Critic settings on a tiny random Qwen3.5-style hybrid model (3 linear-attention layers + 1 full attention)."""

import random

import pytest
import torch

from agentic_critic.critic_input import Segment
from agentic_critic.critic_modes import MODES, ValueHead, episode_values, value_loss

transformers = pytest.importorskip("transformers")
VOCAB = 512


@pytest.fixture(scope="module")
def model():
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_5 import Qwen3_5TextConfig
    except ImportError:
        pytest.skip("this transformers version has no Qwen3.5")
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        vocab_size=VOCAB,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).float().eval()


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(1)
    h = ValueHead(64)
    torch.nn.init.normal_(h.linear.weight, std=0.5)  # the zero init would make every setting trivially equal
    return h


def make_segments(seed: int, n: int) -> list[Segment]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(3, 6))]
        resp, mask = [], []
        for _ in range(rng.randint(2, 4)):
            a, o = rng.randint(2, 5), rng.randint(1, 3)
            resp += [rng.randrange(VOCAB) for _ in range(a + o)]
            mask += [1] * a + [0] * o
        out.append(Segment(prompt + resp, len(resp), mask))
    return out


def values(model, head, segments, mode):
    with torch.no_grad():
        return episode_values(model, head, segments, mode, "cpu")


@pytest.mark.parametrize("seed", range(3))
def test_without_compaction_all_settings_agree(model, head, seed):
    segments = make_segments(seed, 1)
    ref = values(model, head, segments, "shared")[0]
    for mode in MODES:
        assert torch.allclose(values(model, head, segments, mode)[0], ref, atol=1e-5), mode


@pytest.mark.parametrize("seed", range(3))
def test_first_segment_is_the_same_in_every_setting(model, head, seed):
    segments = make_segments(seed, 3)
    ref = values(model, head, segments, "shared")[0]
    for mode in MODES:
        assert torch.allclose(values(model, head, segments, mode)[0], ref, atol=1e-5), mode


@pytest.mark.parametrize("seed", range(3))
def test_full_detach_predicts_exactly_what_full_predicts(model, head, seed):
    segments = make_segments(seed, 3)
    for a, b in zip(values(model, head, segments, "full"), values(model, head, segments, "full_detach")):
        assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("seed", range(3))
def test_what_each_setting_can_see_of_an_earlier_segment(model, head, seed):
    """Change only the first segment and see whose predictions on the last segment move."""
    segments = make_segments(seed, 2)
    other = make_segments(seed + 100, 1)[0]
    changed = [other, segments[1]]
    moved = {
        mode: not torch.allclose(values(model, head, segments, mode)[1], values(model, head, changed, mode)[1], atol=1e-5)
        for mode in MODES
    }
    assert moved == {"shared": False, "state": True, "full_detach": True, "full": True, "kv": True}


def test_state_differs_from_full_because_attention_is_segment_local(model, head):
    segments = make_segments(0, 2)
    assert not torch.allclose(values(model, head, segments, "state")[1], values(model, head, segments, "full")[1], atol=1e-4)


def grad_into_first_segment(model, head, segments, mode) -> float:
    """Gradient norm, at the first segment's input embeddings, of the loss on the last segment only."""
    emb = model.get_input_embeddings()
    captured = []

    def keep(module, inputs, output):
        if output.requires_grad:  # the kv setting also runs passes without gradients
            output.retain_grad()
            captured.append(output)

    hook = emb.register_forward_hook(keep)
    try:
        model.zero_grad()
        vals = episode_values(model, head, segments, mode, "cpu")
        loss, _ = value_loss(vals[-1:], segments[-1:], target=1.0)
        loss.backward()
    finally:
        hook.remove()
    first = captured[0]  # full: one call covering every segment; sequential settings: the first segment's call
    g = first.grad
    if g is None:
        return 0.0
    n_first = len(segments[0].tokens)
    return float(g[0, :n_first].norm())


def test_only_full_backpropagates_into_earlier_segments(model, head):
    segments = make_segments(0, 2)
    grads = {mode: grad_into_first_segment(model, head, segments, mode) for mode in MODES}
    assert grads["full"] > 1e-6
    assert all(grads[m] == 0.0 for m in ("shared", "state", "full_detach", "kv"))


def test_value_loss_counts_only_actor_tokens(model, head):
    segments = make_segments(0, 2)
    vals = values(model, head, segments, "shared")
    total, count = value_loss(vals, segments, target=0.5)
    assert count == sum(sum(s.loss_mask) for s in segments)
    manual = sum(float(((v[torch.tensor(s.loss_mask, dtype=torch.bool)] - 0.5) ** 2).sum()) for v, s in zip(vals, segments))
    assert abs(float(total) - manual) < 1e-5 * max(1.0, abs(manual))


KV_UPSTREAM = pytest.mark.skipif(
    not __import__("os").path.isdir(__import__("os").path.join(__import__("agentic_critic.kv_compact", fromlist=["x"]).UPSTREAM, "compaction")),
    reason="upstream compaction repo not available",
)


@KV_UPSTREAM
@pytest.mark.parametrize("queries", ["hindsight", "causal"])
def test_kv_with_every_key_kept_is_full_detach(model, head, queries):
    segs = make_segments(3, 3)
    ref = episode_values(model, head, segs, "full_detach", "cpu")
    got = episode_values(model, head, segs, "kv", "cpu", kv={"ratio": 1.0, "method": "evict", "queries": queries})
    for a, b in zip(ref, got, strict=True):
        assert torch.allclose(a, b, atol=1e-4)


@KV_UPSTREAM
@pytest.mark.parametrize("kv", [
    {"ratio": 0.5, "method": "evict", "queries": "hindsight"},
    {"ratio": 0.5, "method": "am", "queries": "hindsight"},
    {"ratio": 0.5, "method": "am_cv", "queries": "causal"},
])  # fmt: skip
def test_kv_runs_trains_and_differs_from_full_detach_only_after_the_first_segment(model, head, kv):
    segs = make_segments(4, 3)
    ref = episode_values(model, head, segs, "full_detach", "cpu")
    got = episode_values(model, head, segs, "kv", "cpu", kv=kv)
    assert torch.allclose(ref[0], got[0], atol=1e-5)
    assert all(torch.isfinite(v).all() for v in got)
    assert not torch.allclose(ref[2], got[2], atol=1e-4)
    loss, count = value_loss(got, segs, 0.3)
    model.zero_grad()
    (loss / count).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.zero_grad()
