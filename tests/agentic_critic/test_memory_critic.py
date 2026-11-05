"""CPU checks for the actual independent critic reader and FP32 CPU optimizer."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from agentic_critic.cpu_sharded_adam import CPUShardedAdam
from agentic_critic.critic_input import Segment
from agentic_critic.critic_modes import ValueHead, episode_values
from agentic_critic.memory_critic import MemoryCritic, MemoryEpisode, flatten_episode_values, train_memory_minibatches


@pytest.fixture
def critic():
    return _make_critic()


def _make_critic():
    from transformers import AutoModel
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(9)
    config = Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        vocab_size=128,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    model = AutoModel.from_config(config).float().eval()
    head = ValueHead(32)
    torch.nn.init.normal_(head.linear.weight, std=0.1)
    return MemoryCritic(model, head).eval()


def _episode():
    return MemoryEpisode(
        [Segment([1, 2, 3, 4, 5, 6], 4, [1, 0, 1, 1]), Segment([7, 8, 9, 10, 11], 3, [1, 0, 1])],
        reward=0.7,
    )


def test_full_detach_values_equal_full_and_value_positions_match(critic):
    episode = _episode()
    actual = critic.values(episode, mode="full_detach", device="cpu")
    with torch.no_grad():
        full = episode_values(critic.backbone, critic.head, episode.segments, "full", "cpu")
    for value, expected in zip(actual, full, strict=True):
        torch.testing.assert_close(value, expected, atol=1e-5, rtol=1e-5)
    flattened = flatten_episode_values(episode, actual)
    assert len(flattened) == 9
    torch.testing.assert_close(flattened[:4], actual[0])
    torch.testing.assert_close(flattened[6:9], actual[1])
    assert flattened[4:6].tolist() == [0.0, 0.0]


def test_actor_summary_resets_memory_and_full_detach_retains_it(critic):
    episode = _episode()
    changed = MemoryEpisode([Segment([60, 61, 62, 63], 2, [1, 1]), episode.segments[1]], episode.reward)
    for mode in ("full_detach", "actor-summary"):
        first = critic.values(episode, mode=mode, device="cpu")[1]
        second = critic.values(changed, mode=mode, device="cpu")[1]
        if mode == "actor-summary":
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        else:
            assert not torch.allclose(first, second)


def test_only_current_segment_receives_gradient_from_its_loss(critic):
    episode = _episode()
    episode.segments[0].loss_mask = [0] * episode.segments[0].response_length
    old = critic.values(episode, mode="full_detach", device="cpu")
    captured = []

    def keep(module, args, output):
        if output.requires_grad:
            output.retain_grad()
            captured.append(output)

    hook = critic.backbone.get_input_embeddings().register_forward_hook(keep)
    try:
        critic.backward_episode(
            episode,
            old,
            mode="full_detach",
            device="cpu",
            value_clip=0.2,
            loss_coef=0.5,
            normalizer=2,
            checkpoint=False,
        )
    finally:
        hook.remove()
    assert captured[0].grad is None or captured[0].grad.abs().sum() == 0
    assert captured[-1].grad.abs().sum() > 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_value_loss_matches_direct_regression_and_checkpointing(critic, checkpoint):
    episode = _episode()
    old = critic.values(episode, mode="full_detach", device="cpu")
    count = sum(sum(segment.loss_mask) for segment in episode.segments)
    expected = sum(
        0.5 * (value[torch.tensor(segment.loss_mask, dtype=torch.bool)] - episode.reward).square().sum()
        for value, segment in zip(old, episode.segments, strict=True)
    )
    actual = critic.backward_episode(
        episode,
        old,
        mode="full_detach",
        device="cpu",
        value_clip=0.2,
        loss_coef=0.5,
        normalizer=count,
        checkpoint=checkpoint,
    )
    assert actual == pytest.approx(float(expected), rel=1e-5)
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None)


def _optimizer(parameters):
    return CPUShardedAdam(parameters, lr=0.03, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, chunk_elements=3)


def test_cpu_master_retains_precision_and_optimizer_resume_matches():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 0.5, -0.25], dtype=torch.bfloat16))
    optimizer = _optimizer([parameter])
    reference = torch.nn.Parameter(parameter.float().detach().clone())
    baseline = torch.optim.AdamW([reference], lr=0.03, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    for _ in range(2):
        gradient = torch.tensor([0.2, -0.3, 0.4], dtype=torch.bfloat16)
        parameter.grad, reference.grad = gradient.clone(), gradient.float()
        norm = torch.nn.utils.clip_grad_norm_([reference], 0.25)
        baseline.step()
        actual_norm = optimizer.step(clip_grad=0.25)
        assert actual_norm == pytest.approx(float(norm), rel=1e-5)
        torch.testing.assert_close(parameter, reference.bfloat16(), atol=0, rtol=0)
        restored = _optimizer([parameter])
        restored.load_state_dict(deepcopy(optimizer.state_dict()))
        optimizer = restored
    assert optimizer.masters[0].dtype == torch.float32
    torch.testing.assert_close(optimizer.masters[0], reference)


def test_weights_only_load_refreshes_master_before_first_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    optimizer = _optimizer([parameter])
    with torch.no_grad():
        parameter.copy_(torch.tensor([7.0, 8.0], dtype=torch.bfloat16))
    optimizer.reset_masters_from_model()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step(clip_grad=1)
    torch.testing.assert_close(parameter.float(), torch.tensor([7.0, 8.0]), atol=0.01, rtol=0)
    assert optimizer.masters[0].tolist() == pytest.approx([7 * (1 - 0.03 * 0.01), 8 * (1 - 0.03 * 0.01)])


def _distributed_adam_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.bfloat16)),
        ]
        optimizer = _optimizer(parameters)
        references = [torch.nn.Parameter(p.float().detach().clone()) for p in parameters]
        baseline = torch.optim.AdamW(references, lr=0.03, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        for parameter, reference in zip(parameters, references, strict=True):
            parameter.grad = torch.full_like(parameter, (rank + 1) * 0.25)
            reference.grad = torch.full_like(
                reference, 0.75
            )  # sum of rank contributions, globally normalized upstream
        expected_norm = torch.nn.utils.clip_grad_norm_(references, 0.25)
        baseline.step()
        actual_norm = optimizer.step(clip_grad=0.25)
        assert actual_norm == pytest.approx(float(expected_norm), rel=1e-5)
        assert len(optimizer.masters) == 1
        for parameter, reference in zip(parameters, references, strict=True):
            torch.testing.assert_close(parameter, reference.bfloat16(), atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_fp32_reduction_global_clipping_and_owned_updates(tmp_path):
    mp.spawn(_distributed_adam_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


@pytest.mark.parametrize("weights_only", [False, True])
def test_native_checkpoint_restores_model_masters_and_step(tmp_path, monkeypatch, weights_only):
    from agentic_critic.miles_memory_critic import HFMemoryCriticTrainRayActor

    worker = object.__new__(HFMemoryCriticTrainRayActor)
    worker.model = torch.nn.Linear(3, 1, bias=False).to(torch.bfloat16)
    worker.optimizer = _optimizer(worker.model.parameters())
    worker.optimizer_steps = 0
    with torch.no_grad():
        worker.model.weight.fill_(7)
    worker.optimizer.reset_masters_from_model()
    checkpoint = tmp_path / "memory_critic_0000004"
    checkpoint.mkdir()
    torch.save(
        dict(
            model=worker.model.state_dict(),
            optimizer_steps=5,
            rollout_id=4,
            mode="full_detach",
            world_size=1,
            actor_tp_size=1,
        ),
        checkpoint / "model.pt",
    )
    torch.save(dict(optimizer=worker.optimizer.state_dict()), checkpoint / "optimizer_rank_00000.pt")
    (tmp_path / "memory_critic_latest.json").write_text(json.dumps(dict(directory=checkpoint.name)))
    with torch.no_grad():
        worker.model.weight.fill_(1)
    worker.optimizer.reset_masters_from_model()
    worker.args = SimpleNamespace(
        load=str(tmp_path),
        hf_checkpoint="unused",
        critic_context_mode="full_detach",
        tensor_model_parallel_size=1,
        finetune=False,
        no_load_optim=weights_only,
        no_load_rng=True,
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    start = worker._restore()
    assert start == (0 if weights_only else 5)
    assert worker.optimizer_steps == (0 if weights_only else 5)
    assert worker.model.weight.flatten().tolist() == [7.0] * 3
    assert worker.optimizer.masters[0].flatten().tolist() == [7.0] * 3
    worker.model.weight.grad = torch.zeros_like(worker.model.weight)
    worker.optimizer.step(clip_grad=1)
    assert worker.model.weight.flatten().tolist() == [7.0] * 3  # only sub-BF16-resolution weight decay


def test_hf_backbone_load_preserves_checkpoint_fp32_a_log(critic, tmp_path):
    from agentic_critic.hf_memory_backbone import load_memory_backbone
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    original = Qwen3_5ForCausalLM(critic.backbone.config).float()
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if name.endswith(".A_log"):
                parameter.fill_(0.1234567)
    original.save_pretrained(tmp_path)
    backbone = load_memory_backbone(str(tmp_path))
    a_logs = [(name, p) for name, p in backbone.named_parameters() if name.endswith(".A_log")]
    assert a_logs
    for name, parameter in backbone.named_parameters():
        if name.endswith(".A_log"):
            assert parameter.dtype == torch.float32
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.1234567), atol=0, rtol=0)
        else:
            assert parameter.dtype == torch.bfloat16, name
    backbone.to("cpu")  # Device-only offload preserves the mixed parameter dtypes.
    assert all(p.dtype == torch.float32 for _, p in a_logs)


def _training_episodes():
    rewards = [0.0, 0.05, 0.0, 0.5, 0.2, 0.8, 0.5, 1.0]
    return [
        MemoryEpisode(
            [
                Segment([1, 2, 3 + i, 20 + i, 40 + i], 3, [1, 0, 1]),
                Segment([60 + i, 70 + i, 80 + i, 90 + i], 2, [1, i % 2]),
            ],
            reward,
        )
        for i, reward in enumerate(rewards)
    ]


def _inverse_std_weights(episodes):
    from miles.ray.rollout.train_data_conversion import _compute_critic_loss_weights

    return _compute_critic_loss_weights(
        rewards=[ep.reward for ep in episodes],
        loss_masks=[[m for seg in ep.segments for m in seg.loss_mask] for ep in episodes],
        prompt_groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
        std_floor=0.075,
    )


def _reference_loss(model, episode, old, *, mode, weight, count):
    values = episode_values(model.backbone, model.head, episode.segments, model._reader_mode(mode), "cpu")
    total = values[0].new_zeros(())
    for segment, value, previous in zip(episode.segments, values, old, strict=True):
        mask = torch.tensor(segment.loss_mask, dtype=torch.bool)
        clipped = previous + (value - previous).clamp(-0.2, 0.2)
        errors = torch.maximum((value - episode.reward).square(), (clipped - episode.reward).square())
        total += 0.5 * weight * errors[mask].sum() / count
    return total


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_weighted_clipped_loss_uses_fixed_old_values_and_unweighted_count(critic, mode):
    episode = _episode()
    old = critic.values(episode, mode=mode, device="cpu")
    with torch.no_grad():
        critic.head.linear.bias.add_(0.35)  # Exercise clipping after an earlier minibatch update.
    reference = deepcopy(critic)
    expected = _reference_loss(reference, episode, old, mode=mode, weight=2.3, count=5)
    expected.backward()
    actual = critic.backward_episode(
        episode,
        old,
        mode=mode,
        device="cpu",
        value_clip=0.2,
        loss_coef=0.5,
        normalizer=5,
        loss_weight=2.3,
        checkpoint=True,
    )
    assert actual / 5 == pytest.approx(float(expected.detach()), rel=1e-5)
    for parameter, baseline in zip(critic.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, baseline.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_four_minibatches_match_fresh_memory_reference_and_return_frozen_values(critic, mode):
    episodes = _training_episodes()
    weights = _inverse_std_weights(episodes)
    assert max(weights) > 2 * min(weights)
    schedule = [[4, 1], [6, 3], [0, 7], [2, 5]]
    reference = deepcopy(critic)
    old = [reference.values(episode, mode=mode, device="cpu") for episode in episodes]
    baseline = _optimizer(reference.parameters())
    expected_loss, expected_norms = 0.0, []
    for step, indices in enumerate(schedule, 1):
        baseline.zero_grad()
        count = sum(sum(seg.loss_mask) for i in indices for seg in episodes[i].segments)
        loss = sum(
            _reference_loss(reference, episodes[i], old[i], mode=mode, weight=weights[i], count=count) for i in indices
        )
        expected_loss += float(loss.detach()) * count
        loss.backward()
        for group in baseline.param_groups:
            group["lr"] = 0.03 * min(1.0, step / 3)
        expected_norms.append(baseline.step(clip_grad=0.25))
    result = train_memory_minibatches(
        critic,
        _optimizer(critic.parameters()),
        episodes,
        minibatch_indices=schedule,
        loss_weights=weights,
        mode=mode,
        device="cpu",
        value_clip=0.2,
        loss_coef=0.5,
        learning_rate=0.03,
        warmup_steps=3,
        optimizer_steps=0,
        clip_grad=0.25,
    )
    assert result.optimizer_steps == 4
    assert result.learning_rates == pytest.approx([0.01, 0.02, 0.03, 0.03])
    assert result.token_count == sum(sum(seg.loss_mask) for ep in episodes for seg in ep.segments)
    assert result.loss_sum == pytest.approx(expected_loss, rel=2e-4)
    assert result.grad_norms == pytest.approx(expected_norms, rel=3e-4)
    for actual, expected in zip(critic.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=3e-3)
    for actual, expected in zip(result.values, old, strict=True):
        for value, initial in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, initial, atol=0, rtol=0)
            assert not value.requires_grad
    changed = critic.values(episodes[-1], mode=mode, device="cpu")
    assert not torch.allclose(changed[-1], result.values[-1][-1])


@pytest.mark.parametrize(
    "schedule,weights,match",
    [
        ([[0], [0]], [1.0], "every local episode"),
        ([], [1.0], "every local episode"),
        ([[0]], [], "one finite"),
        ([[0]], [float("nan")], "one finite"),
        ([[0]], [-1.0], "one finite"),
    ],
)
def test_minibatch_schedule_and_weights_fail_before_optimizer(critic, schedule, weights, match):
    optimizer = _optimizer(critic.parameters())
    before = deepcopy(critic.state_dict())
    with pytest.raises(ValueError, match=match):
        train_memory_minibatches(
            critic,
            optimizer,
            [_episode()],
            minibatch_indices=schedule,
            loss_weights=weights,
            mode="full_detach",
            device="cpu",
            value_clip=0.2,
            loss_coef=0.5,
            learning_rate=0.03,
            warmup_steps=0,
            optimizer_steps=0,
            clip_grad=0.25,
        )
    assert not optimizer.optimizer.state
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def _distributed_minibatches_worker(rank, rendezvous, mode, duplicate_shards):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        model = _make_critic()
        reference = deepcopy(model)
        episodes = _training_episodes()
        weights = _inverse_std_weights(episodes)
        old = [reference.values(ep, mode=mode, device="cpu") for ep in episodes]
        baseline = torch.optim.AdamW(reference.parameters(), lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        expected_norms = []
        for start in range(0, len(episodes), 2):
            indices = [start, start + 1]
            count = sum(sum(seg.loss_mask) for i in indices for seg in episodes[i].segments)
            baseline.zero_grad()
            loss = sum(
                _reference_loss(reference, episodes[i], old[i], mode=mode, weight=weights[i], count=count)
                for i in indices
            )
            loss.backward()
            expected_norms.append(float(torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)))
            baseline.step()
        # One case has unequal local token counts; the other emulates the
        # duplicate shards read by actor TP siblings. Neither may change the
        # global unweighted-token mean or the optimizer update.
        local = list(range(len(episodes))) if duplicate_shards else list(range(rank, len(episodes), 2))
        schedule = (
            [[i, i + 1] for i in range(0, len(local), 2)] if duplicate_shards else [[i] for i in range(len(local))]
        )
        result = train_memory_minibatches(
            model,
            CPUShardedAdam(
                model.parameters(), lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, chunk_elements=4096
            ),
            [episodes[i] for i in local],
            minibatch_indices=schedule,
            loss_weights=[weights[i] for i in local],
            mode=mode,
            device="cpu",
            value_clip=0.2,
            loss_coef=0.5,
            learning_rate=0.003,
            warmup_steps=0,
            optimizer_steps=0,
            clip_grad=0.25,
        )
        assert result.optimizer_steps == 4
        assert result.grad_norms == pytest.approx(expected_norms, rel=4e-4), (result.grad_norms, expected_norms)
        expected_count = sum(sum(seg.loss_mask) for ep in episodes for seg in ep.segments)
        assert result.token_count == expected_count * (2 if duplicate_shards else 1)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=3e-3)
        for actual, i in zip(result.values, local, strict=True):
            for value, initial in zip(actual, old[i], strict=True):
                torch.testing.assert_close(value, initial, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("duplicate_shards", [False, True])
def test_two_rank_four_steps_normalize_weighted_loss_and_duplicate_shards(tmp_path, mode, duplicate_shards):
    mp.spawn(
        _distributed_minibatches_worker,
        args=(str(tmp_path / "gloo_minibatches"), mode, duplicate_shards),
        nprocs=2,
        join=True,
    )
