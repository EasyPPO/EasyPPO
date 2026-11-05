"""Terminal rollout transport, critic GAE and the official score-centering loss."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from agentic_critic.terminal_env import Completion, TerminalTrace
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.backends.training_utils.loss.hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns, loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.utils.arguments import validate_async_off_policy_correction
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample


@pytest.fixture
def local_dp_shard(monkeypatch):
    """Exercise one shard's CPU math with the scaling of a two-rank DP layout."""
    singleton = GroupInfo(rank=0, size=1, group=None)
    dp = GroupInfo(rank=1, size=2, group=None)
    monkeypatch.setattr(
        parallel,
        "_parallel_state",
        ParallelState(
            intra_dp=dp,
            intra_dp_cp=dp,
            **{name: singleton for name in ("cp", "tp", "pp", "ep", "etp", "indep_dp")},
        ),
    )


def _terminal_samples():
    trace = TerminalTrace()
    for prompt, tokens, probabilities in (
        ([0, 1], [2, 3], [0.5, 0.25]),
        ([0, 1, 2, 3, 6], [4, 5], [0.3, 0.2]),
    ):
        trace.record(
            prompt,
            Completion(
                tokens=tokens,
                logprobs=np.log(probabilities).tolist(),
                text="terminal action",
                topk_token_ids=np.asarray([tokens + [-1]] * 2, dtype=np.int32),
                topk_log_probs=np.asarray([np.log(probabilities).tolist() + [-np.inf]] * 2, dtype=np.float32),
            ),
            kind="action",
        )
    fields = trace.sample_fields()
    assert len(fields) == 1 and fields[0]["loss_mask"] == [1, 1, 0, 1, 1]
    samples = []
    for index, reward in enumerate((1.7, 2.5)):
        sample = Sample(
            index=index, rollout_id=index, group_index=index, reward=reward, status=Sample.Status.COMPLETED
        )
        for name, value in deepcopy(fields[0]).items():
            setattr(sample, name, value)
        samples.append(sample)
    return samples


def _ppo_args(**overrides):
    return _args(
        **(
            dict(
                advantage_estimator="ppo",
                use_critic=True,
                use_rollout_logprobs=True,
                use_tis=False,
                keep_old_actor=False,
                skip_actor_forward_only=False,
                normalize_advantages=False,
                # Each task has one trajectory. PPO must retain its raw reward
                # even when the GRPO normalization default remains enabled.
                rewards_normalization=True,
                n_samples_per_prompt=1,
                rollout_batch_size=2,
                global_batch_size=2,
                gamma=1.0,
                lambd=1.0,
                kl_coef=0,
                rollout_temperature=1.0,
                ci_test=True,
                eps_clip=0.01,
                eps_clip_high=0.02,
            )
            | overrides
        )
    )


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_terminal_critic_gae_to_official_loss(local_dp_shard, mode, per_token, recompute):
    args = _ppo_args(
        score_centering_is=mode,
        calculate_per_token_loss=per_token,
        recompute_loss_function=recompute,
    )
    validate_score_centering_args(args)
    validate_async_off_policy_correction(args)
    samples = _terminal_samples()
    data = convert_samples_to_train_data(args, samples, {}, None, None)
    assert data["rewards"] == [1.7, 2.5]
    batch = split_train_data_by_dp_raw(args, data, dp_size=2)[1]
    assert batch["sample_indices"] == [1] and batch["rewards"] == [2.5]
    assert not {"log_probs", "old_topk_log_probs", "old_topk_token_ids"}.intersection(batch)
    np.testing.assert_array_equal(batch["rollout_topk_token_ids"][0], samples[1].rollout_topk_token_ids)
    assert (batch["rollout_topk_token_ids"][0][2] == -1).all()
    assert np.isneginf(batch["rollout_topk_log_probs"][0][2]).all()

    # These are the response-aligned values shipped by the independent critic.
    # A deliberately invalid observation value must be excluded from GAE.
    values = torch.tensor([0.25, 3.0, torch.nan, -1.0, 0.5], requires_grad=True)
    sampled_q = torch.tensor(batch["rollout_log_probs"][0], requires_grad=True)
    with torch.no_grad():
        sampled_q[2] = torch.nan
    head_q = torch.tensor(batch["rollout_topk_log_probs"][0], requires_grad=True)
    batch.update(
        total_lengths=[len(samples[1].tokens)],
        unconcat_tokens=[torch.tensor(samples[1].tokens)],
        loss_masks=[torch.tensor(batch["loss_masks"][0])],
        rollout_mask_sums=torch.tensor(batch["rollout_mask_sums"]),
        rollout_log_probs=[sampled_q],
        rollout_topk_log_probs=[head_q],
        values=[values],
    )
    compute_advantages_and_returns(args, batch)
    expected_advantages = torch.tensor([2.25, -0.5, 0.0, 3.5, 2.0])
    torch.testing.assert_close(batch["advantages"][0], expected_advantages)
    torch.testing.assert_close(batch["returns"][0], torch.tensor([2.5, 2.5, 0.0, 2.5, 2.5]))
    assert not batch["advantages"][0].requires_grad and not batch["returns"][0].requires_grad
    assert not batch["rollout_log_probs"][0].requires_grad
    assert not {"log_probs", "old_topk_log_probs", "old_topk_token_ids"}.intersection(batch)

    # The official wrapper must also freeze advantages if an upstream caller
    # supplies a tensor carrying a graph, and ignore invalid observation rows.
    advantages = batch["advantages"][0].clone().requires_grad_()
    with torch.no_grad():
        advantages[2] = torch.nan
    batch["advantages"] = [advantages]
    assert get_loss_function(args) is score_centering_loss_function
    logits = torch.randn(
        1, len(samples[1].tokens), 10, generator=torch.Generator().manual_seed(21), requires_grad=True
    )
    actual, normalizer, metrics = loss_function(args, batch, 3, logits, apply_megatron_loss_scaling=True)

    # Reuse the existing estimator as the integration reference. Its independent
    # dense-gradient mathematics is covered by the original official tests.
    reference_logits = logits.detach().clone().requires_grad_()
    logp = reference_logits[0, 1:-1, :8].log_softmax(-1)
    tokens = torch.tensor(samples[1].tokens[2:])
    ids = torch.tensor(batch["rollout_topk_token_ids"][0], dtype=torch.long)
    active = batch["loss_masks"][0].bool()
    token_loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp.gather(-1, tokens[:, None]).squeeze(-1),
            train_head_log_probs=logp.gather(-1, ids.clamp_min(0)),
            rollout_log_probs=torch.where(active, sampled_q.detach(), 0.0),
            rollout_head_log_probs=head_q.detach(),
            head_mask=(ids >= 0) & active[:, None],
            advantages=expected_advantages,
            mode=mode,
            tis_clip=args.score_centering_tis_clip,
            mis_low=args.score_centering_mis_low,
            mis_high=args.score_centering_mis_high,
        )
    )
    # Token reduction returns a sum for Megatron's later token normalization.
    # Sample reduction includes three microbatches * DP2 / global_batch_size2.
    expected = token_loss[active].sum() if per_token else token_loss[active].mean() * 3
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert (logits.grad[0, 3] == 0).all()  # The tool observation's prediction row.
    assert (logits.grad[..., 8:] == 0).all()  # Padded vocabulary entries.
    assert values.grad is None and advantages.grad is None and sampled_q.grad is None and head_q.grad is None
    assert normalizer == (4 if per_token else 1)
    assert {"pg_loss", "sc_importance_weight", "train_rollout_kl"}.issubset(metrics["keys"])
    assert torch.isfinite(metrics["values"]).all()


@pytest.mark.parametrize("estimator", ["grpo", "ppo"])
@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_supported_startup_and_explicit_async_sampler_choice(estimator, mode):
    args = _ppo_args(advantage_estimator=estimator, use_critic=estimator == "ppo", score_centering_is=mode)
    validate_score_centering_args(args)
    validate_async_off_policy_correction(args)


@pytest.mark.parametrize("estimator", ["gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_other_advantage_estimators_are_rejected(estimator):
    with pytest.raises(ValueError, match="grpo or ppo"):
        validate_score_centering_args(_ppo_args(advantage_estimator=estimator))


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("override", [{"use_tis": True}, {"custom_tis_function_path": "custom.weight"}])
def test_official_is_cannot_be_stacked_with_generic_tis(mode, override):
    with pytest.raises(ValueError, match="incompatible"):
        validate_score_centering_args(_ppo_args(score_centering_is=mode, **override))


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_sc_is_setting_does_not_replace_explicit_async_behavior_choice(mode):
    args = _ppo_args(score_centering_is=mode, use_rollout_logprobs=False)
    validate_score_centering_args(args)
    with pytest.raises(AssertionError, match="explicit behavior-policy correction"):
        validate_async_off_policy_correction(args)


def test_clipped_ppo_baseline_retains_old_actor_choice():
    args = _ppo_args(
        loss_type="policy_loss", rollout_top_logprobs_num=0, use_rollout_logprobs=False, keep_old_actor=True
    )
    validate_score_centering_args(args)
    validate_async_off_policy_correction(args)
