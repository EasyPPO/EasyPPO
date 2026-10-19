"""Numerical parity between the EasyPPO port in Miles and the EasyPPO reference code.

The reference is the verl-based EasyPPO repository (arXiv 2609.36802). Point
``EASYPPO_REPO`` at a checkout; the module is skipped when the checkout or one of
its import-time dependencies (omegaconf, tensordict, transformers) is missing.

Every random batch runs the same rollout through both implementations, in the
order the two trainers use:

    reference (verl)                              port (miles)
    compute_prompt_variance_loss_weights          convert_samples_to_train_data
    compute_gae_advantage_return                  compute_advantages_and_returns
    compute_data_metrics (vf_explained_var)       explained_variance_from_stats
    compute_value_loss, per critic mini-batch     loss_function(loss_type="value_loss")
    filter_overlong_response_kv_batch             drop_truncated_samples_from_actor_loss
    compute_policy_loss_vanilla + kl_penalty      loss_function(loss_type="policy_loss")

Reference functions are the genuine objects. ``verl.trainer.ppo.ray_trainer``
cannot be imported without the ray / vllm stack, so the overlong filter is
compiled from its source file instead of being imported.
"""

import ast
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.rollout import _shuffled_step_indices
from miles.backends.training_utils.loss.objective import (
    compute_advantages_and_returns,
    drop_truncated_samples_from_actor_loss,
    loss_function,
)
from miles.backends.training_utils.metrics.log_utils import (
    compute_explained_variance_stats,
    explained_variance_from_stats,
)
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import Sample
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

EASYPPO_REPO = os.environ.get("EASYPPO_REPO")

# The paper's Frontier-CS setting, where it differs from the AIME config of the reference repo.
RETURN_STD_FLOOR = 0.075
EPS_CLIP = 0.2
EPS_CLIP_C = 3.0
VALUE_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
KL_LOSS_COEF = 0.001
NUM_CRITIC_STEPS = 4

NUM_RANDOM_BATCHES = 64
VOCAB_SIZE = 32
MAX_RESPONSE_LEN = 24
LONG_RESPONSE_LEN = 300

# float32 budgets. The port reorders a handful of float operations relative to the reference
# (chunked GAE prefix sums, E[x^2] - E[x]^2 whitening, clamp(v - old) + old value clipping), so the
# two differ by a few ULP of the compared quantity. Measured maxima over the batches of this module
# (printed by test_report_max_abs_differences): weights 0, returns 6.6e-7, whitened advantages 1.6e-6,
# value loss 1.2e-7, actor loss 6.0e-8, scaled explained variance 3.4e-7. A larger difference is a
# bug to investigate, not a reason to raise a budget.
ATOL_RETURNS = 2e-6
ATOL_ADVANTAGES = 5e-6
ATOL_VALUE_LOSS = 5e-7
ATOL_ACTOR_LOSS = 5e-7
ATOL_EXPLAINED_VARIANCE = 2e-6  # relative to max(1, |1 - EV|), see _scaled_explained_variance_difference


# ---------------------------------------------------------------------------
# Reference (EasyPPO / verl)
# ---------------------------------------------------------------------------


@dataclass
class _KVBatch:
    """Stand-in for transfer_queue.KVBatchMeta: the filter only reads these fields."""

    keys: list[str]
    tags: list[dict]
    partition_id: str = "train"
    fields: Any = None
    extra_info: dict = field(default_factory=dict)


def _compile_functions_from_source(path: Path, names: list[str], namespace: dict) -> dict:
    """Compile the named top-level functions of ``path`` without importing the module."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert sorted(node.name for node in selected) == sorted(names), f"missing reference functions in {path}"
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


# Advantage whitening all-reduces its statistics over the DP group; give it a 1-rank group.
@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("easyppo-parity") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ref():
    if not EASYPPO_REPO or not (Path(EASYPPO_REPO) / "verl").is_dir():
        pytest.skip("set EASYPPO_REPO to an EasyPPO checkout to run the parity test")
    os.environ.setdefault("VERL_USE_EXTERNAL_PLUGINS", "none")
    sys.path.insert(0, EASYPPO_REPO)
    try:
        omegaconf = pytest.importorskip("omegaconf")
        pytest.importorskip("tensordict")
        core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")
        metric_utils = pytest.importorskip("verl.trainer.ppo.metric_utils")
        protocol = pytest.importorskip("verl.protocol")
    finally:
        sys.path.remove(EASYPPO_REPO)

    overlong = _compile_functions_from_source(
        Path(EASYPPO_REPO) / "verl/trainer/ppo/ray_trainer.py",
        ["_overlong_response_metrics", "filter_overlong_response_kv_batch"],
        {"torch": torch, "Any": Any},
    )
    return Namespace(
        core_algos=core_algos,
        compute_data_metrics=metric_utils.compute_data_metrics,
        DataProto=protocol.DataProto,
        OmegaConf=omegaconf.OmegaConf,
        filter_overlong_response_kv_batch=overlong["filter_overlong_response_kv_batch"],
    )


# ---------------------------------------------------------------------------
# Random rollout batches
# ---------------------------------------------------------------------------


@dataclass
class _Rollout:
    """One rollout batch in a framework-neutral layout (per-sample lists)."""

    prompt_ids: list[int]  # prompt group of each sample
    prompt_lens: list[int]
    response_lens: list[int]
    loss_masks: list[torch.Tensor]  # [R_i], 0 marks non-actor tokens
    rewards: list[float]
    tokens: list[torch.Tensor]  # [P_i + R_i]
    old_values: list[torch.Tensor]  # [R_i], critic before the update
    new_values: list[torch.Tensor]  # [R_i], critic being trained
    policy_logits: list[torch.Tensor]  # [P_i + R_i, V], actor being trained
    old_log_probs: list[torch.Tensor]  # [R_i]
    ref_log_probs: list[torch.Tensor]  # [R_i]
    max_response_len: int = MAX_RESPONSE_LEN

    @property
    def batch_size(self) -> int:
        return len(self.prompt_ids)

    @property
    def truncated(self) -> list[bool]:
        return [response_len >= self.max_response_len for response_len in self.response_lens]


def _response_log_probs(logits: torch.Tensor, tokens: torch.Tensor, response_len: int) -> torch.Tensor:
    """Log-prob of each response token under ``logits``; position t-1 predicts token t."""
    prompt_len = tokens.numel() - response_len
    log_probs = torch.log_softmax(logits[prompt_len - 1 : -1], dim=-1)
    return log_probs.gather(-1, tokens[prompt_len:].unsqueeze(-1)).squeeze(-1)


def _drifted_log_probs(log_probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Push about a tenth of the tokens far from the current policy, so ratios leave [1 - eps, 1 + eps] and exceed c."""
    drift = torch.randint(-1, 2, log_probs.shape, generator=g) * (torch.rand(log_probs.shape, generator=g) < 0.15)
    return log_probs + 2.0 * drift


def _make_rollout(
    seed: int,
    *,
    num_prompts: int | None = None,
    samples_per_prompt: int | None = None,
    max_response_len: int = MAX_RESPONSE_LEN,
    case: str = "random",
) -> _Rollout:
    g = torch.Generator().manual_seed(seed)

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (1,), generator=g).item())

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=g)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=g)

    num_prompts = num_prompts or randint(2, 6)
    samples_per_prompt = samples_per_prompt or 4 * randint(1, 3)
    batch_size = num_prompts * samples_per_prompt

    rollout = _Rollout([], [], [], [], [], [], [], [], [], [], [], max_response_len=max_response_len)
    for prompt_id in range(num_prompts):
        prompt_mean, prompt_spread = rand(1).item(), rand(1).item() * 0.5
        binary_rewards = rand(1).item() < 0.25
        for _ in range(samples_per_prompt):
            # a third of the responses hit the length limit
            response_len = max_response_len if rand(1).item() < 0.33 else randint(2, max_response_len - 1)
            prompt_len = randint(1, 6)
            reward = prompt_mean + prompt_spread * randn(1).item()
            if binary_rewards:
                reward = float(rand(1).item() < prompt_mean)

            tokens = torch.randint(0, VOCAB_SIZE, (prompt_len + response_len,), generator=g)
            policy_logits = randn(prompt_len + response_len, VOCAB_SIZE)
            old_logits = policy_logits + 0.3 * randn(prompt_len + response_len, VOCAB_SIZE)
            ref_logits = policy_logits + 0.3 * randn(prompt_len + response_len, VOCAB_SIZE)
            old_values = reward + 0.5 * randn(response_len)

            rollout.prompt_ids.append(prompt_id)
            rollout.prompt_lens.append(prompt_len)
            rollout.response_lens.append(response_len)
            rollout.loss_masks.append(torch.ones(response_len, dtype=torch.int))
            rollout.rewards.append(reward)
            rollout.tokens.append(tokens)
            rollout.old_values.append(old_values)
            rollout.new_values.append(old_values + 0.3 * randn(response_len))
            rollout.policy_logits.append(policy_logits)
            rollout.old_log_probs.append(_drifted_log_probs(_response_log_probs(old_logits, tokens, response_len), g))
            rollout.ref_log_probs.append(_response_log_probs(ref_logits, tokens, response_len))

    first_prompt = range(samples_per_prompt)
    if case == "zero_std_prompt":
        # every sample of the first prompt has the same return: std 0, floor active
        for i in first_prompt:
            rollout.rewards[i] = rollout.rewards[0]
    elif case == "single_surviving_sample":
        # all but one sample of the first prompt are fully masked (aborted): std of one sample is 0
        for i in first_prompt[1:]:
            rollout.loss_masks[i] = torch.zeros_like(rollout.loss_masks[i])
    elif case == "masked_tokens":
        # multi-turn style: environment tokens inside the response carry no loss
        for i in range(batch_size):
            keep = (rand(rollout.response_lens[i]) < 0.7).int()
            rollout.loss_masks[i] = keep if i % 5 else torch.zeros_like(keep)
    elif case == "all_overlong":
        _set_response_lens(rollout, [max_response_len] * batch_size, g)
    elif case == "none_overlong":
        _set_response_lens(rollout, [min(r, max_response_len - 1) for r in rollout.response_lens], g)
    else:
        assert case == "random", case
    return rollout


def _set_response_lens(rollout: _Rollout, response_lens: list[int], g: torch.Generator) -> None:
    for i, response_len in enumerate(response_lens):
        total_len = rollout.prompt_lens[i] + response_len
        tokens = torch.randint(0, VOCAB_SIZE, (total_len,), generator=g)
        policy_logits = torch.randn((total_len, VOCAB_SIZE), generator=g)
        old_values = rollout.rewards[i] + 0.5 * torch.randn((response_len,), generator=g)
        rollout.response_lens[i] = response_len
        rollout.loss_masks[i] = torch.ones(response_len, dtype=torch.int)
        rollout.tokens[i] = tokens
        rollout.policy_logits[i] = policy_logits
        rollout.old_values[i] = old_values
        rollout.new_values[i] = old_values + 0.3 * torch.randn((response_len,), generator=g)
        for target in (rollout.old_log_probs, rollout.ref_log_probs):
            logits = policy_logits + 0.3 * torch.randn((total_len, VOCAB_SIZE), generator=g)
            target[i] = _drifted_log_probs(_response_log_probs(logits, tokens, response_len), g)


def _critic_steps(rollout: _Rollout, seed: int) -> list[list[int]]:
    """Critic mini-batches as the port schedules them; both sides train on the same rows."""
    num_steps = NUM_CRITIC_STEPS if rollout.batch_size % NUM_CRITIC_STEPS == 0 else 1
    return _shuffled_step_indices(rollout.batch_size, num_steps, seed=seed)


# ---------------------------------------------------------------------------
# Port (miles)
# ---------------------------------------------------------------------------


def _miles_args(**overrides) -> Namespace:
    settings = dict(
        advantage_estimator="ppo",
        kl_coef=0.0,
        gamma=1.0,
        lambd=1.0,
        normalize_advantages=True,
        calculate_per_token_loss=True,
        eps_clip=EPS_CLIP,
        eps_clip_high=EPS_CLIP,
        eps_clip_c=EPS_CLIP_C,
        value_clip=VALUE_CLIP,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_kl_loss=True,
        kl_loss_coef=KL_LOSS_COEF,
        kl_loss_type="low_var_kl",
        # rollout-side flags read by convert_samples_to_train_data
        reward_key=None,
        rewards_normalization=True,
        rollout_top_logprobs_num=0,
        # the EasyPPO flags
        critic_return_std_floor=RETURN_STD_FLOOR,
        value_loss_coef=VALUE_LOSS_COEF,
        filter_truncated_in_actor_loss=True,
    )
    return make_args(**{**settings, **overrides})


def _miles_samples(rollout: _Rollout) -> list[Sample]:
    return [
        Sample(
            group_index=rollout.prompt_ids[i],
            index=i,
            tokens=rollout.tokens[i].tolist(),
            response_length=rollout.response_lens[i],
            reward=rollout.rewards[i],
            loss_mask=rollout.loss_masks[i].tolist(),
            status=Sample.Status.TRUNCATED if rollout.truncated[i] else Sample.Status.COMPLETED,
        )
        for i in range(rollout.batch_size)
    ]


def _miles_batch(rollout_data: dict, rows: list[int], keys: list[str]) -> dict:
    batch = {key: [rollout_data[key][i] for i in rows] for key in keys}
    batch["unconcat_tokens"] = batch.pop("tokens")
    return batch


def _miles_loss(args: Namespace, batch: dict, logits: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-mean loss as Megatron applies it: the summed loss over the returned token normalizer."""
    loss_sum, num_tokens, _ = loss_function(
        args, batch, num_microbatches=1, logits=torch.cat(logits).unsqueeze(0), num_rollouts=len(logits)
    )
    return loss_sum / num_tokens, num_tokens


def _value_logits(rollout: _Rollout, rows: list[int]) -> list[torch.Tensor]:
    """Value-head output per sample, ``[P_i + R_i, 1]``: position t-1 holds the value of response token t."""
    return [
        torch.cat([torch.zeros(rollout.prompt_lens[i] - 1), rollout.new_values[i], torch.zeros(1)]).unsqueeze(-1)
        for i in rows
    ]


def _run_miles_critic(rollout: _Rollout, critic_steps: list[list[int]], args: Namespace) -> tuple[dict, dict]:
    """Rollout-side conversion and the critic train step; returns ``(outputs, rollout_data)``."""
    make_parallel_state()

    # rollout side: prompt-group weights are computed on the whole batch before the DP split
    train_data = convert_samples_to_train_data(args, _miles_samples(rollout), {}, None, None)
    rollout_data = {
        "tokens": [torch.tensor(t, dtype=torch.long) for t in train_data["tokens"]],
        "loss_masks": [torch.tensor(m, dtype=torch.int) for m in train_data["loss_masks"]],
        "response_lengths": train_data["response_lengths"],
        "total_lengths": [len(t) for t in train_data["tokens"]],
        "rewards": train_data["rewards"],
        "truncated": train_data["truncated"],
        "critic_loss_weights": train_data["critic_loss_weights"],
        "values": [v.clone() for v in rollout.old_values],
    }

    # advantages / returns from the pre-update values, then one loss per critic mini-batch
    compute_advantages_and_returns(args, rollout_data)
    explained_variance = explained_variance_from_stats(
        compute_explained_variance_stats(rollout_data["returns"], rollout_data["values"], rollout_data["loss_masks"])
    )
    args.loss_type = "value_loss"
    keys = ["tokens", "loss_masks", "response_lengths", "total_lengths", "values", "returns", "critic_loss_weights"]
    value_losses = [
        _miles_loss(args, _miles_batch(rollout_data, rows, keys), _value_logits(rollout, rows))[0]
        for rows in critic_steps
    ]
    outputs = {
        "weights": torch.tensor(train_data["critic_loss_weights"], dtype=torch.float32),
        "returns": rollout_data["returns"],
        "explained_variance": explained_variance,
        "value_losses": torch.stack(value_losses),
    }
    return outputs, rollout_data


def _run_miles(rollout: _Rollout, critic_steps: list[list[int]]) -> dict:
    args = _miles_args()
    outputs, rollout_data = _run_miles_critic(rollout, critic_steps, args)

    # actor: same values, advantages recomputed and whitened over every sample, truncated ones then dropped
    rollout_data["log_probs"] = [x.clone() for x in rollout.old_log_probs]
    rollout_data["ref_log_probs"] = [x.clone() for x in rollout.ref_log_probs]
    compute_advantages_and_returns(args, rollout_data)
    outputs["advantages"] = [a.clone() for a in rollout_data["advantages"]]
    if args.filter_truncated_in_actor_loss:
        drop_truncated_samples_from_actor_loss(rollout_data)
    args.loss_type = "policy_loss"
    keys = ["tokens", "loss_masks", "response_lengths", "total_lengths", "log_probs", "ref_log_probs", "advantages"]
    all_rows = list(range(rollout.batch_size))
    actor_loss, actor_tokens = _miles_loss(args, _miles_batch(rollout_data, all_rows, keys), rollout.policy_logits)
    outputs["actor_loss"] = actor_loss
    outputs["actor_tokens"] = int(actor_tokens)
    return outputs


# ---------------------------------------------------------------------------
# Reference pipeline
# ---------------------------------------------------------------------------


def _pad(tensors: list[torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Right-pad per-sample response tensors to ``[B, R_max]`` (the verl response layout)."""
    return torch.nn.utils.rnn.pad_sequence([t.to(dtype) for t in tensors], batch_first=True)


def _run_reference_critic(ref: Namespace, rollout: _Rollout, critic_steps: list[list[int]]) -> dict:
    algos = ref.core_algos
    batch_size = rollout.batch_size
    response_mask = _pad(rollout.loss_masks)
    old_values = _pad(rollout.old_values)

    # rm_scores: the scalar reward sits on the last actor token of the response
    token_level_scores = torch.zeros_like(old_values)
    for i, loss_mask in enumerate(rollout.loss_masks):
        if loss_mask.any():
            token_level_scores[i, loss_mask.nonzero().max()] = rollout.rewards[i]

    # PPOTrainer._compute_advantage
    uids = np.array([f"prompt-{p}" for p in rollout.prompt_ids], dtype=object)
    weights, _ = algos.compute_prompt_variance_loss_weights(
        sequence_rewards=token_level_scores.sum(dim=-1),
        sample_uids=uids,
        beta=0.5,
        w_min=RETURN_STD_FLOOR,
        w_max=None,
        valid_mask=response_mask.bool().any(dim=-1),
    )
    advantages, returns = algos.compute_gae_advantage_return(
        token_level_rewards=token_level_scores, values=old_values, response_mask=response_mask, gamma=1.0, lam=1.0
    )

    # compute_data_metrics: critic/vf_explained_var
    max_prompt_len = max(rollout.prompt_lens)
    prompt_attention = torch.stack(
        [(torch.arange(max_prompt_len) >= max_prompt_len - p).long() for p in rollout.prompt_lens]
    )
    response_attention = _pad([torch.ones(r) for r in rollout.response_lens], dtype=torch.long)
    data = ref.DataProto.from_dict(
        tensors={
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores,
            "advantages": advantages,
            "returns": returns,
            "values": old_values,
            "prompts": torch.zeros(batch_size, max_prompt_len, dtype=torch.long),
            "responses": torch.zeros_like(response_attention),
            "attention_mask": torch.cat([prompt_attention, response_attention], dim=-1),
            "response_mask": response_mask,
        }
    )
    explained_variance = ref.compute_data_metrics(data, use_critic=True)["critic/vf_explained_var"]

    # PPOTrainer._update_critic: one optimizer step per mini-batch, token-mean over that mini-batch
    new_values = _pad(rollout.new_values)
    value_losses = []
    for rows in critic_steps:
        vf_loss, _ = algos.compute_value_loss(
            vpreds=new_values[rows],
            returns=returns[rows],
            values=old_values[rows],
            response_mask=response_mask[rows],
            cliprange_value=VALUE_CLIP,
            loss_agg_mode="token-mean",
            dp_size=1,
            batch_num_tokens=response_mask[rows].sum(),
            loss_weights=weights[rows],
        )
        value_losses.append(vf_loss)

    return {
        "weights": weights,
        "returns": returns,
        "advantages": advantages,
        "explained_variance": explained_variance,
        "value_losses": torch.stack(value_losses),
        "response_mask": response_mask,
    }


def _overlong_kv_batch(rollout: _Rollout) -> _KVBatch:
    return _KVBatch(
        keys=[str(i) for i in range(rollout.batch_size)], tags=[{"response_len": r} for r in rollout.response_lens]
    )


def _run_reference(ref: Namespace, rollout: _Rollout, critic_steps: list[list[int]]) -> dict:
    algos = ref.core_algos
    outputs = _run_reference_critic(ref, rollout, critic_steps)

    # PPOTrainer._step_once: the overlong rows are masked only now, for the actor update
    kv_batch, _ = ref.filter_overlong_response_kv_batch(
        _overlong_kv_batch(rollout),
        enabled=True,
        max_response_length=rollout.max_response_len,
        metrics_prefix="training",
    )
    keep = torch.tensor([not tag.get("overlong_filtered", False) for tag in kv_batch.tags])
    actor_mask = outputs["response_mask"] * keep.unsqueeze(-1)

    # ppo_loss: clipped policy loss plus the KL loss, both token-mean over the actor mini-batch
    log_prob = _pad(
        [
            _response_log_probs(rollout.policy_logits[i], rollout.tokens[i], rollout.response_lens[i])
            for i in range(rollout.batch_size)
        ]
    )
    config = ref.OmegaConf.create(
        {
            "clip_ratio": EPS_CLIP,
            "clip_ratio_low": EPS_CLIP,
            "clip_ratio_high": EPS_CLIP,
            "clip_ratio_c": EPS_CLIP_C,
            "global_batch_info": {"dp_size": 1, "batch_num_tokens": actor_mask.sum().item()},
        }
    )
    pg_loss, _ = algos.compute_policy_loss_vanilla(
        old_log_prob=_pad(rollout.old_log_probs),
        log_prob=log_prob,
        advantages=outputs["advantages"],
        response_mask=actor_mask.bool(),
        loss_agg_mode="token-mean",
        config=config,
    )
    kld = algos.kl_penalty(logprob=log_prob, ref_logprob=_pad(rollout.ref_log_probs), kl_penalty="low_var_kl")
    kl_loss = algos.agg_loss(
        loss_mat=kld, loss_mask=actor_mask.bool(), loss_agg_mode="token-mean", **config.global_batch_info
    )
    outputs["actor_loss"] = pg_loss + kl_loss * KL_LOSS_COEF
    outputs["actor_tokens"] = int(actor_mask.sum())
    return outputs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _scaled_explained_variance_difference(miles: dict, reference: dict) -> float:
    """EV difference in units of the variance ratio it is computed from.

    ``1 - EV = Var(returns - values) / Var(returns)`` is unbounded for a poor critic (it reached 17.6 in
    these batches), and the float32 rounding of the returns scales with it, so the budget is relative to
    ``max(1, |1 - EV|)``.
    """
    difference = abs(miles["explained_variance"] - reference["explained_variance"])
    return difference / max(1.0, abs(1.0 - reference["explained_variance"]))


def _max_abs_differences(rollout: _Rollout, miles: dict, reference: dict) -> dict[str, float]:
    """Max abs difference per ingredient; token-level tensors are compared on loss tokens only."""
    active = torch.cat([m.bool() for m in rollout.loss_masks])

    def on_loss_tokens(miles_list: list[torch.Tensor], reference_padded: torch.Tensor) -> float:
        reference_list = [reference_padded[i, :r] for i, r in enumerate(rollout.response_lens)]
        diff = (torch.cat(miles_list) - torch.cat(reference_list)).abs()[active]
        return diff.max().item() if diff.numel() else 0.0

    return {
        "weights": (miles["weights"] - reference["weights"]).abs().max().item(),
        "returns": on_loss_tokens(miles["returns"], reference["returns"]),
        "advantages": on_loss_tokens(miles["advantages"], reference["advantages"]),
        "explained_variance": abs(miles["explained_variance"] - reference["explained_variance"]),
        "explained_variance_scaled": _scaled_explained_variance_difference(miles, reference),
        "value_loss": (miles["value_losses"] - reference["value_losses"]).abs().max().item(),
        "actor_loss": (miles["actor_loss"] - reference["actor_loss"]).abs().item(),
    }


def _assert_parity(rollout: _Rollout, miles: dict, reference: dict) -> dict[str, float]:
    diffs = _max_abs_differences(rollout, miles, reference)
    assert torch.equal(miles["weights"], reference["weights"]), diffs
    assert miles["actor_tokens"] == reference["actor_tokens"]
    assert diffs["returns"] <= ATOL_RETURNS, diffs
    assert diffs["advantages"] <= ATOL_ADVANTAGES, diffs
    assert diffs["explained_variance_scaled"] <= ATOL_EXPLAINED_VARIANCE, diffs
    assert diffs["value_loss"] <= ATOL_VALUE_LOSS, diffs
    assert diffs["actor_loss"] <= ATOL_ACTOR_LOSS, diffs
    return diffs


def _run_both(ref: Namespace, rollout: _Rollout, seed: int) -> dict[str, float]:
    critic_steps = _critic_steps(rollout, seed)
    return _assert_parity(rollout, _run_miles(rollout, critic_steps), _run_reference(ref, rollout, critic_steps))


@pytest.mark.parametrize("seed", range(NUM_RANDOM_BATCHES))
def test_random_batches_match_reference(ref, seed):
    _run_both(ref, _make_rollout(seed), seed)


@pytest.mark.parametrize("case", ["zero_std_prompt", "single_surviving_sample", "masked_tokens", "none_overlong"])
@pytest.mark.parametrize("seed", range(1000, 1008))
def test_adversarial_batches_match_reference(ref, case, seed):
    _run_both(ref, _make_rollout(seed, case=case), seed)


@pytest.mark.parametrize("seed", range(2000, 2004))
def test_paper_batch_shape_matches_reference(ref, seed):
    """16 prompts x 32 samples, four critic mini-batches of 128 responses."""
    rollout = _make_rollout(seed, num_prompts=16, samples_per_prompt=32)
    critic_steps = _critic_steps(rollout, seed)
    assert [len(rows) for rows in critic_steps] == [128] * 4
    _assert_parity(rollout, _run_miles(rollout, critic_steps), _run_reference(ref, rollout, critic_steps))


@pytest.mark.parametrize("seed", range(4000, 4004))
def test_long_responses_match_reference(ref, seed):
    """Responses longer than the 128-token chunk of the port's chunked GAE."""
    _run_both(ref, _make_rollout(seed, max_response_len=LONG_RESPONSE_LEN), seed)


def test_floor_is_active_for_zero_std_prompt(ref):
    rollout = _make_rollout(1000, case="zero_std_prompt")
    miles = _run_miles(rollout, _critic_steps(rollout, 0))
    floor_weight = miles["weights"][0]
    assert (miles["weights"] <= floor_weight).all(), "the zero-std prompt must carry the largest weight"
    samples_per_prompt = rollout.prompt_ids.count(0)
    assert torch.equal(miles["weights"][:samples_per_prompt], floor_weight.expand(samples_per_prompt))


def test_single_surviving_sample_gets_floor_weight_and_masked_samples_zero(ref):
    rollout = _make_rollout(1000, case="single_surviving_sample")
    miles = _run_miles(rollout, _critic_steps(rollout, 0))
    samples_per_prompt = rollout.prompt_ids.count(0)
    assert miles["weights"][0] == miles["weights"].max()
    assert (miles["weights"][1:samples_per_prompt] == 0).all()


@pytest.mark.parametrize("seed", range(3000, 3004))
def test_all_overlong_batch_is_refused_like_reference(ref, seed):
    """Both implementations refuse an actor step whose every sample hit the length limit."""
    rollout = _make_rollout(seed, case="all_overlong")
    with pytest.raises(ValueError, match="removed every training sample"):
        ref.filter_overlong_response_kv_batch(
            _overlong_kv_batch(rollout),
            enabled=True,
            max_response_length=rollout.max_response_len,
            metrics_prefix="training",
        )
    with pytest.raises(ValueError, match="removed every training sample"):
        convert_samples_to_train_data(_miles_args(), _miles_samples(rollout), {}, None, None)


@pytest.mark.parametrize("seed", range(3000, 3004))
def test_all_overlong_batch_still_trains_the_critic(ref, seed):
    """Overlong samples are filtered for the actor only: critic weights, returns and value loss still match."""
    rollout = _make_rollout(seed, case="all_overlong")
    critic_steps = _critic_steps(rollout, seed)
    miles, rollout_data = _run_miles_critic(rollout, critic_steps, _miles_args(filter_truncated_in_actor_loss=False))
    reference = _run_reference_critic(ref, rollout, critic_steps)

    assert all(rollout_data["truncated"])
    assert torch.equal(miles["weights"], reference["weights"])
    assert (miles["value_losses"] - reference["value_losses"]).abs().max().item() <= ATOL_VALUE_LOSS
    assert _scaled_explained_variance_difference(miles, reference) <= ATOL_EXPLAINED_VARIANCE

    # the actor step itself: every loss mask is dropped, so not a single loss token is left
    drop_truncated_samples_from_actor_loss(rollout_data)
    assert all(int(mask.sum()) == 0 for mask in rollout_data["loss_masks"])


def test_report_max_abs_differences(ref, capsys):
    """Print the measured maxima over every batch of this module (run with -s to see them)."""
    cases = [(seed, "random") for seed in range(NUM_RANDOM_BATCHES)]
    cases += [
        (seed, case)
        for case in ["zero_std_prompt", "single_surviving_sample", "masked_tokens", "none_overlong"]
        for seed in range(1000, 1008)
    ]
    maxima: dict[str, float] = {}
    rollouts = [(seed, _make_rollout(seed, case=case)) for seed, case in cases]
    rollouts += [(seed, _make_rollout(seed, num_prompts=16, samples_per_prompt=32)) for seed in range(2000, 2004)]
    rollouts += [(seed, _make_rollout(seed, max_response_len=LONG_RESPONSE_LEN)) for seed in range(4000, 4004)]
    for seed, rollout in rollouts:
        for key, value in _run_both(ref, rollout, seed).items():
            maxima[key] = max(maxima.get(key, 0.0), value)

    with capsys.disabled():
        print(f"\nEasyPPO parity, max abs difference over {len(rollouts)} batches (float32):")
        for key, value in maxima.items():
            print(f"  {key:<28} {value:.3e}")
    assert maxima["weights"] == 0.0


# ---------------------------------------------------------------------------
# Flags off: plain Miles PPO is untouched (no reference needed)
# ---------------------------------------------------------------------------


def test_flags_off_value_loss_ignores_weights_and_keeps_miles_normalizer():
    rollout = _make_rollout(7, case="single_surviving_sample")
    make_parallel_state()
    args = make_args(advantage_estimator="ppo", kl_coef=0.0, calculate_per_token_loss=True, loss_type="value_loss")
    all_rows = list(range(rollout.batch_size))
    returns = [torch.full((r,), reward) for r, reward in zip(rollout.response_lens, rollout.rewards, strict=True)]
    rollout_data = {
        "tokens": rollout.tokens,
        "loss_masks": rollout.loss_masks,
        "response_lengths": rollout.response_lens,
        "total_lengths": [t.numel() for t in rollout.tokens],
        "values": rollout.old_values,
        "returns": returns,
        "critic_loss_weights": [123.0] * rollout.batch_size,
    }
    keys = ["tokens", "loss_masks", "response_lengths", "total_lengths", "values", "returns", "critic_loss_weights"]
    loss_sum, num_tokens, _ = loss_function(
        args,
        _miles_batch(rollout_data, all_rows, keys),
        num_microbatches=1,
        logits=torch.cat(_value_logits(rollout, all_rows)).unsqueeze(0),
        num_rollouts=rollout.batch_size,
    )

    old_values, new_values, flat_returns = (torch.cat(x) for x in (rollout.old_values, rollout.new_values, returns))
    clipped = old_values + (new_values - old_values).clamp(-VALUE_CLIP, VALUE_CLIP)
    expected = torch.max((clipped - flat_returns) ** 2, (new_values - flat_returns) ** 2)
    expected_sum = sum(
        (x * mask).sum() for x, mask in zip(expected.split(rollout.response_lens), rollout.loss_masks, strict=True)
    )
    assert torch.equal(loss_sum, expected_sum)
    # plain Miles counts one token for every sample with an empty loss mask
    num_empty = sum(int(mask.sum()) == 0 for mask in rollout.loss_masks)
    assert num_empty > 0
    assert int(num_tokens) == sum(int(mask.sum()) for mask in rollout.loss_masks) + num_empty


def test_flags_off_conversion_adds_no_weights_and_accepts_all_truncated():
    rollout = _make_rollout(3000, case="all_overlong")
    args = _miles_args(critic_return_std_floor=None, filter_truncated_in_actor_loss=False)
    train_data = convert_samples_to_train_data(args, _miles_samples(rollout), {}, None, None)
    assert "critic_loss_weights" not in train_data
    assert all(train_data["truncated"])


@pytest.mark.parametrize(("num_samples", "num_steps"), [(64, 4), (128, 4), (10, 4), (7, 1)])
def test_shuffled_step_indices_partition_the_samples(num_samples, num_steps):
    steps = _shuffled_step_indices(num_samples, num_steps, seed=11)
    assert len(steps) == num_steps
    assert sorted(i for step in steps for i in step) == list(range(num_samples))
    assert max(len(step) for step in steps) - min(len(step) for step in steps) <= 1
    assert steps == _shuffled_step_indices(num_samples, num_steps, seed=11)
    if num_samples >= 64:
        assert steps != _shuffled_step_indices(num_samples, num_steps, seed=12)
