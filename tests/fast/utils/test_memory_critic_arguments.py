"""Memory-critic schedules fail before any terminal rollouts are launched."""

from argparse import Namespace

import pytest

from miles.utils.arguments import _validate_easyppo_args


def _args(**overrides):
    values = dict(
        use_critic=True,
        train_backend="megatron",
        critic_context_mode="full_detach",
        critic_return_std_floor=0.075,
        critic_num_steps_per_rollout=4,
        value_loss_coef=0.5,
        log_critic_explained_variance=True,
        filter_truncated_in_actor_loss=True,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        offload_train=True,
        offload=False,
        offload_train_target="cpu",
        gamma=1,
        lambd=1,
        kl_coef=0,
        num_steps_per_rollout=1,
        calculate_per_token_loss=True,
        optimizer="adam",
        lr_decay_style="constant",
        delay_split_train_data_by_dp=False,
        use_dynamic_batch_size=True,
        multi_lora=False,
        megatron_config=None,
        indep_dp=False,
        enable_witness=False,
        async_save=False,
        use_fault_tolerance=False,
        rollout_batch_size=32,
        n_samples_per_prompt=16,
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=4,
        spec=["miles_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        qkv_format="thd",
        linear_attention_backend="fla",
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_four_critic_steps_with_inverse_std_support_dp_two_and_four(mode, tp_size):
    _validate_easyppo_args(_args(critic_context_mode=mode, tensor_model_parallel_size=tp_size))


def test_prompt_responses_may_span_dp_ranks_within_one_global_minibatch():
    # One prompt per global minibatch, four of its responses per DP rank.
    _validate_easyppo_args(_args(rollout_batch_size=4, n_samples_per_prompt=16))


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), -float("inf"), 0.0, -0.075])
@pytest.mark.parametrize("mode", ["full_detach", "actor-summary", "shared"])
def test_inverse_std_floor_must_be_finite_and_positive(floor, mode):
    with pytest.raises(AssertionError, match="finite and positive"):
        _validate_easyppo_args(_args(critic_context_mode=mode, critic_return_std_floor=floor))


@pytest.mark.parametrize("steps", [3, 64])
def test_critic_steps_must_divide_complete_prompt_groups(steps):
    with pytest.raises(ValueError, match="prompt group count.*divisible"):
        _validate_easyppo_args(_args(critic_num_steps_per_rollout=steps))


@pytest.mark.parametrize("prompts,responses", [(None, 16), (0, 16), (32, 0)])
def test_memory_critic_requires_positive_prompt_and_response_counts(prompts, responses):
    with pytest.raises(ValueError, match="positive rollout prompt and response counts"):
        _validate_easyppo_args(_args(rollout_batch_size=prompts, n_samples_per_prompt=responses))


def test_critic_minibatch_episode_count_must_divide_actor_dp():
    with pytest.raises(ValueError, match="mini-batch episode count.*actor DP"):
        _validate_easyppo_args(_args(rollout_batch_size=4, n_samples_per_prompt=2))


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_num_nodes": 0},
        {"actor_num_gpus_per_node": 7, "tensor_model_parallel_size": 4},
        {"tensor_model_parallel_size": 0},
    ],
)
def test_memory_critic_topology_must_form_complete_tp_groups(overrides):
    with pytest.raises(ValueError, match="complete tensor-parallel groups"):
        _validate_easyppo_args(_args(**overrides))


@pytest.mark.parametrize("mode", ["full_detach", "actor-summary"])
def test_native_optimizer_schedule_and_checkpoint_options_are_available(mode):
    _validate_easyppo_args(
        _args(
            critic_context_mode=mode,
            offload_train=False,
            optimizer="sgd",
            lr_decay_style="cosine",
            async_save=True,
            use_fault_tolerance=True,
        )
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"qkv_format": "sbhd"}, "THD"),
        ({"linear_attention_backend": "flashinfer"}, "FLA"),
        ({"spec": None}, "model spec"),
        ({"megatron_to_hf_mode": "bridge"}, "native Qwen3.5 layers"),
        ({"custom_model_provider_path": "custom.model"}, "native Qwen3.5 layers"),
    ],
)
def test_full_detach_requires_the_native_state_carrying_layers(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_easyppo_args(_args(**overrides))
