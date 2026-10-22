"""Qwen3.5 critic training configuration."""

import os
import shlex
from dataclasses import dataclass

from miles.utils.external_utils import command_utils

_ROLLOUT_BATCH_SIZE = 8
_N_SAMPLES_PER_PROMPT = 16
_CRITIC_MINI_BATCH_SIZE = 32

@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    prompt_data: str = ""
    custom_rm_path: str = ""
    model_name: str = "Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
    hf_repo: str = "HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = 8
    num_rollout: int = 40
    rollout_max_response_len: int = 32768
    save_interval: int = 5
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""
    rollout_batch_size: int = _ROLLOUT_BATCH_SIZE
    n_samples_per_prompt: int = _N_SAMPLES_PER_PROMPT
    critic_mini_batch_size: int = _CRITIC_MINI_BATCH_SIZE
    num_critic_only_steps: int = 40
    tensor_model_parallel_size: int = 1
    max_tokens_per_gpu: int = 120000
    # Agentic Frontier-CS: multi-turn episodes with judge feedback and actor compaction. One episode is one
    # rollout and yields one sample per actor segment. The generate function is given by module path and must be
    # importable (PYTHONPATH); it sets the reward itself, so --custom-rm-path is not used.
    agentic_generate_function_path: str = ""
    critic_context_mode: str = "shared"
    # Explicit tracking names. With wandb_group set, runs of one comparison share the group and differ by name;
    # without it the recipe keeps Miles' default naming.
    wandb_project: str = ""
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_entity: str = ""

    @property
    def samples_per_rollout(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def critic_num_steps_per_rollout(self) -> int:
        return max(1, self.samples_per_rollout // self.critic_mini_batch_size)


def _wandb_args(args: ScriptArgs) -> str:
    if not args.wandb_group:
        return command_utils.get_default_wandb_args(__file__, run_id=args.run_id)
    mode = os.environ.get("WANDB_MODE", "")
    key = os.environ.get("WANDB_API_KEY")
    if mode == "disabled" or (not key and mode != "offline"):
        return ""
    parts = [
        "--use-wandb",
        f"--wandb-project {shlex.quote(args.wandb_project or 'agentic-critic')}",
        f"--wandb-group {shlex.quote(args.wandb_group)}",
        f"--wandb-run-name {shlex.quote(args.wandb_run_name or args.wandb_group)}",
        "--disable-wandb-random-suffix",
    ]
    if args.wandb_entity:
        parts.append(f"--wandb-team {shlex.quote(args.wandb_entity)}")
    if mode == "offline":
        parts.append("--wandb-mode offline")
    if key:
        parts.append(f"--wandb-key {shlex.quote(key)}")
    return " ".join(parts) + " "


def _require_frontier_cs_inputs(args: ScriptArgs) -> None:
    if args.agentic_generate_function_path:
        if not args.prompt_data:
            raise ValueError("the agentic recipe needs --prompt-data (problems as jsonl: prompt, metadata.problem_id)")
        return
    if not args.prompt_data or not args.custom_rm_path:
        raise ValueError(
            "Miles has no Frontier-CS rollout or reward function yet: pass --prompt-data (problems as jsonl) "
            "and --custom-rm-path (judge-backed reward function) to launch this recipe."
        )


def execute(args: ScriptArgs):
    _require_frontier_cs_inputs(args)
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    # The critic starts from the same SFT init as the actor: --critic-load falls back to --load,
    # and --critic-save to --save with a '_critic' suffix.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}/ "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {args.save_interval} "
    )

    if args.agentic_generate_function_path:
        # The episode loop builds the chat turns itself from the raw statement, so no chat template here. The
        # samples of one episode share a rollout_id, which makes Miles treat the batch as compact rollouts: the
        # global batch size counts rollouts (episodes), and --use-dynamic-global-batch-size must stay off.
        task_args = (
            "--input-key prompt "
            "--metadata-key metadata "
            f"--custom-generate-function-path {args.agentic_generate_function_path} "
            f"--critic-context-mode {args.critic_context_mode} "
        )
    else:
        task_args = (
            "--input-key prompt "
            "--label-key label "
            "--apply-chat-template "
            f"--custom-rm-path {args.custom_rm_path} "
        )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        f"{task_args}"
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        # One actor step per rollout.
        f"--global-batch-size {args.samples_per_rollout} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        # Bound each packed micro-batch by the configured token budget.
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    ppo_args = (
        "--advantage-estimator ppo "
        "--gamma 1.0 "
        "--lambd 1.0 "
        # EasyPPO's GAE always whitens the advantages over the rollout batch
        "--normalize-advantages "
        f"--num-critic-only-steps {args.num_critic_only_steps} "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.2 "
        "--eps-clip-c 3.0 "
        "--value-clip 0.2 "
        "--entropy-coef 0.00 "
        # Reward-level KL (--kl-coef) is rejected with ppo; EasyPPO regularizes at the loss level too.
        "--kl-coef 0.00 "
        "--use-kl-loss "
        "--kl-loss-coef 0.001 "
        "--kl-loss-type low_var_kl "
        # token-mean loss aggregation for both the actor and the critic
        "--calculate-per-token-loss "
    )

    easyppo_args = (
        "--filter-truncated-in-actor-loss "
        "--critic-return-std-floor 0.075 "
        f"--critic-num-steps-per-rollout {args.critic_num_steps_per_rollout} "
        "--value-loss-coef 0.5 "
        "--log-critic-explained-variance "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--critic-lr 2e-6 "
        "--lr-decay-style constant "
        "--lr-warmup-iters 0 "
        "--critic-lr-warmup-iters 0 "
        "--weight-decay 0.01 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 1.0 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{easyppo_args} "
        f"{_wandb_args(args)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )


