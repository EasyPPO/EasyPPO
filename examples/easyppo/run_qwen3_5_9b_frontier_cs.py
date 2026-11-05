"""EasyPPO on Frontier-CS with Qwen3.5-9B (SFT init), single node, Megatron backend.

=====================

The agentic recipe uses the bundled `agentic_critic.miles_fcs.generate` wrapper
and an external Frontier-CS HTTP judge. Pass --prompt-data and
--agentic-generate-function-path agentic_critic.miles_fcs.generate. For single-turn
training, supply --prompt-data and a judge-backed --custom-rm-path instead.

See examples/easyppo/USAGE.md for setup, context modes and critic-only training.

EasyPPO ("EasyPPO: Stabilizing the Critic Is Key", arXiv 2609.36802) is PPO with a learned
critic plus three changes, each behind its own Miles flag:

  * actor-only overlong filtering        --filter-truncated-in-actor-loss
  * noise-normalized critic regression   --critic-return-std-floor
  * smaller critic mini-batches          --critic-num-steps-per-rollout

Where each number below comes from:

  paper (arXiv 2609.36802v1, Tables A.1-A.2, Frontier-CS column; not present in the EasyPPO repo):
    GAE gamma = lambda = 1, 30 critic-only rollout steps, clip 0.2 (lower and upper),
    return-std floor 0.075, no LR warmup, critic LR 2e-6, 16 prompts x 32 samples,
    4 critic mini-batches of 128 responses, Qwen3.5-9B from the Frontier-CS SFT init,
    max prompt 8192 and max response 32768 tokens,
    actor LR 1e-6, weight decay 0.01, Adam betas (0.9, 0.999), grad clip 1.0, dual-clip c 3.0,
    value clip 0.2, KL loss coef 0.001 (low_var_kl), entropy coef 0, token-mean loss aggregation,
    rollout temperature 1.0. These shared settings also match configs/easyppo_aime.yaml.
  EasyPPO repo, code (not configurable there): value loss scaled by 0.5, advantages whitened
    over the rollout batch, critic mini-batches shuffled.
  run_qwen3_dense.py, Qwen3.5-9B recipe: parallelism, token budget and SGLang engine sizing.

=====================

Args:
  --prompt-data: Frontier-CS problems as jsonl. Required.
  --custom-rm-path: Reward function for single-turn training; unused with the agentic wrapper.
  --model-name: Directory name of the HF checkpoint under --model-dir.
  --hf-repo: HF repository the SFT init is downloaded from by `prepare`.
  --num-gpus-per-node: GPUs of the node (default: 8). The actor world size, and the critic's.
  --num-rollout: Number of rollout steps, the first 30 of which train the critic only.
  --rollout-max-response-len: Generation length limit; responses that hit it are "overlong".
  --save-interval: Checkpoint every this many rollout steps, for the actor and the critic.
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python examples/easyppo/run_qwen3_5_9b_frontier_cs.py \\
      --prompt-data /root/datasets/frontier-cs/train.jsonl \\
      --custom-rm-path my_frontier_cs.reward.judge_score
"""

import shlex
from dataclasses import dataclass, field

import typer

from miles.utils.external_utils import command_utils

# The paper's batch: 16 prompts x 32 samples, critic mini-batches of 128 responses. A smoke run overrides them.
_ROLLOUT_BATCH_SIZE = 16
_N_SAMPLES_PER_PROMPT = 32
_CRITIC_MINI_BATCH_SIZE = 128


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = field(default_factory=command_utils.create_run_id)
    prompt_data: str = ""
    custom_rm_path: str = ""
    model_name: str = "Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
    hf_repo: str = "HuanzhiMao/Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = 8
    num_rollout: int = 300
    rollout_max_response_len: int = 32768
    save_interval: int = 5
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""
    rollout_batch_size: int = _ROLLOUT_BATCH_SIZE
    n_samples_per_prompt: int = _N_SAMPLES_PER_PROMPT
    critic_mini_batch_size: int = _CRITIC_MINI_BATCH_SIZE
    num_critic_only_steps: int = 30
    tensor_model_parallel_size: int = 2
    max_tokens_per_gpu: int = 41984
    # Agentic Frontier-CS: multi-turn episodes with judge feedback and actor compaction. One episode is one
    # rollout and yields one sample per actor segment. The generate function is given by module path and must be
    # importable (PYTHONPATH); it sets the reward itself, so --custom-rm-path is not used.
    agentic_generate_function_path: str = ""
    critic_context_mode: str = "shared"

    @property
    def samples_per_rollout(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def critic_num_steps_per_rollout(self) -> int:
        return max(1, self.samples_per_rollout // self.critic_mini_batch_size)


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(shlex.join(["mkdir", "-p", args.model_dir, args.data_dir]))
    U.exec_command_cpu(
        shlex.join(["hf", "download", args.hf_repo, "--local-dir", f"{args.model_dir}/{args.model_name}"])
    )
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _require_frontier_cs_inputs(args: ScriptArgs) -> None:
    if args.agentic_generate_function_path:
        if not args.prompt_data:
            raise ValueError("the agentic recipe needs --prompt-data (problems as jsonl: prompt, metadata.problem_id)")
        return
    if not args.prompt_data or not args.custom_rm_path:
        raise ValueError(
            "For single-turn training, pass --prompt-data (problems as jsonl) "
            "and --custom-rm-path (judge-backed reward function) to launch this recipe."
        )


def execute(args: ScriptArgs):
    _require_frontier_cs_inputs(args)
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    # The critic starts from the same SFT init as the actor: --critic-load falls back to --load,
    # and --critic-save to --save with a '_critic' suffix.
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(args.model_dir + '/' + args.model_name + '/')} "
        f"--ref-load {shlex.quote(args.model_dir + '/' + args.model_name + '_torch_dist')} "
        f"--load {shlex.quote(load_save_path)} "
        f"--save {shlex.quote(load_save_path)} "
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
        f"--prompt-data {shlex.quote(args.prompt_data)} "
        f"{task_args}"
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        # one actor step per rollout, over all 512 responses
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
        # default 41984 >= max prompt 8192 + max response 32768; memory at this length is unverified
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
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
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


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    _require_frontier_cs_inputs(args)
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
