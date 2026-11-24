"""Train Qwen3.5-9B with asynchronous Terminus-2 PPO on a Miles Ray cluster.

Use shared AgentPick/TB2.1 JSONL, converted weights and a run directory.
See Claude.md, Terminal Bench, for cluster setup and both critic commands.
"""

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from examples.easyppo import async_ppo


@dataclass
class ScriptArgs(async_ppo.ScriptArgs):
    dataset: str = "tmax"
    model_name: str = "Qwen3.5-9B"
    hf_repo: str = "Qwen/Qwen3.5-9B"
    hf_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    megatron_model_type: str = "qwen3.5-9B"
    num_nodes: int = 2
    trainer_num_nodes: int | None = None
    num_gpus_per_node: int = 8
    eval_num_gpus: int | None = None
    tensor_model_parallel_size: int = 1
    rollout_gpus_per_engine: int = 1
    terminal_environment: str = "modal"
    terminal_agent_protocol: str = "terminus-2"
    rollout_batch_size: int = 32
    samples_per_prompt: int = 12
    num_rollout: int = 150
    eval_interval: int = 15
    eval_temperature: float = 0.7
    eval_top_p: float = 0.95
    save_interval: int = 5
    actor_lr_warmup_iters: int = 1
    use_rollout_logprobs: bool = True
    compact_threshold: int = 16384
    max_tokens_per_gpu: int = 120 * 1024
    critic_max_tokens_per_gpu: int = 150 * 1024
    score_centering_recompute_loss: bool = False
    max_total_tokens: int = 100 * 1024
    max_context_tokens: int = 32768
    max_new_tokens: int = 8192
    max_steps: int = 120
    actor_lr: float = 3e-6
    weight_decay: float = 0
    seed: int = 42
    sglang_mem_fraction_static: float = 0.92
    sglang_server_concurrency: int = 512
    async_max_concurrent_samples: int = 1000

    def __post_init__(self):
        if self.num_nodes < 1 or self.num_gpus_per_node < 1:
            raise ValueError("Node and per-node GPU counts must be positive")
        total = self.num_nodes * self.num_gpus_per_node
        if self.eval_num_gpus is None:
            self.eval_num_gpus = max(1, total // 8)
        if self.trainer_num_nodes is None and self.trainer_gpus_per_node is None:
            trainer_gpus = 4 if (self.num_nodes, self.num_gpus_per_node) == (2, 8) else max(1, 3 * total // 8)
            # Ray places one worker per slot. Express the trainer world as equal-sized groups.
            widths = range(1, min(trainer_gpus, self.num_gpus_per_node) + 1)
            self.trainer_gpus_per_node = max(width for width in widths if trainer_gpus % width == 0)
            self.trainer_num_nodes = trainer_gpus // self.trainer_gpus_per_node
        elif self.trainer_num_nodes is None:
            self.trainer_num_nodes = 1
        super().__post_init__()


def main(default_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-context-mode", required=default_mode is None, default=default_mode, choices=["full_detach", "actor-summary"])
    parser.add_argument("--score-centering", action=argparse.BooleanOptionalAction, default=ScriptArgs.score_centering, help="Use the Miles score-centering actor loss with PPO critic advantages.")
    parser.add_argument("--prepare-model-only", action="store_true", help="Download the pinned base model and convert its Miles checkpoint.")
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--prompt-data", type=Path, help="Prepared AgentPick training JSONL")
    parser.add_argument("--eval-data", type=Path, help="Prepared TB2.1 evaluation JSONL")
    parser.add_argument("--num-critic-only-steps", type=int, default=ScriptArgs.num_critic_only_steps)
    parser.add_argument("--num-rollout", type=int, default=ScriptArgs.num_rollout)
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=ScriptArgs.max_tokens_per_gpu,
        help="Maximum input tokens per actor microbatch.",
    )
    parser.add_argument("--critic-max-tokens-per-gpu", type=int, default=ScriptArgs.critic_max_tokens_per_gpu,
                        help="Maximum input tokens per critic microbatch.")
    parser.add_argument("--num-nodes", type=int, default=ScriptArgs.num_nodes)
    parser.add_argument("--num-gpus-per-node", type=int, default=ScriptArgs.num_gpus_per_node)
    parser.add_argument("--trainer-num-nodes", type=int, default=ScriptArgs.trainer_num_nodes)
    parser.add_argument("--trainer-gpus-per-node", type=int)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=ScriptArgs.tensor_model_parallel_size)
    parser.add_argument("--eval-num-gpus", type=int, default=ScriptArgs.eval_num_gpus)
    parser.add_argument("--async-max-concurrent-samples", type=int, default=ScriptArgs.async_max_concurrent_samples)
    parser.add_argument("--use-distributed-post", action=argparse.BooleanOptionalAction, default=True,
                        help="Dispatch rollout HTTP POSTs through Ray actors.")
    parser.add_argument("--ray-hostfile", default=ScriptArgs.ray_hostfile)
    parser.add_argument("--run-id", help="Reuse the same ID and output directory to resume.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gpu-arch", choices=["blackwell", "hopper"], default="blackwell")
    parser.add_argument("--output-dir", type=Path)
    opts = parser.parse_args()
    if opts.prepare_model_only:
        args = ScriptArgs(model_dir=opts.model_dir)
        model = str(Path(args.model_dir) / args.model_name)
        backend = args.create_backend()
        backend.exec_command_cpu(shlex.join(["hf", "download", args.hf_repo, "--revision", args.hf_revision, "--local-dir", model]))
        backend.convert_checkpoint(model_name=args.model_name, megatron_model_type=args.megatron_model_type, num_gpus_per_node=1, dir_dst=args.model_dir, hf_checkpoint=model, megatron_path=args.megatron_path)
        return
    if not all((opts.tasks_dir, opts.output_dir, opts.prompt_data, opts.eval_data)):
        parser.error("training requires --tasks-dir, --output-dir, --prompt-data and --eval-data")
    if opts.async_max_concurrent_samples < ScriptArgs.samples_per_prompt:
        parser.error("--async-max-concurrent-samples must fit one complete prompt group")
    args = ScriptArgs(
        num_nodes=opts.num_nodes,
        num_gpus_per_node=opts.num_gpus_per_node,
        trainer_num_nodes=opts.trainer_num_nodes,
        trainer_gpus_per_node=opts.trainer_gpus_per_node,
        tensor_model_parallel_size=opts.tensor_model_parallel_size,
        eval_num_gpus=opts.eval_num_gpus,
        async_max_concurrent_samples=opts.async_max_concurrent_samples,
        ray_hostfile=opts.ray_hostfile,
        join_ray_workers=opts.num_nodes > 1,
        model_dir=opts.model_dir,
        output_dir=str(opts.output_dir),
        tasks_dir=str(opts.tasks_dir),
        prompt_data=str(opts.prompt_data),
        eval_data=str(opts.eval_data),
        critic_context_mode=opts.critic_context_mode,
        score_centering=opts.score_centering,
        max_tokens_per_gpu=opts.max_tokens_per_gpu,
        critic_max_tokens_per_gpu=opts.critic_max_tokens_per_gpu,
        num_rollout=opts.num_rollout,
        num_critic_only_steps=opts.num_critic_only_steps,
        **({"run_id": opts.run_id} if opts.run_id else {}),
    )
    async_ppo._execute(
        args,
        dataset="tmax",
        script_path=__file__,
        extra_args=(
            "--fully-async",
            *(("--use-distributed-post",) if opts.use_distributed_post else ()),
            "--max-keep", "3",
            "--use-checkpoint-opt-param-scheduler",
            "--critic-max-tokens-per-gpu", str(args.critic_max_tokens_per_gpu),
            "--ppo-prompt-mean",
            *(("--log-probs-backend", "fused") if args.score_centering else ()),
            "--pause-generation-mode",
            "in_place",
            "--async-max-concurrent-samples",
            str(args.async_max_concurrent_samples),
            "--max-weight-staleness",
            "4",
            "--sglang-max-running-requests",
            "512",
            "--sglang-attention-backend",
            "triton" if opts.gpu_arch == "blackwell" else "fa3",
            "--sglang-linear-attn-backend",
            "triton",
            "--linear-attention-backend",
            "fla",
        ),
    )


if __name__ == "__main__":
    main()
