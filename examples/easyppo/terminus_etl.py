"""Run the single-GPU asynchronous Terminus-2 configuration-drift smoke."""

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from examples.easyppo import async_ppo
from scripts.make_miles_terminal_data import task_records, write_jsonl


@dataclass
class SmokeArgs(async_ppo.ScriptArgs):
    dataset: str = "tmax"
    model_name: str = "Qwen3.5-9B"
    hf_repo: str = "Qwen/Qwen3.5-9B"
    hf_revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    megatron_model_type: str = "qwen3.5-9B"
    num_nodes: int = 1
    trainer_num_nodes: int = 1
    num_gpus_per_node: int = 1
    tensor_model_parallel_size: int = 1
    rollout_gpus_per_engine: int = 1
    join_ray_workers: bool = False
    terminal_environment: str = "modal"
    terminal_agent_protocol: str = "terminus-2"
    rollout_batch_size: int = 4
    samples_per_prompt: int = 2
    num_rollout: int = 1
    num_critic_only_steps: int = 0
    skip_eval_before_train: bool = True
    score_centering: bool = False
    use_rollout_logprobs: bool = True
    compact_threshold: int = 12000
    max_tokens_per_gpu: int = 160 * 1024
    max_context_tokens: int = 63488
    max_new_tokens: int = 16384
    max_steps: int = 120
    actor_lr: float = 3e-6
    weight_decay: float = 0
    seed: int = 42
    sglang_server_concurrency: int = 8

    @property
    def eval_name(self):
        return "etl_smoke" if self.prompt_data == self.eval_data else "tb21"

    @property
    def colocate(self):
        return True


def main(default_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-context-mode", required=default_mode is None, default=default_mode, choices=["full_detach", "actor-summary"])
    parser.add_argument("--prepare-model-only", action="store_true", help="Download the pinned base model and convert its Miles checkpoint.")
    parser.add_argument("--tasks-dir", type=Path)
    parser.add_argument("--prompt-data", type=Path, help="Prepared AgentPick training JSONL")
    parser.add_argument("--eval-data", type=Path, help="Prepared TB2.1 evaluation JSONL")
    parser.add_argument("--num-rollout", type=int, default=1)
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=SmokeArgs.max_tokens_per_gpu,
        help="Maximum input tokens per training microbatch.",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gpu-arch", choices=["blackwell", "hopper"], default="blackwell")
    parser.add_argument("--output-dir", type=Path)
    opts = parser.parse_args()
    if opts.prepare_model_only:
        args = SmokeArgs(model_dir=opts.model_dir)
        model = str(Path(args.model_dir) / args.model_name)
        backend = args.create_backend()
        backend.exec_command_cpu(shlex.join(["hf", "download", args.hf_repo, "--revision", args.hf_revision, "--local-dir", model]))
        backend.convert_checkpoint(model_name=args.model_name, megatron_model_type=args.megatron_model_type, num_gpus_per_node=1, dir_dst=args.model_dir, hf_checkpoint=model, megatron_path=args.megatron_path)
        return
    if opts.tasks_dir is None or opts.output_dir is None:
        parser.error("training requires --tasks-dir and --output-dir")
    args = SmokeArgs(critic_context_mode=opts.critic_context_mode, model_dir=opts.model_dir, output_dir=str(opts.output_dir), tasks_dir=str(opts.tasks_dir))
    out = opts.output_dir / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    from agentic_critic.harbor_terminus import ETL_SMOKE_TASK_ID

    if (opts.prompt_data is None) != (opts.eval_data is None):
        parser.error("--prompt-data and --eval-data must be supplied together")
    if opts.num_rollout < 1:
        parser.error("--num-rollout must be positive")
    args.num_rollout = opts.num_rollout
    args.max_tokens_per_gpu = opts.max_tokens_per_gpu
    if opts.prompt_data:
        args.prompt_data, args.eval_data = str(opts.prompt_data), str(opts.eval_data)
        args.eval_interval, args.eval_temperature, args.eval_top_p = 20, 0.7, 0.95
    else:
        task = opts.tasks_dir / "agentpick-csk4gby2/tasks" / ETL_SMOKE_TASK_ID
        rows = task_records(task, dataset="agentpick", split="train", path_root=opts.tasks_dir)
        for row in rows:
            row["metadata"]["agent_name"] = "terminus-2"
        args.prompt_data = args.eval_data = str(out / "etl.jsonl")
        write_jsonl(Path(args.prompt_data), rows)
    async_ppo._execute(
        args,
        dataset="tmax",
        script_path=__file__,
        extra_args=(
            "--fully-async",
            "--ppo-prompt-mean",
            "--ppo-head-chunk-size", "8192",
            "--pause-generation-mode",
            "retract",
            "--async-max-concurrent-samples",
            str(args.rollout_batch_size * args.samples_per_prompt),
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
