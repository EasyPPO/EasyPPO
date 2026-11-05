"""Shared Qwen3.5-27B terminal recipe and one-step asynchronous launch helpers.

Dataset-specific run_* wrappers own the CLI. Synchronous launch helpers reuse
these parameter blocks while choosing the synchronous driver and PPO actor loss.
"""

import base64
import json
import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import agentic_critic

from miles.utils.external_utils import command_utils as U


@dataclass(frozen=True)
class _Recipe:
    batch: int
    oversample: int


_RECIPES = {"frontiersmith": _Recipe(32, 35), "tmax": _Recipe(32, 35)}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    action: Literal["prepare", "train", "all"] = "all"
    num_nodes: int = 8
    num_gpus_per_node: int = 8
    trainer_num_nodes: int = 2
    rollout_num_gpus: int = 0
    dataset: Literal["frontiersmith", "tmax"] = "frontiersmith"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    model_name: str = "Qwen3.5-27B"
    hf_repo: str = "Qwen/Qwen3.5-27B"
    hf_revision: str = "fc05daec18b0a78c049392ed2e771dde82bdf654"
    megatron_path: str = "/root/Megatron-LM"
    prompt_data: str = ""
    eval_data: str = ""
    tasks_dir: str = ""
    frontiersmith_source_dir: str = ""
    frontier_cs_source_dir: str = ""
    trials_dir: str = ""
    critic_context_mode: Literal["full_detach", "actor-summary"] = "full_detach"
    critic_load: str = ""
    critic_weights_only: bool = False
    num_rollout: int = 200
    num_critic_only_steps: int = 30
    samples_per_prompt: int = 16
    critic_steps_per_rollout: int = 4
    critic_return_std_floor: float = 0.075
    rollout_batch_size: int = 0
    over_sampling_batch_size: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    actor_lr: float = 1e-6
    critic_lr: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    policy_clip: float = 0.2
    policy_clip_high: float = 0.28
    score_centering: bool = True
    score_centering_candidates: int = 128
    score_centering_is: Literal["none", "tis", "mis"] = "tis"
    score_centering_tis_clip: float = 2.0
    score_centering_mis_low: float = 0.5
    score_centering_mis_high: float = 5.0
    score_centering_chunk_size: int = 128
    score_centering_recompute_loss: bool = True
    value_clip: float = 0.2
    value_loss_coef: float = 0.5
    max_context_tokens: int = 32768
    compact_threshold: int = 24576
    max_new_tokens: int = 8192
    max_total_tokens: int = 120000
    summary_tokens: int = 2048
    max_steps: int = 300
    command_timeout: int = 120
    max_output_chars: int = 16000
    reward_mode: Literal["native", "test-count"] = "native"
    reward_key: str = "reward"
    pass_count_key: str = "passing_assertions"
    reward_divisor: float = 20.0
    max_tokens_per_gpu: int = 262144
    tensor_model_parallel_size: int = 4
    rollout_gpus_per_engine: int = 4
    sglang_mem_fraction_static: float = 0.7
    sglang_server_concurrency: int = 128
    eval_interval: int = 10
    eval_samples_per_prompt: int = 5
    eval_temperature: float = 1.0
    eval_top_p: float = 1.0
    eval_top_k: int = -1
    eval_max_tasks: int = 0
    eval_subset_seed: int = 42
    save_interval: int = 10
    seed: int = 1234
    chat_template_path: str = field(
        default_factory=lambda: str(Path(agentic_critic.__file__).with_name("qwen3_5_interleaved.jinja"))
    )

    def __post_init__(self):
        if self.dataset not in _RECIPES:
            raise ValueError("Choose frontiersmith or tmax")
        recipe = _RECIPES[self.dataset]
        if self.rollout_batch_size == 0:
            self.rollout_batch_size = recipe.batch
        if self.over_sampling_batch_size == 0:
            self.over_sampling_batch_size = max(recipe.oversample, self.rollout_batch_size)
        if self.rollout_num_gpus == 0:
            self.rollout_num_gpus = (self.num_nodes - self.trainer_num_nodes) * self.num_gpus_per_node

    @property
    def train_data(self) -> str:
        return self.prompt_data or str(Path(self.data_dir) / f"{self.dataset}.train.jsonl")

    @property
    def validation_data(self) -> str:
        filename = "fcs.eval.jsonl" if self.dataset == "frontiersmith" else "tb21.eval.jsonl"
        return self.eval_data or str(Path(self.data_dir) / filename)

    @property
    def eval_name(self) -> str:
        name = "fcs" if self.dataset == "frontiersmith" else "tb21"
        return name + "_quick" if self.eval_max_tasks else name

    @property
    def task_root(self) -> str:
        return self.tasks_dir or str(Path(self.data_dir) / "harbor")

    @property
    def source_dir(self) -> str:
        return self.frontiersmith_source_dir or str(Path(self.data_dir) / "FrontierSmith")

    @property
    def eval_source_dir(self) -> str:
        return self.frontier_cs_source_dir or str(Path(self.data_dir) / "Frontier-CS-eval")

    @property
    def trial_root(self) -> str:
        return self.trials_dir or str(Path(self.output_dir) / self.run_id / "trials")

    @property
    def checkpoint_root(self) -> str:
        return str(Path(self.output_dir) / self.run_id / "checkpoints")


def _validate(args: ScriptArgs) -> None:
    if not 0 < args.trainer_num_nodes < args.num_nodes or args.num_gpus_per_node <= 0:
        raise ValueError("Reserve positive, separate trainer and rollout node allocations")
    trainer_gpus = args.trainer_num_nodes * args.num_gpus_per_node
    if trainer_gpus + args.rollout_num_gpus != args.num_nodes * args.num_gpus_per_node or args.rollout_num_gpus <= 0:
        raise ValueError("Training GPUs plus dedicated rollout GPUs must equal the allocation")
    if args.rollout_gpus_per_engine <= 0 or args.rollout_num_gpus % args.rollout_gpus_per_engine:
        raise ValueError("Rollout GPUs must divide into complete inference engines")
    if args.rollout_gpus_per_engine not in (1, 2, 4, 8):
        raise ValueError("Qwen3.5-27B rollout TP must divide its attention and recurrent heads")
    if args.sglang_server_concurrency <= 0:
        raise ValueError("SGLang server concurrency must be positive")
    if args.tensor_model_parallel_size <= 0 or trainer_gpus % args.tensor_model_parallel_size:
        raise ValueError("Training GPUs must divide into complete tensor-parallel groups")
    # DP partitions episodes; one prompt's responses may span several ranks.
    global_batch_size = args.rollout_batch_size * args.samples_per_prompt
    if args.rollout_batch_size <= 0 or global_batch_size % (trainer_gpus // args.tensor_model_parallel_size):
        raise ValueError("Rollout episode batch must be positive and divisible by trainer DP size")
    if args.samples_per_prompt <= 1 or args.critic_steps_per_rollout <= 0:
        raise ValueError("Inverse-std training requires repeated samples and positive critic steps")
    if args.rollout_batch_size % args.critic_steps_per_rollout:
        raise ValueError("Critic mini-batches must contain complete prompt groups")
    critic_mini_batch = args.rollout_batch_size * args.samples_per_prompt // args.critic_steps_per_rollout
    if critic_mini_batch % (trainer_gpus // args.tensor_model_parallel_size):
        raise ValueError("Each critic mini-batch must distribute equal episode counts across trainer DP")
    if not math.isfinite(args.critic_return_std_floor) or args.critic_return_std_floor <= 0:
        raise ValueError("Critic inverse-std floor must be finite and positive")
    if args.eval_max_tasks < 0 or args.eval_subset_seed < 0:
        raise ValueError("Evaluation task limit and subset seed must be nonnegative")
    if args.eval_samples_per_prompt <= 0:
        raise ValueError("Evaluation samples per prompt must be positive")
    if args.dataset not in ("frontiersmith", "tmax"):
        raise ValueError("Choose frontiersmith or tmax")
    if args.over_sampling_batch_size < args.rollout_batch_size:
        raise ValueError("Oversampling batch cannot be smaller than the collected batch")
    if args.num_rollout <= 0 or not 0 <= args.num_critic_only_steps <= args.num_rollout:
        raise ValueError("Critic-only budget must be between zero and the positive rollout budget")
    if args.critic_context_mode not in ("full_detach", "actor-summary"):
        raise ValueError("Choose full_detach or actor-summary")
    if args.critic_weights_only and not args.critic_load:
        raise ValueError("--critic-weights-only requires --critic-load and a fresh run-id")
    if args.max_tokens_per_gpu < args.max_context_tokens:
        raise ValueError("Training token budget must fit at least one actor context")
    if args.eval_interval <= 0 or args.save_interval <= 0:
        raise ValueError("Evaluation and checkpoint intervals must be positive")
    if min(args.actor_lr, args.critic_lr, args.adam_eps, args.clip_grad, args.value_loss_coef) <= 0:
        raise ValueError("Learning rates, Adam epsilon, gradient clip and value coefficient must be positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0 or not 0 < args.sglang_mem_fraction_static < 1:
        raise ValueError("Invalid rollout sampling or SGLang memory setting")
    if args.top_k != -1 and args.top_k <= 0:
        raise ValueError("top-k must be -1 (disabled) or positive")
    if not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1 or args.weight_decay < 0:
        raise ValueError("Invalid Adam betas or weight decay")
    if (not args.score_centering and not 0 < args.policy_clip < 1) or args.value_clip < 0:
        raise ValueError("Invalid policy/value clipping")
    if not math.isfinite(args.policy_clip_high) or args.policy_clip_high < 0:
        raise ValueError("Invalid upper policy clipping")
    if args.score_centering:
        if args.temperature != 1 or args.top_p != 1 or args.top_k != -1:
            raise ValueError("Terminal score centering requires unfiltered training sampling at temperature one")
        if args.score_centering_candidates <= 0 or args.score_centering_chunk_size <= 0:
            raise ValueError("Score centering requires positive candidate/chunk counts")
        if args.score_centering_is not in ("none", "tis", "mis"):
            raise ValueError("Choose score-centering-is none, tis or mis")
        if (
            not math.isfinite(args.score_centering_tis_clip)
            or args.score_centering_tis_clip <= 0
            or not math.isfinite(args.score_centering_mis_low)
            or not math.isfinite(args.score_centering_mis_high)
            or not 0 < args.score_centering_mis_low <= args.score_centering_mis_high
        ):
            raise ValueError("Invalid official score-centering importance-weight bounds")
    if args.reward_mode not in ("native", "test-count") or args.reward_divisor <= 0:
        raise ValueError("Invalid terminal reward mode or divisor")
    # Validate the same budget contract the online terminal wrapper consumes.
    # Optional terminal dependencies are loaded only when the launcher runs.
    from agentic_critic.terminal_env import TerminalConfig

    TerminalConfig(
        max_steps=args.max_steps,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
        compact_threshold=args.compact_threshold,
        summary_tokens=args.summary_tokens,
        command_timeout=args.command_timeout,
        max_output_chars=args.max_output_chars,
    )


def _validate_dataset(args: ScriptArgs, *, dataset: str) -> None:
    if args.dataset != dataset:
        raise ValueError(f"This launcher requires --dataset {dataset}")


def _prepare(args: ScriptArgs, *, dataset: str) -> None:
    _validate_dataset(args, dataset=dataset)
    _validate(args)
    backend = args.create_backend()
    backend.exec_command_cpu(shlex.join(["mkdir", "-p", args.model_dir, args.data_dir]))
    if args.dataset == "frontiersmith":
        script, options = (
            "make_miles_frontiersmith_data.py",
            [
                "--source-dir",
                args.source_dir,
                "--frontier-cs-source-dir",
                args.eval_source_dir,
                "--tasks-dir",
                args.task_root,
                "--out",
                args.data_dir,
            ],
        )
    else:
        script, options = (
            "make_miles_terminal_data.py",
            [
                "--dataset",
                "tmax",
                "--tasks-dir",
                args.task_root,
                "--out",
                args.data_dir,
                "--tokenizer",
                str(Path(args.model_dir) / args.model_name),
            ],
        )
    hf_path = str(Path(args.model_dir) / args.model_name)
    backend.exec_command_cpu(
        shlex.join(["hf", "download", args.hf_repo, "--revision", args.hf_revision, "--local-dir", hf_path])
    )
    backend.exec_command_cpu(shlex.join(["python", str(U.repo_base_dir / "scripts" / script), *options]))
    backend.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type="qwen3.5-27B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=hf_path,
        megatron_path=args.megatron_path,
    )


def _checkpoint_args(args: ScriptArgs) -> list[str]:
    fields = [
        "--hf-checkpoint",
        str(Path(args.model_dir) / args.model_name),
        "--ref-load",
        str(Path(args.model_dir) / (args.model_name + "_torch_dist")),
        "--load",
        args.checkpoint_root,
        "--save",
        args.checkpoint_root,
        "--critic-save",
        args.checkpoint_root + "_critic",
        "--save-interval",
        str(args.save_interval),
    ]
    if args.critic_load:
        fields += ["--critic-load", args.critic_load]
    if args.critic_weights_only:
        fields += ["--finetune", "--no-load-optim", "--no-load-rng"]
    return fields


def _rollout_args(args: ScriptArgs) -> list[str]:
    eval_config = {
        "datasets": [
            {
                "name": args.eval_name,
                "path": args.validation_data,
                "max_samples": args.eval_max_tasks or None,
                "sample_seed": args.eval_subset_seed,
            }
        ]
    }
    return [
        "--prompt-data",
        args.train_data,
        "--input-key",
        "prompt",
        "--metadata-key",
        "metadata",
        "--custom-generate-function-path",
        "agentic_critic.miles_terminal.generate",
        "--rollout-shuffle",
        "--seed",
        str(args.seed),
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--over-sampling-batch-size",
        str(args.over_sampling_batch_size),
        "--n-samples-per-prompt",
        str(args.samples_per_prompt),
        "--global-batch-size",
        str(args.rollout_batch_size * args.samples_per_prompt),
        "--rollout-temperature",
        str(args.temperature),
        "--rollout-top-p",
        str(args.top_p),
        "--rollout-top-k",
        str(args.top_k),
        "--rollout-max-response-len",
        str(args.max_new_tokens),
        "--rollout-max-context-len",
        str(args.max_context_tokens),
        "--update-weights-interval",
        "1",
        "--eval-config",
        "base64:" + base64.b64encode(json.dumps(eval_config).encode()).decode(),
        "--eval-interval",
        str(args.eval_interval),
        "--no-eval-on-epoch",
        "--no-save-on-epoch",
        "--n-samples-per-eval-prompt",
        str(args.eval_samples_per_prompt),
        "--eval-temperature",
        str(args.eval_temperature),
        "--eval-top-p",
        str(args.eval_top_p),
        "--eval-top-k",
        str(args.eval_top_k),
        "--eval-max-response-len",
        str(args.max_new_tokens),
    ]


def _terminal_args(args: ScriptArgs) -> list[str]:
    return [
        "--chat-template-path",
        args.chat_template_path,
        "--terminal-enable-thinking",
        "--terminal-tasks-dir",
        args.task_root,
        "--terminal-trials-dir",
        args.trial_root,
        "--terminal-environment",
        "docker",
        "--terminal-compaction",
        "summary",
        "--terminal-compact-threshold",
        str(args.compact_threshold),
        "--terminal-max-context-tokens",
        str(args.max_context_tokens),
        "--terminal-max-new-tokens",
        str(args.max_new_tokens),
        "--terminal-max-total-tokens",
        str(args.max_total_tokens),
        "--terminal-summary-tokens",
        str(args.summary_tokens),
        "--terminal-max-steps",
        str(args.max_steps),
        "--terminal-command-timeout",
        str(args.command_timeout),
        "--terminal-max-output-chars",
        str(args.max_output_chars),
        "--terminal-reward-mode",
        args.reward_mode,
        "--terminal-reward-key",
        args.reward_key,
        "--terminal-pass-count-key",
        args.pass_count_key,
        "--terminal-reward-divisor",
        str(args.reward_divisor),
    ]


def _algorithm_args(args: ScriptArgs, *, keep_old_actor: bool = True) -> list[str]:
    argv = [
        "--advantage-estimator",
        "ppo",
        "--gamma",
        "1",
        "--lambd",
        "1",
        "--critic-context-mode",
        args.critic_context_mode,
        "--num-critic-only-steps",
        str(args.num_critic_only_steps),
        "--critic-num-steps-per-rollout",
        str(args.critic_steps_per_rollout),
        "--critic-return-std-floor",
        str(args.critic_return_std_floor),
        "--filter-truncated-in-actor-loss",
        "--value-clip",
        str(args.value_clip),
        "--value-loss-coef",
        str(args.value_loss_coef),
        "--entropy-coef",
        "0",
        "--kl-coef",
        "0",
        "--calculate-per-token-loss",
        "--log-critic-explained-variance",
        "--optimizer",
        "adam",
        "--lr",
        str(args.actor_lr),
        "--critic-lr",
        str(args.critic_lr),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
        "--adam-eps",
        str(args.adam_eps),
        "--weight-decay",
        str(args.weight_decay),
        "--clip-grad",
        str(args.clip_grad),
        "--lr-decay-style",
        "constant",
        "--lr-warmup-iters",
        "0",
        "--critic-lr-warmup-iters",
        "0",
    ]
    if args.score_centering:
        argv += [
            "--loss-type",
            "score_centering",
            "--use-rollout-logprobs",
            # This Miles base predates the upstream SC auto-enable. Use the
            # existing SGLang flag explicitly, matching the official recipe.
            "--sglang-enable-fp32-lm-head",
            "--rollout-top-logprobs-num",
            str(args.score_centering_candidates),
            "--score-centering-is",
            args.score_centering_is,
            "--log-probs-chunk-size",
            str(args.score_centering_chunk_size),
        ]
        if args.score_centering_is == "tis":
            argv += ["--score-centering-tis-clip", str(args.score_centering_tis_clip)]
        elif args.score_centering_is == "mis":
            argv += [
                "--score-centering-mis-low",
                str(args.score_centering_mis_low),
                "--score-centering-mis-high",
                str(args.score_centering_mis_high),
            ]
        if args.score_centering_recompute_loss:
            argv += ["--recompute-loss-function"]
    else:
        if keep_old_actor:
            argv += ["--keep-old-actor"]
        argv += [
            "--eps-clip",
            str(args.policy_clip),
            "--eps-clip-high",
            str(args.policy_clip_high),
        ]
    return argv


def _resource_args(args: ScriptArgs) -> list[str]:
    return [
        "--actor-num-nodes",
        str(args.trainer_num_nodes),
        "--actor-num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--critic-num-nodes",
        str(args.trainer_num_nodes),
        "--critic-num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--rollout-num-gpus",
        str(args.rollout_num_gpus),
        "--rollout-num-gpus-per-engine",
        str(args.rollout_gpus_per_engine),
        "--sglang-mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--sglang-server-concurrency",
        str(args.sglang_server_concurrency),
        "--tensor-model-parallel-size",
        str(args.tensor_model_parallel_size),
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--use-distributed-optimizer",
        "--offload-train",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--attention-dropout",
        "0",
        "--hidden-dropout",
        "0",
        "--attention-backend",
        "flash",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ]


def _execute(args: ScriptArgs, *, dataset: str, script_path: str) -> None:
    _validate_dataset(args, dataset=dataset)
    _validate(args)
    argv = (
        _checkpoint_args(args)
        + _rollout_args(args)
        + _terminal_args(args)
        + _algorithm_args(args)
        + _resource_args(args)
        + shlex.split(U.get_default_wandb_args(script_path, run_id=args.run_id))
    )
    args.create_backend().execute_train(
        train_args=shlex.join(argv),
        train_script="train_async.py",
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-27B",
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path},
    )
