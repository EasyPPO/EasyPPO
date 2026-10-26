"""Miles generate function for TMAX and Terminal-Bench 2.1 Harbor tasks."""

import argparse
import math
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from agentic_critic.terminal_env import Completion, TerminalConfig


def _config(args, metadata: dict) -> TerminalConfig:
    defaults = TerminalConfig(**{f.name: getattr(args, f"terminal_{f.name}") for f in fields(TerminalConfig)})
    config = defaults.with_overrides(metadata.get("terminal", {}))
    context_limit = args.rollout_max_context_len
    if context_limit is not None and config.max_context_tokens > context_limit:
        raise ValueError("terminal max_context_tokens exceeds --rollout-max-context-len")
    return config


def _validate_runtime(args):
    if args.partial_rollout:
        raise ValueError("Terminal episodes do not support --partial-rollout")
    # These side channels need their own per-segment merge contract. Refuse
    # them explicitly instead of accepting a flag and silently losing data.
    for name in (
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "use_sampling_support_replay",
        "rollout_top_logprobs_num",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Terminal episodes do not yet support {name}")


async def generate(input):
    # Optional Miles/Harbor dependencies are only needed by the online wrapper.
    from agentic_critic.harbor_terminal import run_trial
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, compute_routing_headers
    from miles.utils.http_utils import post
    from miles.utils.types import Sample

    args, base = input.args, input.sample
    _validate_runtime(args)
    metadata = base.metadata or {}
    config = _config(args, metadata)
    task_path = Path(metadata["task_path"])
    if not task_path.is_absolute():
        if not args.terminal_tasks_dir:
            raise ValueError("Relative task_path requires --terminal-tasks-dir")
        task_path = Path(args.terminal_tasks_dir) / task_path
    if not (task_path / "task.toml").is_file():
        raise ValueError(f"Not a Harbor task directory: {task_path}")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    async def decode(prompt: list[int], limit: int) -> Completion:
        # Per-call remaining budget may only narrow the caller's sampling budget.
        sampling = {
            **input.sampling_params,
            "max_new_tokens": min(limit, input.sampling_params.get("max_new_tokens", args.rollout_max_response_len)),
        }
        payload, _ = compute_request_payload(args, prompt, sampling, evaluation=input.evaluation)
        if payload is None:
            raise RuntimeError("Terminal context limit disagrees with Miles request budget")
        output = await post(url, payload, headers=compute_routing_headers(args, base))
        info = output["meta_info"]
        items = info.get("output_token_logprobs") or []
        return Completion(
            tokens=[x[1] for x in items],
            logprobs=[x[0] for x in items],
            text=output["text"],
            finish_reason=info["finish_reason"]["type"],
        )

    trace, rewards, trial_dir = await run_trial(
        task_path=task_path,
        trials_dir=Path(args.terminal_trials_dir),
        environment=args.terminal_environment,
        tokenizer=input.state.tokenizer,
        decode=decode,
        config=config,
    )
    reward_key = metadata.get("reward_key", args.terminal_reward_key)
    if reward_key not in rewards:
        raise ValueError(f"Harbor verifier has no reward {reward_key!r}: {list(rewards)}")
    reward = float(rewards[reward_key])
    if not math.isfinite(reward):
        raise ValueError(f"Harbor reward must be finite, got {reward}")
    samples = []
    for i, sample_fields in enumerate(trace.sample_fields()):
        sample = deepcopy(base)
        sample.rollout_id = base.rollout_id if base.rollout_id is not None else base.index
        if sample.rollout_id is None:
            raise ValueError("Terminal rollouts require a sample index or rollout_id")
        for name, value in sample_fields.items():
            setattr(sample, name, value)
        sample.response = input.state.tokenizer.decode(sample.tokens[-sample.response_length :])
        sample.reward = reward
        sample.status = Sample.Status.TRUNCATED if trace.truncated else Sample.Status.COMPLETED
        sample.metadata = {
            **metadata,
            "segment": i,
            "n_compactions": trace.n_compactions,
            "n_format_errors": trace.n_format_errors,
            "n_commands": trace.n_commands,
            "exit_status": trace.exit_status,
            "harbor_rewards": rewards,
            "trial_dir": trial_dir,
            "generated_tokens": trace.generated_tokens,
            "model_calls": len(trace.calls),
        }
        samples.append(sample)
    if not samples:
        # An oversized initial instruction is not a valid zero-token training sample.
        raise ValueError(f"Terminal episode produced no actor tokens: {trace.exit_status}; task={task_path}")
    return GenerateFnOutput(samples=samples)


def _add_arguments(parser):
    defaults = TerminalConfig()
    for f in fields(TerminalConfig):
        default = getattr(defaults, f.name)
        kwargs = {"default": default, "type": type(default)}
        if type(default) is bool:
            kwargs = {"default": default, "action": argparse.BooleanOptionalAction}
        if f.name == "compaction":
            kwargs["choices"] = ("summary", "keep_last_turns", "none")
        parser.add_argument("--terminal-" + f.name.replace("_", "-"), **kwargs)
    parser.add_argument("--terminal-tasks-dir", default="")
    parser.add_argument("--terminal-trials-dir", default="terminal-trials")
    parser.add_argument("--terminal-environment", choices=("docker", "daytona", "modal"), default="docker")
    parser.add_argument("--terminal-reward-key", default="reward")


generate.add_arguments = _add_arguments
