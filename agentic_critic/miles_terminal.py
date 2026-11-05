"""Miles generate function for TMAX and Terminal-Bench 2.1 Harbor tasks."""

import argparse
import asyncio
import math
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path

from agentic_critic.terminal_env import Completion, TerminalConfig


@dataclass
class _ActiveTrial:
    task: asyncio.Task
    args: argparse.Namespace
    evaluation: bool
    abort_requested: bool = False


_ACTIVE_TRIALS: dict[asyncio.Task, _ActiveTrial] = {}


@dataclass(frozen=True)
class _DiscardedTrial:
    failure_type: str
    reason: str


def _collector_aborted(state, *, evaluation: bool) -> bool:
    return not evaluation and bool(getattr(state, "aborted", False))


async def abort(args) -> None:
    """Cancel oversampled training trials and wait for Harbor's teardown."""
    loop = asyncio.get_running_loop()
    active = tuple(
        trial
        for trial in _ACTIVE_TRIALS.values()
        if trial.args is args and not trial.evaluation and trial.task.get_loop() is loop
    )
    for trial in active:
        if not trial.abort_requested:
            trial.abort_requested = True
            trial.task.cancel()
    await asyncio.gather(*(trial.task for trial in active), return_exceptions=True)


async def _run_cancellable_trial(*, args, evaluation: bool, state=None, **kwargs):
    # Harbor is optional outside the online wrapper. Cancellation must reach
    # Trial.run(), which owns sandbox deletion, rather than only aborting decode.
    from agentic_critic.harbor_terminal import run_trial

    task = asyncio.create_task(run_trial(**kwargs))
    active = _ActiveTrial(task=task, args=args, evaluation=evaluation)
    _ACTIVE_TRIALS[task] = active
    try:
        result = await task
        if asyncio.current_task().cancelling():
            # Trial cleanup may suppress the cancellation it received.
            raise asyncio.CancelledError("Terminal rollout owner was cancelled")
        if _collector_aborted(state, evaluation=evaluation):
            return _DiscardedTrial("CancelledError", "Collector discarded terminal rollout tail")
        return result
    except asyncio.CancelledError as exc:
        if asyncio.current_task().cancelling():
            raise
        if _collector_aborted(state, evaluation=evaluation):
            return _DiscardedTrial(type(exc).__name__, str(exc) or "Collector aborted terminal rollout")
        if not active.abort_requested:
            raise
        # Only this plugin's oversampling abort becomes an ABORTED sample.
        # External cancellation still propagates to the caller.
        return None
    except Exception as exc:
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError("Terminal rollout owner was cancelled") from exc
        if _collector_aborted(state, evaluation=evaluation):
            # Engine abort completes before the generic collector invokes our
            # hook. Its late decode/teardown error still belongs to this tail.
            return _DiscardedTrial(type(exc).__name__, str(exc))
        raise
    finally:
        _ACTIVE_TRIALS.pop(task, None)


def _config(args, metadata: dict) -> TerminalConfig:
    defaults = TerminalConfig(**{f.name: getattr(args, f"terminal_{f.name}") for f in fields(TerminalConfig)})
    config = defaults.with_overrides(metadata.get("terminal", {}))
    context_limit = args.rollout_max_context_len
    if context_limit is not None and config.max_context_tokens > context_limit:
        raise ValueError("terminal max_context_tokens exceeds --rollout-max-context-len")
    return config


def _validate_runtime(args, *, evaluation: bool = False):
    if args.partial_rollout:
        raise ValueError("Terminal episodes do not support --partial-rollout")
    # These side channels need their own per-segment merge contract. Refuse
    # them explicitly instead of accepting a flag and silently losing data.
    for name in (
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "use_sampling_support_replay",
    ):
        if getattr(args, name, False):
            raise ValueError(f"Terminal episodes do not yet support {name}")
    candidate_count = getattr(args, "rollout_top_logprobs_num", 0)
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError("Terminal rollout-top-logprobs-num must be a nonnegative integer")
    if candidate_count and not evaluation:
        if args.rollout_sampling_logprobs_mode != "selected" or args.rollout_temperature != 1.0:
            raise ValueError("Terminal top-k collection requires selected sampling logprobs and rollout temperature 1")
    if not math.isfinite(args.terminal_reward_divisor) or args.terminal_reward_divisor <= 0:
        raise ValueError("terminal reward divisor must be finite and positive")


def _validate_topk_sampling(sampling: dict) -> None:
    expected = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    for key, value in expected.items():
        if sampling.get(key, value) not in (None, value):
            raise ValueError(f"Terminal top-k collection requires unfiltered training sampling with {key}={value}")


def _completion_from_output(output: dict, *, candidate_count: int) -> Completion:
    info = output["meta_info"]
    candidates = {}
    if candidate_count:
        # Optional framework imports are needed only for the candidate feature.
        from miles.rollout.generate_utils.rollout_topk_logprobs import extract_rollout_topk_logprobs

        ids, logps = extract_rollout_topk_logprobs(info, candidate_count)
        candidates = {"topk_token_ids": ids, "topk_log_probs": logps}
    items = info.get("output_token_logprobs") or []
    tokens, logprobs = [x[1] for x in items], [x[0] for x in items]
    return Completion(
        tokens=tokens,
        logprobs=logprobs,
        text=output["text"],
        finish_reason=info["finish_reason"]["type"],
        **candidates,
    )


def _reward(args, metadata: dict, rewards: dict) -> tuple[float, dict]:
    # Reward type belongs to the dataset; its request metadata overrides the
    # launch default. The test-count scale is run-wide, never batch-dependent.
    mode = metadata.get("reward_mode", args.terminal_reward_mode)
    key = metadata.get("reward_key", args.terminal_reward_key)
    if mode not in ("native", "test-count"):
        raise ValueError(f"Unknown terminal reward mode: {mode!r}")
    fallback = None
    count = None
    if mode == "test-count":
        name = args.terminal_pass_count_key
        if name not in rewards:
            fallback = f"missing {name}"
        else:
            try:
                candidate = float(rewards[name])
                if (
                    isinstance(rewards[name], bool)
                    or not math.isfinite(candidate)
                    or candidate < 0
                    or not candidate.is_integer()
                ):
                    raise ValueError("invalid passing assertion count")
                count = candidate
            except (TypeError, ValueError, OverflowError):
                fallback = f"unparsable {name}"
    if count is not None:
        reward = count / args.terminal_reward_divisor
    else:
        if key not in rewards:
            raise ValueError(f"Harbor verifier has no reward {key!r}: {list(rewards)}")
        reward = float(rewards[key])
        if mode == "test-count":
            if reward not in (0.0, 1.0):
                raise ValueError("Test-count fallback requires a binary reward; use native for continuous scores")
            fallback += f"; used binary {key}"
    if not math.isfinite(reward):
        raise ValueError(f"Harbor reward must be finite, got {reward}")
    return reward, {"reward_mode": mode, "reward_fallback": fallback}


async def generate(input):
    # Optional Miles/Harbor dependencies are only needed by the online wrapper.
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, compute_routing_headers
    from miles.utils.http_utils import post
    from miles.utils.types import Sample

    args, base = input.args, input.sample
    _validate_runtime(args, evaluation=input.evaluation)
    candidate_count = args.rollout_top_logprobs_num if not input.evaluation else 0
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
        if _collector_aborted(input.state, evaluation=input.evaluation):
            raise asyncio.CancelledError("Collector aborted terminal rollout before decode")
        # Per-call remaining budget may only narrow the caller's sampling budget.
        sampling = {
            **input.sampling_params,
            "max_new_tokens": min(limit, input.sampling_params.get("max_new_tokens", args.rollout_max_response_len)),
        }
        payload, _ = compute_request_payload(args, prompt, sampling, evaluation=input.evaluation)
        if payload is None:
            raise RuntimeError("Terminal context limit disagrees with Miles request budget")
        if candidate_count:
            _validate_topk_sampling(payload["sampling_params"])
        output = await post(url, payload, headers=compute_routing_headers(args, base))
        if _collector_aborted(input.state, evaluation=input.evaluation):
            # Aborted SGLang replies can omit token/candidate side channels.
            # Wait for Harbor teardown instead of validating discarded output.
            raise asyncio.CancelledError("Collector aborted terminal rollout during decode")
        return _completion_from_output(output, candidate_count=candidate_count)

    result = await _run_cancellable_trial(
        args=args,
        evaluation=input.evaluation,
        state=input.state,
        task_path=task_path,
        trials_dir=Path(args.terminal_trials_dir),
        environment=args.terminal_environment,
        tokenizer=input.state.tokenizer,
        decode=decode,
        config=config,
    )
    if result is None or isinstance(result, _DiscardedTrial):
        sample = deepcopy(base)
        sample.status = Sample.Status.ABORTED
        sample.reward = None
        discarded = result or _DiscardedTrial("CancelledError", "Oversampling aborted terminal trial")
        sample.metadata = {
            **metadata,
            "terminal_abort_failure_type": discarded.failure_type,
            "terminal_abort_reason": discarded.reason,
        }
        return GenerateFnOutput(samples=sample)
    trace, rewards, trial_dir = result
    reward, reward_metadata = _reward(args, metadata, rewards)
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
            **reward_metadata,
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
    parser.add_argument("--terminal-reward-mode", choices=("native", "test-count"), default="native")
    parser.add_argument("--terminal-pass-count-key", default="passing_assertions")
    parser.add_argument("--terminal-reward-divisor", type=float, default=20.0)


generate.add_arguments = _add_arguments
