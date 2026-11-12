import asyncio
import logging
import os
import re
import shutil
from pathlib import Path

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.rollout.data_source import compute_global_dataset_state_path
from miles.utils.arguments import parse_args
from miles.utils.async_utils import Disposer, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.lora.utils import lora_rollout_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


def _add_checkpoint_retention_arg(parser):
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help="Keep this many completed PPO checkpoints (0 disables pruning).",
    )
    return parser


def _prune_checkpoints(args, rollout_id: int, keep: int) -> None:
    roots = (Path(args.save), Path(args.critic_save))
    state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=rollout_id))
    if any(read_checkpoint_tracker_iteration(root) != rollout_id for root in roots) or (
        args.rollout_global_dataset and not state_path.is_file()
    ):
        raise RuntimeError(f"Checkpoint {rollout_id} is incomplete. Skipping checkpoint pruning.")

    iterations_by_root = [
        {
            int(match[1]): path
            for path in root.iterdir()
            if (match := re.fullmatch(r"iter_(\d{7,})", path.name)) and path.is_dir() and not path.is_symlink()
        }
        for root in roots
    ]
    actor, critic = iterations_by_root
    if rollout_id not in actor or rollout_id not in critic:
        raise RuntimeError(f"Checkpoint directories for rollout {rollout_id} are missing. Skipping checkpoint pruning.")

    complete = sorted(iteration for iteration in actor.keys() & critic.keys() if iteration <= rollout_id)
    if len(complete) < keep:
        return
    oldest_kept = complete[-keep]
    for iteration in sorted(actor.keys() | critic.keys()):
        if iteration >= oldest_kept:
            break
        Path(compute_global_dataset_state_path(args.save, rollout_id=iteration)).unlink(missing_ok=True)
        for checkpoints in iterations_by_root:
            if iteration in checkpoints:
                shutil.rmtree(checkpoints[iteration])


async def train(args, *, disposer: Disposer):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    keep_checkpoints = getattr(args, "keep_last_checkpoints", 0)
    if keep_checkpoints < 0:
        raise ValueError("--keep-last-checkpoints must be non-negative")
    if keep_checkpoints and (
        not args.use_critic
        or args.megatron_config is not None
        or not args.save
        or not args.critic_save
        or Path(args.save).resolve() == Path(args.critic_save).resolve()
    ):
        raise ValueError(
            "--keep-last-checkpoints requires standard Megatron PPO with separate actor and critic save directories"
        )
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    if args.colocate_memory_peak_device == "gpu":
        assert (
            args.offload_train and args.offload_rollout
        ), "--colocate-memory-peak-device gpu requires --offload-train and --offload-rollout"
        assert not args.use_critic, "--colocate-memory-peak-device gpu is not wired for the critic path"

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await update_weights(args, actor_model, rollout_executor, inference_controller)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await inference_controller.onload_kv()

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def offload_train():
        if args.use_critic:
            return
        if args.offload_train:
            await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1 or keep_checkpoints > 0

        async def save_training_model(model):
            if args.use_critic and args.offload_train:
                await model.onload()
            await model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic and args.offload_train:
                await model.offload()

        # A frozen actor still needs the matching rollout checkpoint so that
        # joint actor/critic resume can recover during critic-only warmup.
        await save_training_model(actor_model)
        if args.use_critic:
            await save_training_model(critic_model)
        await rollout_executor.save(rollout_id)
        if keep_checkpoints:
            _prune_checkpoints(args, rollout_id, keep_checkpoints)

    if args.num_rollout > args.start_rollout_id and args.eval_interval is not None and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        if args.start_rollout_id == 0:
            await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)
        else:
            await eval_dispatcher.dispatch(args.start_rollout_id - 1)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        await inference_controller.prepare_rollout(rollout_id)
        rollout_data_pack = await rollout_executor.get(rollout_id)

        if args.offload_rollout:
            if args.colocate_memory_peak_device == "gpu":
                await inference_controller.offload_kv()
                await actor_model.onload()
                await inference_controller.offload_weights()
            else:
                offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
                if "kv_cache" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
                if "weight" in args.offload_rollout_level:
                    offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
                await inference_controller.offload(tags=offload_tags)

        if args.use_critic:
            values = await critic_model.train(rollout_id, rollout_data_pack.for_critic())
            if args.offload_train:
                await critic_model.offload()
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_pack, external_data=values)
                if args.offload_train:
                    await actor_model.offload()
            remove_train_output_refs(values)
        else:
            await actor_model.train(rollout_id, rollout_data_pack)
        remove_rollout_data_refs(args, rollout_data_pack)

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch if getattr(args, "save_on_epoch", True) else None,
            args.num_rollout,
        ):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        # One predicate for both blocks: the handoff below exists to feed this eval on the last rollout.
        eval_due = should_run_periodic_action(
            rollout_id,
            args.eval_interval,
            num_rollout_per_epoch if getattr(args, "eval_on_epoch", True) else None,
            args.num_rollout,
        )
        if rollout_id + 1 < args.num_rollout or eval_due:
            if args.colocate_memory_peak_device == "gpu":
                await actor_model.clear_memory()
                if lora_rollout_enabled(args):
                    await actor_model.offload_grad_buffer()
                await inference_controller.onload_weights()
                await offload_train()
            else:
                await offload_train()
                if args.offload_rollout:
                    await inference_controller.onload_weights()
            await update_weights(args, actor_model, rollout_executor, inference_controller, rollout_id=rollout_id)
            if args.offload_rollout:
                await inference_controller.onload_kv()

        if eval_due:
            await inference_controller.prepare_eval()
            await eval_dispatcher.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=_add_checkpoint_retention_arg)
    asyncio.run(with_disposer(train, args))
