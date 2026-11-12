import asyncio
import logging
import os
import re
import shutil
from pathlib import Path

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.training_utils.checkpoint.tracker import (
    CHECKPOINT_TRACKER_FILENAME,
    read_checkpoint_tracker_iteration,
)
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
from miles.utils.file_utils import atomic_write_text
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.lora.utils import lora_rollout_enabled
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)
PPO_CHECKPOINT_TRACKER = "latest_complete_ppo_checkpoint.txt"


def _checkpoint_directories(root: Path) -> dict[int, Path]:
    return {
        int(match[1]): path
        for path in root.iterdir()
        if (match := re.fullmatch(r"iter_(\d{7,})", path.name)) and path.is_dir() and not path.is_symlink()
    }


def _recover_checkpoint(args) -> None:
    if args.finetune or (args.requested_load, args.critic_load) != (args.save, args.critic_save):
        return

    actor_root, critic_root = Path(args.save), Path(args.critic_save)
    tracker = actor_root / PPO_CHECKPOINT_TRACKER
    actor_step = read_checkpoint_tracker_iteration(actor_root)
    critic_step = read_checkpoint_tracker_iteration(critic_root)
    if not tracker.is_file():
        # Adopt a complete checkpoint written before the joint tracker was introduced.
        if actor_step is not None and actor_step == critic_step:
            state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=actor_step))
            if not args.rollout_global_dataset or state_path.is_file():
                atomic_write_text(tracker, f"{actor_step}\n")
        return

    rollout_id = int(tracker.read_text())
    for root, step in ((actor_root, actor_step), (critic_root, critic_step)):
        if step != rollout_id:
            logger.warning("Recovering %s from rollout %s to complete PPO checkpoint %s", root, step, rollout_id)
            atomic_write_text(root / CHECKPOINT_TRACKER_FILENAME, f"{rollout_id}\n")
        for iteration, path in _checkpoint_directories(root).items():
            if iteration > rollout_id:
                shutil.rmtree(path)


def _finalize_checkpoint(args, rollout_id: int, retain_interval: int) -> None:
    roots = (Path(args.save), Path(args.critic_save))
    state_path = Path(compute_global_dataset_state_path(args.save, rollout_id=rollout_id))
    if any(read_checkpoint_tracker_iteration(root) != rollout_id for root in roots) or (
        args.rollout_global_dataset and not state_path.is_file()
    ):
        raise RuntimeError(f"Checkpoint {rollout_id} is incomplete. Skipping checkpoint pruning.")

    iterations_by_root = [_checkpoint_directories(root) for root in roots]
    actor, critic = iterations_by_root
    if rollout_id not in actor or rollout_id not in critic:
        raise RuntimeError(f"Checkpoint directories for rollout {rollout_id} are missing. Skipping checkpoint pruning.")

    atomic_write_text(roots[0] / PPO_CHECKPOINT_TRACKER, f"{rollout_id}\n")
    for iteration in sorted(actor.keys() | critic.keys()):
        if iteration >= rollout_id:
            break
        if iteration == 0 or iteration % retain_interval == 0:
            continue
        Path(compute_global_dataset_state_path(args.save, rollout_id=iteration)).unlink(missing_ok=True)
        for checkpoints in iterations_by_root:
            if iteration in checkpoints:
                shutil.rmtree(checkpoints[iteration])


async def train(args, *, disposer: Disposer):
    assert not args.fully_async, "--fully-async requires the async driver: run train_async.py"
    retain_interval = getattr(args, "save_retain_interval", None) if args.use_critic else None
    if retain_interval is not None:
        if (
            args.megatron_config is not None
            or not args.save
            or not args.critic_save
            or Path(args.save).resolve() == Path(args.critic_save).resolve()
        ):
            raise ValueError("PPO checkpoint retention requires separate actor and critic save directories")
        # Megatron would prune the actor and critic independently as each save completes.
        args.save_retain_interval = None
        _recover_checkpoint(args)
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
        force_sync = force_sync or rollout_id == args.num_rollout - 1 or retain_interval is not None

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
        if retain_interval is not None:
            _finalize_checkpoint(args, rollout_id, retain_interval)

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
    args = parse_args()
    asyncio.run(with_disposer(train, args))
