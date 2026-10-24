"""Synchronous, post-update validation using the resident Miles critic."""

import asyncio
from pathlib import Path

from agentic_critic_pipeline.validation import finalize, load_plan
from miles.utils.data import remove_rollout_data_refs
from miles.utils.tracking_utils import tracking


async def evaluate(args, critic_model, rollout_executor, rollout_id):
    plan = load_plan(args.critic_validation_plan)
    batch = rollout_id + 1  # completed logical batches, not optimizer mini-steps
    output = Path(args.critic_validation_output) / f"batch_{batch:04d}"
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Validation directory already exists; use a fresh output: {output}") from None
    mode = "full_history" if args.critic_context_mode == "full_history" else "actor_summary"
    for index, replay in enumerate(plan["replay"]):
        pack = await rollout_executor.get_critic_validation(replay["path"], plan["group_size"])
        try:
            await critic_model.evaluate_critic(
                batch=batch,
                rollout_data_pack=pack,
                output=str(output / f"chunk_{index:04d}" / "{rollout_id}_{rank}.pt"),
            )
        finally:
            remove_rollout_data_refs(args, pack)
    summary = await asyncio.to_thread(finalize, plan, output, mode=mode, batch=batch)
    metrics = {"critic_validation/batch": batch}
    for name, row in summary["ev"].items():
        for key, value in row.items():
            if value is not None:
                metrics[f"critic_validation/{name}/{key}"] = value
    for name, row in summary["jumps"].items():
        for key, value in row.items():
            if value is not None:
                metrics[f"critic_validation/jumps/{name}/{key}"] = value
    tracking.define_step_key_metric_group("critic_validation", "critic_validation/batch")
    tracking.log(args, metrics, step_key="critic_validation/batch")
