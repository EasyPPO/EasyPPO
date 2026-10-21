"""Launch Qwen3.5 critic training from stored trajectories."""

import argparse
import json
import os
import shlex
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "rollouts", "output", "model-dir", "prompt-data"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mode", choices=("full_history", "actor_summary", "shared"), required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--save-interval", type=int, default=40)
    parser.add_argument("--problems-per-batch", type=int, default=8)
    parser.add_argument("--episodes-per-problem", type=int, default=16)
    parser.add_argument("--mini-batches", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.steps, args.gpus, args.tensor_parallel, args.save_interval, args.max_tokens) < 1:
        raise ValueError("steps, GPUs, parallelism and save interval must be positive")
    batch_size = args.problems_per_batch * args.episodes_per_problem
    if min(args.problems_per_batch, args.episodes_per_problem, args.mini_batches) < 1:
        raise ValueError("Batch dimensions must be positive")
    if batch_size % args.mini_batches or args.mini_batches > batch_size:
        raise ValueError("Episode batch must divide into the requested nonempty mini-batches")
    if args.gpus % args.tensor_parallel:
        raise ValueError("GPUs must be divisible by tensor parallelism")
    # The launcher assembles shell strings. Refuse ambiguous paths instead
    # of silently interpreting shell metacharacters or splitting a path.
    for path in (args.model, args.rollouts, args.output, args.model_dir, args.prompt_data):
        if shlex.quote(str(path)) != str(path):
            raise ValueError(f"Miles recipe requires shell-safe paths: {path}")
    for batch in range(args.steps):
        if not (args.rollouts / f"rollout_{batch}.pt").is_file():
            raise FileNotFoundError(f"Missing replay batch {batch}")
    manifest = json.loads((args.rollouts / "manifest.json").read_text())
    config = manifest["args"]
    if (config["problems_per_step"], config["episodes_per_problem"]) != (
        args.problems_per_batch,
        args.episodes_per_problem,
    ):
        raise ValueError("Replay dimensions do not match the configured prompts and episodes")
    rows = {int(row["step"]): row for row in manifest["steps"]}
    for step in range(args.steps):
        row = rows.get(step)
        if row is None or row["episodes"] != batch_size or len(set(row["problems"])) != args.problems_per_batch:
            raise ValueError(f"Replay batch {step} does not contain the configured complete episode groups")
    batches = manifest["steps"][: args.steps]
    if len(batches) != args.steps or any(
        batch["step"] != index or batch["episodes"] != batch_size or len(batch["problems"]) != args.problems_per_batch
        for index, batch in enumerate(batches)
    ):
        raise ValueError("Replay manifest does not contain the requested complete episode batches")
    if not args.prompt_data.is_file() or not (args.model / "config.json").is_file():
        raise FileNotFoundError("Prompt data or model config is missing")

    from miles import critic_recipe as recipe

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_name = "Qwen3.5-9B-FrontierCS-SFT-DSv3.1"
    alias = args.model_dir / model_name
    if alias.exists() and alias.resolve() != args.model.resolve():
        raise ValueError(f"Model cache points at a different checkpoint: {alias}")
    if not alias.exists():
        alias.symlink_to(args.model.resolve(), target_is_directory=True)
    output = args.output / "critic"
    if (output / "checkpoints_critic/latest_checkpointed_iteration.txt").exists():
        raise FileExistsError("Output already contains a trained critic; use a fresh OUTPUT")
    if (args.output / "run.json").exists() or any(args.output.glob("checkpoint-*")):
        raise FileExistsError("Output contains checkpoints from another trainer; use a fresh OUTPUT")
    mode = "shared" if args.mode == "actor_summary" else args.mode
    extra = (
        f"--load-debug-rollout-data {args.rollouts}/rollout_{{rollout_id}}.pt "
        "--debug-train-only --no-save-optim "
        "--use-precision-aware-optimizer --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 "
        f"--save-debug-train-data {output}/dump/{{rollout_id}}_{{rank}}.pt"
    )
    recipe_args = recipe.ScriptArgs(
        run_id="critic",
        prompt_data=str(args.prompt_data),
        model_name=model_name,
        model_dir=str(args.model_dir),
        data_dir=str(args.rollouts),
        output_dir=str(args.output),
        num_gpus_per_node=args.gpus,
        tensor_model_parallel_size=args.tensor_parallel,
        rollout_batch_size=args.problems_per_batch,
        n_samples_per_prompt=args.episodes_per_problem,
        critic_mini_batch_size=batch_size // args.mini_batches,
        num_rollout=args.steps,
        num_critic_only_steps=args.steps,
        save_interval=args.save_interval,
        max_tokens_per_gpu=args.max_tokens,
        agentic_generate_function_path="agentic_critic_runtime.miles_fcs.generate",
        critic_context_mode=mode,
        megatron_path=os.environ["MEGATRON_PATH"],
        extra_args=extra,
    )
    recipe_args.create_backend().convert_checkpoint(
        model_name=model_name,
        megatron_model_type=recipe_args.megatron_model_type,
        num_gpus_per_node=args.gpus,
        dir_dst=str(args.model_dir),
        hf_checkpoint=str(args.model),
        megatron_path=recipe_args.megatron_path,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "miles_run.json").write_text(
        json.dumps(
            {
                "backend": "miles",
                "mode": args.mode,
                "miles_mode": mode,
                "rollouts": str(args.rollouts),
                "logical_batches": args.steps,
                "episodes_per_batch": batch_size,
                "problems_per_batch": args.problems_per_batch,
                "episodes_per_problem": args.episodes_per_problem,
                "mini_batches": args.mini_batches,
                "optimizer_updates": args.steps * args.mini_batches,
                "checkpoints": str(output / "checkpoints_critic"),
                "dump_dir": str(output / "dump"),
                "optimizer_saved": False,
            },
            indent=2,
        )
        + "\n"
    )
    recipe.execute(recipe_args)


if __name__ == "__main__":
    main()
