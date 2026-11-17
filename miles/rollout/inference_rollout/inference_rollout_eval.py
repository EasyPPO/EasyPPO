import asyncio
import copy
import logging
import uuid
from typing import Any

from tqdm import tqdm

from miles.rollout.base_types import stamp_kv_cache_namespace
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.eval_progress import InitialEvalProgress
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    compute_sampling_params,
    generate_and_rm,
)
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def run_eval_datasets(
    state: GenerateState,
    prompt_dataset_cache: dict[Any, Dataset],
    *,
    kv_cache_namespace: str | None = None,
    progress_dir: str | None = None,
) -> dict[str, dict[str, Any]]:
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in args.eval_datasets:
        coros.append(
            eval_rollout_single_dataset(
                state,
                dataset_cfg,
                prompt_dataset_cache,
                kv_cache_namespace=kv_cache_namespace,
                progress_dir=progress_dir,
            )
        )
    results_list = await asyncio.gather(*coros, return_exceptions=True)
    errors = [result for result in results_list if isinstance(result, Exception)]
    if errors:
        raise ExceptionGroup(
            "Evaluation datasets failed; completed samples have been retained when configured", errors
        )
    for result in results_list:
        if isinstance(result, BaseException):
            raise result
    return {k: v for r in results_list for k, v in r.items()}


async def eval_rollout_single_dataset(
    state: GenerateState,
    dataset_cfg: EvalDatasetConfig,
    prompt_dataset_cache: dict[Any, Dataset],
    *,
    kv_cache_namespace: str | None = None,
    progress_dir: str | None = None,
) -> dict[str, dict[str, list[Any]]]:
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if cache_key not in prompt_dataset_cache:
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        prompt_dataset_cache[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = prompt_dataset_cache[cache_key]

    base_sampling_params = compute_sampling_params(
        args,
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
    )
    progress = InitialEvalProgress(progress_dir, args, dataset_cfg, base_sampling_params) if progress_dir else None

    async def generate_sample(sample, sampling_params):
        path = progress.path(sample, sampling_params) if progress else None
        if progress:
            cached = progress.load(path, namespace=kv_cache_namespace)
            if cached is not None:
                return cached
        output = await generate_and_rm(state, sample, sampling_params=sampling_params, evaluation=True)
        if progress:
            progress.save(path, output)
        return output

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset_cfg.select_samples(dataset.samples)):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = dataset_cfg.custom_generate_function_path
            stamp_kv_cache_namespace(sample, namespace=kv_cache_namespace)
            if policy_uses_routing_key(args):
                sample.routing_key = str(uuid.uuid4())
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(asyncio.create_task(generate_sample(sample, sampling_params)))

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    errors = []
    try:
        for completed in asyncio.as_completed(tasks):
            try:
                sample = await completed
            except Exception as exc:
                logger.exception("Evaluation trial failed in %s", dataset_cfg.name)
                errors.append(exc)
                pbar.update(1)
                continue
            if do_print:
                s = (sample[0] if sample else None) if isinstance(sample, list) else sample
                if s is not None:
                    logger.info(
                        "eval_rollout_single_dataset example: %s reward=%s", str(s.prompt) + s.response, s.reward
                    )
                do_print = False
            data.extend(sample if isinstance(sample, list) else [sample])
            pbar.update(1)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
    if errors:
        raise ExceptionGroup(
            f"{len(errors)} evaluation trial(s) failed in {dataset_cfg.name}; no partial score reported", errors
        )

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }
