"""Place persistent Terminus trial owners on the Ray allocation."""

import asyncio
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from agentic_critic.trial_archive import TrialArchiveWriter
from miles.utils.http_utils import distributed_post_actors, use_post_actors


# One compression thread per node/role, not per trial worker.
_ArchiveActor = ray.remote(num_cpus=0, num_gpus=0)(TrialArchiveWriter)


@ray.remote(num_cpus=1, num_gpus=0, max_task_retries=0)
class TerminalWorker:
    def __init__(self, args, *, evaluation, post_actors, archive_writer=None):
        from miles.utils.processing_utils import load_tokenizer

        use_post_actors(post_actors)
        self.evaluation = evaluation
        self.archive_writer = archive_writer
        self.state = SimpleNamespace(
            args=args,
            tokenizer=load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            ),
        )

    async def run(self, sample, sampling_params):
        from agentic_critic.miles_terminal import _generate
        from agentic_critic.terminal_env import finish_write
        from miles.rollout.base_types import GenerateFnInput

        if self.archive_writer is None:
            return await _generate(GenerateFnInput(self.state, sample, sampling_params, evaluation=self.evaluation))

        # Each invocation owns a local root, including startup retries.
        # Do not mutate shared worker state: Ray handles concurrent calls.
        root = Path(tempfile.mkdtemp(prefix="easyppo-trial-", dir="/tmp"))
        state = SimpleNamespace(**vars(self.state), terminal_trial_root=root)
        archive = None
        try:
            output = await _generate(GenerateFnInput(state, sample, sampling_params, evaluation=self.evaluation))
        finally:
            if any(root.iterdir()):
                async def enqueue():
                    return await self.archive_writer.enqueue.remote(str(root))

                # Transfer ownership only after Harbor teardown, including cancellation.
                # The RPC queues a path, not the trace or top-k arrays.
                archive = await finish_write(enqueue())
            else:
                root.rmdir()
        if archive is not None:
            for item in output.samples if isinstance(output.samples, list) else [output.samples]:
                item.metadata["trial_archive"] = archive
                trial_dir = item.metadata.get("trial_dir")
                member = root.name
                if trial_dir is not None and Path(trial_dir).is_relative_to(root):
                    member = (Path(root.name) / Path(trial_dir).relative_to(root)).as_posix()
                    item.metadata["trial_dir"] = f"{archive}#{member}"
                item.metadata["trial_archive_member"] = member
        return output


class TerminalWorkers:
    def __init__(self, args, *, count, evaluation):
        nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
        if not nodes:
            raise RuntimeError("Terminus requires Ray nodes advertising GPU resources")
        if count < 1:
            raise ValueError("Terminus worker count must be positive")
        counts = (args.terminal_train_workers_per_node, args.terminal_eval_workers_per_node)
        if min(counts) < 0:
            raise ValueError("Terminal worker counts must be nonnegative")
        for node in nodes:
            # Miles reserves one CPU alongside each GPU in its placement group.
            required = sum(counts) + min(node["Resources"]["GPU"], args.num_gpus_per_node)
            if node["Resources"].get("CPU", 0) < required:
                raise ValueError(
                    f"Ray node {node['NodeID']} has fewer than {required} CPU slots for terminal workers and GPU bundles"
                )
        thread_env = dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"), "1"
        )
        post_actors = distributed_post_actors()
        self.archive_writers = {}
        if getattr(args, "terminal_archive_trials", False):
            directory = Path(args.terminal_trials_dir).parent / "archives" / ("eval" if evaluation else "train")
            for node in nodes:
                self.archive_writers[node["NodeID"]] = _ArchiveActor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
                ).remote(str(directory))
        self.workers = [
            TerminalWorker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False),
                runtime_env={"env_vars": thread_env},
            ).remote(
                args, evaluation=evaluation, post_actors=post_actors,
                archive_writer=self.archive_writers.get(node["NodeID"]),
            )
            for _ in range(count)
            for node in nodes
        ]
        self.active = [0] * len(self.workers)
        role = "eval" if evaluation else "training"
        logging.getLogger(__name__).info(
            "Requested %d persistent %s workers on each of %d Ray nodes", count, role, len(nodes)
        )

    async def flush_archives(self):
        await asyncio.gather(*(writer.flush.remote() for writer in self.archive_writers.values()))

    async def run(self, sample, sampling_params):
        index = min(range(len(self.workers)), key=self.active.__getitem__)
        self.active[index] += 1
        try:
            result = self.workers[index].run.remote(sample, sampling_params)
            future = asyncio.wrap_future(result.future())
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                ray.cancel(result, force=False)
                await asyncio.gather(future, return_exceptions=True)
                raise
        finally:
            self.active[index] -= 1
