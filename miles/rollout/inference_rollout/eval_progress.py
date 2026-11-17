"""Atomic per-sample progress for an initial evaluation with unchanged weights."""

import hashlib
import json
import logging
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from miles.utils.file_utils import atomic_torch_save
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _model_identity(args):
    result = {}
    for name in ("hf_checkpoint", "load", "ref_load"):
        value = getattr(args, name, None)
        if not value:
            continue
        root = Path(value)
        # Stat checkpoint files rather than reading hundreds of GB of weights.
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        result[name] = [str(root), [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]]
    return result


class InitialEvalProgress:
    def __init__(self, directory, args, dataset_cfg, sampling_params):
        config = {
            name: value
            for name, value in vars(args).items()
            if (name.startswith("terminal_") and name not in ("terminal_memory_mb", "terminal_max_retries"))
            or name
            in (
                "custom_generate_function_path",
                "rollout_function_path",
                "custom_rm_path",
                "rm_type",
                "apply_chat_template",
                "apply_chat_template_kwargs",
                "chat_template_path",
                "rollout_seed",
                "sglang_enable_deterministic_inference",
                "rollout_max_context_len",
            )
        }
        template = getattr(args, "chat_template_path", None)
        root = Path(__file__).resolve().parents[3]
        sources = sorted((root / "agentic_critic").glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
        code = [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
        identity = dict(
            format=1,
            model=_model_identity(args),
            config=config,
            dataset=asdict(dataset_cfg),
            sampling=sampling_params,
            template=Path(template).read_text() if template else None,
            code=code,
        )
        self.directory = Path(directory) / _digest(identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task_root = Path(getattr(args, "terminal_tasks_dir", ""))
        self._task_digests = {}

    def _task_digest(self, metadata):
        if not metadata or not metadata.get("task_path"):
            return None
        task = self.task_root / metadata["task_path"]
        if task not in self._task_digests:
            config = tomllib.loads((task / "task.toml").read_text())
            # A new Modal attempt has a new judge endpoint/token; that transport
            # change does not change the problem, verifier or model being scored.
            env = config.get("verifier", {}).get("env", {})
            env.pop("JUDGE_URL", None)
            env.pop("FRONTIERCS_JUDGE_TOKEN", None)
            files = sorted(p for p in task.rglob("*") if p.is_file() and p.name != "task.toml")
            self._task_digests[task] = _digest(
                dict(
                    config=config,
                    files=[(str(p.relative_to(task)), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
                )
            )
        return self._task_digests[task]

    def path(self, sample, sampling_params):
        # The index distinguishes independent repeats of the same task. Routing
        # keys and KV namespaces are invocation-specific and not cache identity.
        identity = dict(
            index=sample.index,
            prompt=sample.prompt,
            tokens=sample.tokens,
            metadata=sample.metadata,
            label=sample.label,
            sampling=sampling_params,
            task=self._task_digest(sample.metadata),
        )
        return self.directory / f"{sample.index}-{_digest(identity)}.pt"

    def load(self, path, *, namespace):
        if not path.is_file():
            return None
        try:
            # These are local artifacts written by this run, using the same
            # Sample serialization as rollout checkpoints (including arrays).
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples = [Sample.from_dict(item) for item in payload["samples"]]
            self._validate(samples)
            multiple = payload["multiple"]
        except Exception:
            logger.warning("Ignoring incomplete initial eval result %s", path, exc_info=True)
            return None
        for sample in samples:
            sample.kv_cache_namespace = namespace
        return samples if multiple else samples[0]

    def save(self, path, output):
        samples = output if isinstance(output, list) else [output]
        self._validate(samples)
        atomic_torch_save(path, dict(multiple=isinstance(output, list), samples=[s.to_dict() for s in samples]))

    @staticmethod
    def _validate(samples):
        if not samples or any(
            s.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or s.reward is None for s in samples
        ):
            raise ValueError("Only completed, scored evaluation samples can be persisted")
