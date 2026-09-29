"""Shared path and image configuration for the 8- and 32-GPU B200 launchers."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
REMOTE_CONFIG_ENV = "EASYPPO_MODAL_CONFIG_JSON"


@dataclass(frozen=True)
class ModalConfig:
    image: str
    source_root: str
    volume_name: str
    volume_mount: str
    model_dir: str
    output_dir: str
    hf_cache_dir: str
    vllm_cache_dir: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("source_root", "volume_mount"):
            path = PurePosixPath(getattr(self, name))
            if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
                raise ValueError(f"{name} must be an absolute non-root path without '..'")
        source, mount = PurePosixPath(self.source_root), PurePosixPath(self.volume_mount)
        if source.is_relative_to(mount) or mount.is_relative_to(source):
            raise ValueError("source_root and volume_mount must not overlap")
        for name in ("model_dir", "output_dir", "hf_cache_dir", "vllm_cache_dir"):
            path = PurePosixPath(getattr(self, name))
            if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
                raise ValueError(f"{name} must be a non-empty relative path inside volume_mount")

    def volume_path(self, relative):
        return str(PurePosixPath(self.volume_mount) / relative)

    @property
    def model_path(self):
        return self.volume_path(self.model_dir)

    def output_path(self, run_name):
        return str(PurePosixPath(self.volume_path(self.output_dir)) / run_name)

    def container_env(self):
        return {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": self.source_root,
            "HF_HOME": self.volume_path(self.hf_cache_dir),
            "VLLM_CACHE_ROOT": self.volume_path(self.vllm_cache_dir),
            "WANDB_MODE": "online",
            # Freeze the local selection for every remote container and Ray worker.
            # A custom config file may live outside the uploaded checkout.
            REMOTE_CONFIG_ENV: json.dumps(asdict(self)),
        }


def load_modal_config():
    """Read a local YAML file, or the exact settings forwarded to a container."""
    if REMOTE_CONFIG_ENV in os.environ:
        values = json.loads(os.environ[REMOTE_CONFIG_ENV])
    else:
        path = Path(os.environ.get("EASYPPO_MODAL_CONFIG", DEFAULT_CONFIG)).expanduser()
        with path.open() as stream:
            values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("Modal configuration must be a mapping")
    return ModalConfig(**values)
