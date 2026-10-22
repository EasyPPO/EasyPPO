"""Critic training runs get distinct wandb names and never fail because of wandb."""

import importlib.util
import shlex
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

wandb = pytest.importorskip("wandb")

from miles import critic_recipe  # noqa: E402
from miles.utils.tracking_utils import wandb_utils  # noqa: E402
from miles.utils.tracking_utils.base import WandbBackend  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_miles_train():
    spec = importlib.util.spec_from_file_location("miles_train", ROOT / "scripts/agentic_critic/miles_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_critics_get_distinct_run_names_in_one_group():
    miles_train = load_miles_train()
    names = {
        mode: miles_train.wandb_names(Namespace(mode=mode, wandb_group="cmp-1", wandb_run_name=""))
        for mode in ("full_history", "actor_summary")
    }
    assert names["full_history"] == ("cmp-1", "cmp-1-train-full_history")
    assert names["actor_summary"] == ("cmp-1", "cmp-1-train-actor_summary")
    group, name = miles_train.wandb_names(Namespace(mode="full_history", wandb_group="", wandb_run_name=""))
    assert group.startswith("critic-") and name == f"{group}-train-full_history"


def recipe_args(**kwargs):
    return critic_recipe.ScriptArgs(run_id="critic", **kwargs)


def test_recipe_passes_group_and_name(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret key")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    argv = shlex.split(
        critic_recipe._wandb_args(
            recipe_args(
                wandb_group="cmp-1",
                wandb_run_name="cmp-1-train-full_history",
                wandb_project="proj",
                wandb_entity="team",
            )
        )
    )
    values = {
        argv[i]: argv[i + 1]
        for i in range(len(argv) - 1)
        if argv[i].startswith("--") and not argv[i + 1].startswith("--")
    }
    assert values["--wandb-group"] == "cmp-1"
    assert values["--wandb-run-name"] == "cmp-1-train-full_history"
    assert values["--wandb-project"] == "proj" and values["--wandb-team"] == "team"
    assert values["--wandb-key"] == "secret key"
    assert "--use-wandb" in argv and "--disable-wandb-random-suffix" in argv


def test_recipe_tracking_switches(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert critic_recipe._wandb_args(recipe_args(wandb_group="g")) == ""
    monkeypatch.setenv("WANDB_MODE", "offline")
    argv = critic_recipe._wandb_args(recipe_args(wandb_group="g"))
    assert "--wandb-mode offline" in argv and "--wandb-key" not in argv
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert critic_recipe._wandb_args(recipe_args(wandb_group="g")) == ""


def test_primary_init_uses_run_name(monkeypatch):
    seen = {}
    monkeypatch.setattr(wandb, "init", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(wandb, "define_metric", lambda *a, **k: None)
    monkeypatch.setattr(wandb, "run", SimpleNamespace(id="abc"))
    args = Namespace(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="cmp-1",
        wandb_run_name="cmp-1-train-actor_summary",
        wandb_team=None,
        wandb_project="proj",
        wandb_run_id=None,
        wandb_dir=None,
        env_report=None,
    )
    wandb_utils.init_wandb_primary(args)
    assert seen["group"] == "cmp-1" and seen["name"] == "cmp-1-train-actor_summary"
    assert args.wandb_run_id == "abc"


def test_wandb_failures_never_raise(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("wandb down")

    monkeypatch.setattr(wandb_utils, "init_wandb_primary", boom)
    args = Namespace(wandb_run_id="stale")
    backend = WandbBackend()
    backend.init(args, primary=True)
    assert args.wandb_run_id is None  # secondaries must not attach to a run that does not exist
    backend.log({"train/loss": 1.0}, step=1)
    backend.define_step_key_metric_group("train", "train/step")
    backend.finish()

    # An initialised run whose log/finish start failing mid-training.
    healthy = WandbBackend()
    monkeypatch.setattr(wandb, "run", SimpleNamespace(id="abc"))
    monkeypatch.setattr(wandb, "log", boom)
    monkeypatch.setattr(wandb, "define_metric", boom)
    monkeypatch.setattr(wandb, "finish", boom)
    for step in range(3):
        healthy.log({"train/loss": 1.0}, step=step)
    healthy.define_step_key_metric_group("train", "train/step")
    healthy.finish()
    assert healthy._warned == {"log", "define_metric", "finish"}


def test_secondary_without_run_logs_nothing(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    calls = []
    monkeypatch.setattr(wandb, "log", lambda *a, **k: calls.append(a))
    backend = WandbBackend()
    backend.init(Namespace(wandb_run_id=None), primary=False)
    backend.log({"x": 1})
    assert calls == []
