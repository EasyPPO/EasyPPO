# Copyright 2026 EasyPPO contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Check timeout-only continuation with no remote calls or training processes."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

modal = pytest.importorskip("modal")


@pytest.fixture
def launcher(monkeypatch):
    # Importing the launcher constructs SDK objects only; never hydrate or run them.
    for key in ("WANDB_API_KEY", "REGISTRY_USERNAME", "REGISTRY_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    path = Path(__file__).resolve().parents[2] / "modal_launchers/b200.py"
    spec = importlib.util.spec_from_file_location("modal_b200", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_RUN_ID", "existing-run")
    monkeypatch.setenv("WANDB_RESUME", "must")
    module.train = SimpleNamespace(remote=Mock())
    return module


def test_timeout_then_success_preserves_training_arguments(launcher):
    launcher.train.remote.side_effect = [modal.exception.FunctionTimeoutError("timeout"), None]
    launcher.resume(model_path="/models/base", run_name="original", max_timeout_restarts=2)
    assert launcher.train.remote.call_args_list == [
        call(model_path="/models/base", run_name="original"),
        call(model_path="/models/base", run_name="original"),
    ]


@pytest.mark.parametrize("limit", [0, 2])
def test_timeout_restart_limit(launcher, limit):
    error = modal.exception.FunctionTimeoutError("timeout")
    launcher.train.remote.side_effect = error
    with pytest.raises(modal.exception.FunctionTimeoutError) as caught:
        launcher.resume(max_timeout_restarts=limit)
    assert caught.value is error
    assert launcher.train.remote.call_count == limit + 1


@pytest.mark.parametrize(
    "error",
    [RuntimeError("training error"), subprocess.CalledProcessError(1, "trainer"), TimeoutError("other timeout")],
)
def test_other_failures_are_not_retried(launcher, error):
    launcher.train.remote.side_effect = error
    with pytest.raises(type(error)) as caught:
        launcher.resume()
    assert caught.value is error
    launcher.train.remote.assert_called_once()


@pytest.mark.parametrize("mode", ["must", "allow"])
def test_success_is_not_restarted(launcher, monkeypatch, mode):
    monkeypatch.setenv("WANDB_RESUME", mode)
    launcher.resume()
    launcher.train.remote.assert_called_once()


@pytest.mark.parametrize("key", ["WANDB_API_KEY", "WANDB_RUN_ID", "WANDB_RESUME"])
def test_missing_wandb_settings_fail_before_remote_call(launcher, monkeypatch, key):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match=key):
        launcher.resume()
    launcher.train.remote.assert_not_called()


def test_negative_limit_fails_before_remote_call(launcher):
    with pytest.raises(ValueError, match="non-negative"):
        launcher.resume(max_timeout_restarts=-1)
    launcher.train.remote.assert_not_called()
