import json
import logging
import shlex
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils import common
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, get_mooncake_object_store_args


@pytest.mark.parametrize("conflicting_override", [False, True])
def test_volume_access_configuration_reaches_workers_without_override(monkeypatch, conflicting_override):
    monkeypatch.setenv("MILES_VOLUME_IO_ROOT", "/checkpoints")
    monkeypatch.setenv("MILES_VOLUME_IO_LOCK", "/tmp/volume.lock")
    request = SimpleNamespace(train_backend_fsdp=True, extra_env_vars={})
    config = SimpleNamespace(cuda_core_dump=False, extra_env_vars=json.dumps(
        {"MILES_VOLUME_IO_LOCK": "/wrong.lock"} if conflicting_override else {}
    ))
    if conflicting_override:
        with pytest.raises(ValueError, match="cannot override"):
            common.train_env_vars(request, {}, config=config)
    else:
        env = common.train_env_vars(request, {}, config=config)
        assert env["MILES_VOLUME_IO_ROOT"] == "/checkpoints"
        assert env["MILES_VOLUME_IO_LOCK"] == "/tmp/volume.lock"


class TestGetMooncakeObjectStoreArgs:
    def test_a_remote_master_host_reaches_the_serialized_store_address(self) -> None:
        """A split deployment connects to the master host supplied by its driving release."""
        argv = shlex.split(get_mooncake_object_store_args(master_port=61234, master_host="mooncake.run.svc"))

        kwargs = json.loads(argv[argv.index(MOONCAKE_INIT_KWARGS_FLAG) + 1])

        assert kwargs["master_server_address"] == "mooncake.run.svc:61234"


class TestGetDefaultWandbArgs:
    def test_environment_auth_does_not_serialize_the_key(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "test-key-must-not-appear-in-command")
        monkeypatch.setenv("MILES_WANDB_KEY_FROM_ENV", "1")
        command = common.get_default_wandb_args("tests/e2e/test_run.py", run_id="personal")
        assert "--use-wandb" in command
        assert "--wandb-key" not in command
        assert "test-key-must-not-appear-in-command" not in command

    def test_missing_credentials_configures_logging_before_reporting_the_skip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher must make its skip message visible before it emits that message."""
        events: list[str] = []
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setattr(
            "miles.utils.logging_utils.configure_logger_raw", lambda name: events.append(f"configure:{name}")
        )

        class RecordingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                events.append(record.getMessage())

        handler = RecordingHandler()
        common.logger.addHandler(handler)
        common.logger.setLevel(logging.INFO)
        try:
            assert common.get_default_wandb_args("tests/e2e/test_run.py") == ""
        finally:
            common.logger.removeHandler(handler)

        assert events == ["configure:launcher", "Skip wandb configuration since WANDB_API_KEY is not found"]
