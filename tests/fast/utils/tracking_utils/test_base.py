import logging
from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.tracking_utils.base import PrometheusBackend, TrackingBackend, TrackingManager
from miles.utils.tracking_utils.tracking import BACKEND_REGISTRY
from miles.utils.workers.types import ClusterBackend


@pytest.mark.parametrize("fail", [False, True])
async def test_wandb_receives_the_driver_exit_status(monkeypatch, fail):
    import wandb
    from miles.utils.async_utils import Disposer
    from miles.utils.tracking_utils import tracking
    from miles.utils.tracking_utils.base import WandbBackend

    codes = []
    manager = TrackingManager({})
    manager._backends = [WandbBackend()]
    monkeypatch.setattr(tracking, "_manager", manager)
    monkeypatch.setattr(wandb, "finish", lambda *, exit_code: codes.append(exit_code))

    async def run():
        async with Disposer() as disposer:
            disposer.add_exit(tracking.finish_tracking_on_exit)
            if fail:
                raise RuntimeError("training failed")

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            await run()
    else:
        await run()
    assert codes == [int(fail)]


class _RecordingBackend(TrackingBackend):
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        type(self).inits.append(args)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        type(self).logs.append(metrics)

    def finish(self) -> None:
        return


class _RayOnlyBackend(_RecordingBackend):
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    @classmethod
    def is_supported(cls, args) -> bool:
        return False


class _PickyBackend(_RecordingBackend):
    inits: list[Any] = []
    logs: list[dict[str, Any]] = []

    @classmethod
    def is_supported(cls, args) -> bool:
        raise AssertionError("is_supported must not be consulted for a disabled backend")


class TestPrometheusBackendIsSupported:
    @pytest.mark.parametrize(
        ("cluster_backend", "expected"), [(ClusterBackend.RAY, True), (ClusterBackend.KUBERNETES, False)]
    )
    def test_follows_the_cluster_backend(self, cluster_backend, expected):
        """The collector is a named Ray actor, so it exists only under the ray backend."""
        assert PrometheusBackend.is_supported(SimpleNamespace(cluster_backend=cluster_backend.value)) is expected


class TestTrackingManagerInit:
    def setup_method(self):
        for cls in (_RecordingBackend, _RayOnlyBackend, _PickyBackend):
            cls.inits = []
            cls.logs = []

    def test_initialises_an_enabled_supported_backend(self):
        """The ordinary path is untouched: an enabled backend still gets built."""
        manager = TrackingManager({"recording": (_RecordingBackend, "use_recording")})
        args = SimpleNamespace(use_recording=True)

        manager.init(args)

        assert _RecordingBackend.inits == [args]

    def test_skips_an_enabled_unsupported_backend(self, caplog):
        """An enabled but unsupported backend is dropped whole, not half-initialised."""
        manager = TrackingManager({"ray_only": (_RayOnlyBackend, "use_ray_only")})

        with caplog.at_level(logging.WARNING):
            manager.init(SimpleNamespace(use_ray_only=True))

        assert _RayOnlyBackend.inits == []
        assert "ray_only" in caplog.text

    def test_unsupported_backend_receives_no_metrics(self):
        """Dropping the backend also removes it from the later log fan-out."""
        manager = TrackingManager(
            {
                "ray_only": (_RayOnlyBackend, "use_ray_only"),
                "recording": (_RecordingBackend, "use_recording"),
            }
        )
        manager.init(SimpleNamespace(use_ray_only=True, use_recording=True))

        manager.log({"loss": 1.0}, step=0)

        assert _RecordingBackend.logs == [{"loss": 1.0}]
        assert _RayOnlyBackend.logs == []

    def test_registry_gates_the_real_prometheus_entry(self):
        """The registry entry itself is gated, not only a synthetic backend."""
        manager = TrackingManager({"prometheus": BACKEND_REGISTRY["prometheus"]})

        manager.init(SimpleNamespace(use_prometheus=True, cluster_backend=ClusterBackend.KUBERNETES.value))

        assert manager._backends == []

    def test_disabled_backend_is_never_asked_for_support(self):
        """The enable flag decides first, so a support check may read backend-only args."""
        manager = TrackingManager({"picky": (_PickyBackend, "use_picky")})

        manager.init(SimpleNamespace(use_picky=False))

        assert _PickyBackend.inits == []
