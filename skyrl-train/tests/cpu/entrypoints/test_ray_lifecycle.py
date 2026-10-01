import asyncio
import contextlib
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest
import ray
from ray.util.queue import Queue

from skyrl_train.config.trajectory_runner_capabilities import EntrypointOperation, TrajectoryRunnerMode
from skyrl_train.entrypoints import ray_lifecycle
from skyrl_train.entrypoints.main_base import EntrypointSupervisor, resolve_entrypoint_node_id, run_ray_driver
from skyrl_train.config.utils import get_default_config
from skyrl_train import telemetry
from skyrl_train.entrypoints import main_base
from skyrl_train.utils import utils as trainer_utils


@ray.remote
def _wait_until_cancelled(events: Queue) -> None:
    async def wait() -> None:
        events.put("started")
        try:
            await asyncio.Event().wait()
        finally:
            events.put("stopped")

    asyncio.run(wait())


EXTERNAL_OWNER = "iris-task-runtime"


class _Stream:
    def __init__(self, name: str, events: list):
        self._name = name
        self._events = events

    def flush(self) -> None:
        self._events.append(f"{self._name}.flush")


@pytest.mark.parametrize(
    ("owner", "exit_args", "expected_events"),
    [
        pytest.param(None, (), ["ray.shutdown"], id="local-owner-shuts-down-and-returns"),
        pytest.param(
            EXTERNAL_OWNER,
            (),
            ["atexit.unregister(ray.shutdown)", "stdout.flush", "stderr.flush", ("os._exit", 0)],
            id="external-owner-exits-without-destructors",
        ),
        pytest.param(
            EXTERNAL_OWNER,
            (128 + signal.SIGTERM,),
            ["atexit.unregister(ray.shutdown)", "stdout.flush", "stderr.flush", ("os._exit", 128 + signal.SIGTERM)],
            id="external-owner-preserves-termination-code",
        ),
    ],
)
def test_ray_teardown_follows_the_cluster_owner(monkeypatch, owner, exit_args, expected_events):
    events = []

    def shutdown():
        events.append("ray.shutdown")

    def unregister(function):
        events.append("atexit.unregister(ray.shutdown)" if function is shutdown else ("atexit.unregister", function))

    if owner is None:
        monkeypatch.delenv("SKYRL_RAY_CLUSTER_OWNER", raising=False)
    else:
        monkeypatch.setenv("SKYRL_RAY_CLUSTER_OWNER", owner)
    monkeypatch.setattr(ray_lifecycle.ray, "shutdown", shutdown)
    monkeypatch.setattr(ray_lifecycle.atexit, "unregister", unregister)
    monkeypatch.setattr(ray_lifecycle.sys, "stdout", _Stream("stdout", events))
    monkeypatch.setattr(ray_lifecycle.sys, "stderr", _Stream("stderr", events))
    monkeypatch.setattr(ray_lifecycle.os, "_exit", lambda code: events.append(("os._exit", code)))

    ray_lifecycle.shutdown_ray()
    ray_lifecycle.exit_without_ray_destructors(*exit_args)

    assert events == expected_events


def test_runner_evidence_rejection_happens_before_ray_initialization():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pytest
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.main_base import run_ray_driver
from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode

cfg = get_default_config()
cfg.trainer.logger = "console"
cfg.trainer.algorithm.use_tis = True
cfg.trainer.algorithm.tis_imp_ratio_cap = 2.0
with pytest.raises(ValueError, match="mini-swe cannot supply exact sampled completion"):
    run_ray_driver(cfg, None, TrajectoryRunnerMode.MINI_SWE)
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_driver_preserves_remote_exception_before_external_owner_exit(tmp_path, monkeypatch):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    remote_failure = RuntimeError("original remote failure")
    shutdown = Mock()
    immediate_exit = Mock()
    monkeypatch.setenv("SKYRL_DEBUG_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(trainer_utils, "initialize_ray", Mock())
    monkeypatch.setattr(main_base, "validate_trajectory_runner_capabilities", Mock())
    monkeypatch.setattr(main_base.EntrypointSupervisor, "wait", Mock(side_effect=remote_failure))
    monkeypatch.setattr(ray_lifecycle, "shutdown_ray", shutdown)
    monkeypatch.setattr(ray_lifecycle, "exit_without_ray_destructors", immediate_exit)
    monkeypatch.setattr(telemetry, "process_telemetry", lambda _role: contextlib.nullcontext())

    with pytest.raises(RuntimeError, match="original remote failure"):
        run_ray_driver(cfg, Mock(), TrajectoryRunnerMode.SKYRL_GYM)

    shutdown.assert_called_once_with()
    immediate_exit.assert_called_once_with(1)
    receipts = list((tmp_path / "outcomes").glob("*.exception.json"))
    assert len(receipts) == 1
    assert "original remote failure" in receipts[0].read_text()


def test_generate_only_distillation_rejection_happens_before_ray_initialization(monkeypatch, local_distillation_config):
    cfg = local_distillation_config(get_default_config())
    cfg.trainer.logger = "console"
    initialize_ray = Mock()
    monkeypatch.setattr(trainer_utils, "initialize_ray", initialize_ray)

    with pytest.raises(ValueError, match="training-only"):
        run_ray_driver(
            cfg,
            Mock(),
            TrajectoryRunnerMode.HARBOR,
            operation=EntrypointOperation.GENERATE,
        )

    initialize_ray.assert_not_called()


@pytest.mark.usefixtures("ray_module")
def test_entrypoint_node_resolution_selects_live_matching_node():
    node_ip = ray.util.get_node_ip_address()

    node_id = resolve_entrypoint_node_id(node_ip)

    assert node_id == ray.get_runtime_context().get_node_id()


@pytest.mark.usefixtures("ray_module")
def test_entrypoint_node_resolution_rejects_unknown_node():
    with pytest.raises(ValueError, match="Expected exactly one live Ray node"):
        resolve_entrypoint_node_id("192.0.2.1")


@pytest.mark.usefixtures("ray_module")
def test_entrypoint_supervisor_allows_remote_cleanup_before_returning():
    events = Queue()
    entrypoint_ref = _wait_until_cancelled.remote(events)
    # Timeouts bound hangs only: starting a Ray worker on a loaded host can take tens of seconds.
    assert events.get(timeout=60) == "started"
    supervisor = EntrypointSupervisor(shutdown_timeout_seconds=60)

    supervisor.request_termination(signal.SIGTERM)
    exit_code = supervisor.wait(entrypoint_ref)

    assert exit_code == 128 + signal.SIGTERM
    assert events.get(timeout=60) == "stopped"
