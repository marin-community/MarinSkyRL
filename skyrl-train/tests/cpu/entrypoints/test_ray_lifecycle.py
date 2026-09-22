import asyncio
import contextlib
import signal
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


def test_shutdown_ray_disconnects_locally_owned_cluster(monkeypatch):
    shutdown = Mock()
    monkeypatch.delenv("SKYRL_RAY_CLUSTER_OWNER", raising=False)
    monkeypatch.setattr(ray_lifecycle.ray, "shutdown", shutdown)

    ray_lifecycle.shutdown_ray()

    shutdown.assert_called_once_with()


def test_shutdown_ray_leaves_externally_owned_cluster_connected(monkeypatch):
    shutdown = Mock()
    unregister = Mock()
    monkeypatch.setenv("SKYRL_RAY_CLUSTER_OWNER", "iris-task-runtime")
    monkeypatch.setattr(ray_lifecycle.ray, "shutdown", shutdown)
    monkeypatch.setattr(ray_lifecycle.atexit, "unregister", unregister)

    ray_lifecycle.shutdown_ray()

    shutdown.assert_not_called()
    unregister.assert_called_once_with(shutdown)


def test_external_ray_owner_exits_without_ray_destructors(monkeypatch):
    exit_process = Mock()
    monkeypatch.setenv("SKYRL_RAY_CLUSTER_OWNER", "iris-task-runtime")
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)

    ray_lifecycle.exit_without_ray_destructors()

    exit_process.assert_called_once_with(0)


def test_external_ray_owner_preserves_termination_exit_code(monkeypatch):
    exit_process = Mock()
    monkeypatch.setenv("SKYRL_RAY_CLUSTER_OWNER", "iris-task-runtime")
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)

    ray_lifecycle.exit_without_ray_destructors(128 + signal.SIGTERM)

    exit_process.assert_called_once_with(128 + signal.SIGTERM)


def test_external_ray_owner_flushes_logs_before_immediate_exit(monkeypatch):
    exit_process = Mock()
    stdout = Mock()
    stderr = Mock()
    monkeypatch.setenv("SKYRL_RAY_CLUSTER_OWNER", "iris-task-runtime")
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)
    monkeypatch.setattr(ray_lifecycle.sys, "stdout", stdout)
    monkeypatch.setattr(ray_lifecycle.sys, "stderr", stderr)

    ray_lifecycle.exit_without_ray_destructors(1)

    stdout.flush.assert_called_once_with()
    stderr.flush.assert_called_once_with()
    exit_process.assert_called_once_with(1)


def test_local_ray_owner_returns_through_normal_process_exit(monkeypatch):
    exit_process = Mock()
    monkeypatch.delenv("SKYRL_RAY_CLUSTER_OWNER", raising=False)
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)

    ray_lifecycle.exit_without_ray_destructors()

    exit_process.assert_not_called()


def test_runner_evidence_rejection_happens_before_ray_initialization(monkeypatch):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.use_tis = True
    cfg.trainer.algorithm.tis_imp_ratio_cap = 2.0
    initialize_ray = Mock()
    monkeypatch.setattr(trainer_utils, "initialize_ray", initialize_ray)

    with pytest.raises(ValueError, match="mini-swe cannot supply exact sampled completion"):
        run_ray_driver(cfg, Mock(), TrajectoryRunnerMode.MINI_SWE)

    initialize_ray.assert_not_called()


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


def test_generate_only_skips_training_validation(monkeypatch):
    cfg = get_default_config()
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.offload_optimizer_during_rollouts = True
    validation_complete = RuntimeError("generation validation complete")
    monkeypatch.setattr(
        main_base,
        "validate_trajectory_runner_capabilities",
        Mock(side_effect=validation_complete),
    )

    with pytest.raises(RuntimeError, match="generation validation complete"):
        run_ray_driver(
            cfg,
            Mock(),
            TrajectoryRunnerMode.SKYRL_GYM,
            operation=EntrypointOperation.GENERATE,
        )


@pytest.mark.usefixtures("ray_init")
def test_entrypoint_node_resolution_selects_live_matching_node():
    node_ip = ray.util.get_node_ip_address()

    node_id = resolve_entrypoint_node_id(node_ip)

    assert node_id == ray.get_runtime_context().get_node_id()


@pytest.mark.usefixtures("ray_init")
def test_entrypoint_node_resolution_rejects_unknown_node():
    with pytest.raises(ValueError, match="Expected exactly one live Ray node"):
        resolve_entrypoint_node_id("192.0.2.1")


@pytest.mark.usefixtures("ray_init")
def test_entrypoint_supervisor_allows_remote_cleanup_before_returning():
    events = Queue()
    entrypoint_ref = _wait_until_cancelled.remote(events)
    assert events.get(timeout=10) == "started"
    supervisor = EntrypointSupervisor(shutdown_timeout_seconds=10)

    supervisor.request_termination(signal.SIGTERM)
    exit_code = supervisor.wait(entrypoint_ref)

    assert exit_code == 128 + signal.SIGTERM
    assert events.get(timeout=10) == "stopped"
