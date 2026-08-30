import asyncio
import signal
from unittest.mock import Mock

import pytest
import ray
from ray.util.queue import Queue

from skyrl_train.entrypoints import ray_lifecycle
from skyrl_train.entrypoints.main_base import EntrypointSupervisor, resolve_entrypoint_node_id


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


def test_local_ray_owner_returns_through_normal_process_exit(monkeypatch):
    exit_process = Mock()
    monkeypatch.delenv("SKYRL_RAY_CLUSTER_OWNER", raising=False)
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)

    ray_lifecycle.exit_without_ray_destructors()

    exit_process.assert_not_called()


@pytest.mark.usefixtures("ray_init")
def test_entrypoint_node_resolution_selects_live_matching_node():
    node_ip = ray.util.get_node_ip_address()

    node_id = resolve_entrypoint_node_id(node_ip)

    matching_nodes = {node["NodeID"] for node in ray.nodes() if node["Alive"] and node["NodeManagerAddress"] == node_ip}
    assert node_id in matching_nodes


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
