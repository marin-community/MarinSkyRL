from unittest.mock import Mock

from skyrl_train.entrypoints import ray_lifecycle


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


def test_local_ray_owner_returns_through_normal_process_exit(monkeypatch):
    exit_process = Mock()
    monkeypatch.delenv("SKYRL_RAY_CLUSTER_OWNER", raising=False)
    monkeypatch.setattr(ray_lifecycle.os, "_exit", exit_process)

    ray_lifecycle.exit_without_ray_destructors()

    exit_process.assert_not_called()
