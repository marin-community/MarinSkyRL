from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("harbor")

from harbor.environments.base import BaseEnvironment

from marinskyrl import iris_harbor_environment as iris_environment


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(BaseEnvironment, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://iris-controller:10000")
    result = iris_environment.IrisEnvironment()
    result.session_id = "snowball/test"
    result.logger = Mock()
    result.task_env_config = SimpleNamespace(
        cpus=2,
        memory_mb=4096,
        storage_mb=8192,
        docker_image="ghcr.io/marin-community/snowball-test:sha",
    )
    return result


def test_uses_ambient_controller_url(environment):
    assert environment._cluster is None
    assert environment._controller_url == "http://iris-controller:10000"


def test_connection_arguments_are_exclusive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(BaseEnvironment, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.delenv("IRIS_CONTROLLER_URL", raising=False)

    with pytest.raises(ValueError, match="IRIS_CONTROLLER_URL"):
        iris_environment.IrisEnvironment()
    with pytest.raises(ValueError, match="exactly one"):
        iris_environment.IrisEnvironment(cluster="rno2a", controller_url="http://iris-controller:10000")


def test_endpoint_lives_until_environment_stops(environment, monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    client = Mock()
    job = Mock(job_id="job-1")
    client.submit.return_value = job
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(return_value=client))
    monkeypatch.setattr(iris_environment, "ControllerServiceClientSync", Mock())
    monkeypatch.setattr(environment, "_wait_for_running", Mock(return_value="task-1"))

    environment._start_sync()

    endpoint.close.assert_not_called()
    assert environment._endpoint is endpoint
    client.submit.assert_called_once()
    assert client.submit.call_args.kwargs["task_image"] == environment.task_env_config.docker_image

    environment._stop_sync()

    job.terminate.assert_called_once_with()
    client.shutdown.assert_called_once_with()
    endpoint.close.assert_called_once_with()
    assert environment._endpoint is None


def test_start_failure_closes_endpoint(environment, monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(side_effect=RuntimeError("cannot connect")))

    with pytest.raises(RuntimeError, match="cannot connect"):
        environment._start_sync()

    endpoint.close.assert_called_once_with()
    assert environment._endpoint is None
