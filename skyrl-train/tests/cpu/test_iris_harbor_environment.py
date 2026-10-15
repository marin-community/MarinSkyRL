from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("harbor")

from harbor.environments.base import BaseEnvironment
from iris.client import Job, Task
from iris.client.workload import TaskStatus
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from iris.rpc import job_pb2

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


@pytest.fixture
def controller(monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    client = Mock()
    job = Mock(spec=Job, job_id="job-1")
    task = Mock(spec=Task, task_id=JobName.from_wire("/user/harbor/0"))
    task.status.return_value = Mock(spec=TaskStatus, state=TaskState.RUNNING, error_message="")
    job.tasks.return_value = [task]
    client.submit.return_value = job
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(return_value=client))
    monkeypatch.setattr(iris_environment, "ControllerServiceClientSync", Mock())
    return SimpleNamespace(endpoint=endpoint, client=client, job=job, task=task)


def test_endpoint_lives_until_environment_stops(environment, controller):
    environment._start_sync()

    controller.endpoint.close.assert_not_called()
    controller.client.submit.assert_called_once()
    submitted = controller.client.submit.call_args.kwargs
    assert submitted["task_image"] == environment.task_env_config.docker_image
    assert submitted["container_profile"] == job_pb2.CONTAINER_PROFILE_SANDBOX
    assert submitted["egress_policy"] == job_pb2.EGRESS_POLICY_INTERNET

    environment._stop_sync()

    controller.job.cancel.assert_called_once_with()
    controller.client.shutdown.assert_called_once_with()
    controller.endpoint.close.assert_called_once_with()


def test_failed_sandbox_raises_task_error_and_cleans_up(environment, controller):
    controller.task.status.return_value = Mock(
        spec=TaskStatus, state=TaskState.FAILED, error_message="image pull failed"
    )

    with pytest.raises(iris_environment.IrisSandboxError, match="image pull failed"):
        environment._start_sync()

    controller.job.cancel.assert_called_once_with()
    controller.client.shutdown.assert_called_once_with()
    controller.endpoint.close.assert_called_once_with()


def test_start_failure_closes_endpoint(environment, monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(side_effect=RuntimeError("cannot connect")))

    with pytest.raises(RuntimeError, match="cannot connect"):
        environment._start_sync()

    endpoint.close.assert_called_once_with()
