import hashlib
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("harbor")

from harbor.environments.base import BaseEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from iris.client import Job
from iris.client.workload_codec import task_status_from_proto
from iris.cluster.types import JobName
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


def test_endpoint_lives_until_environment_stops(environment, monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    client = Mock()
    job = Mock(spec=Job, job_id="job-1")
    client.submit.return_value = job
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(return_value=client))
    monkeypatch.setattr(iris_environment, "ControllerServiceClientSync", Mock())
    monkeypatch.setattr(environment, "_wait_for_running", Mock(return_value="task-1"))

    environment._start_sync()

    endpoint.close.assert_not_called()
    client.submit.assert_called_once()
    assert client.submit.call_args.kwargs["task_image"] == environment.task_env_config.docker_image

    environment._stop_sync()

    job.cancel.assert_called_once_with()
    client.shutdown.assert_called_once_with()
    endpoint.close.assert_called_once_with()


def test_start_failure_closes_endpoint(environment, monkeypatch: pytest.MonkeyPatch):
    endpoint = SimpleNamespace(url="http://iris-controller:10000", credentials=None, close=Mock())
    monkeypatch.setattr(iris_environment, "connect_controller", Mock(return_value=endpoint))
    monkeypatch.setattr(iris_environment.IrisClient, "remote", Mock(side_effect=RuntimeError("cannot connect")))

    with pytest.raises(RuntimeError, match="cannot connect"):
        environment._start_sync()

    endpoint.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_state", [job_pb2.TASK_STATE_RUNNING, job_pb2.TASK_STATE_FAILED])
async def test_mapped_dockerfile_image_starts_in_gvisor_and_restores_task_files(tmp_path, monkeypatch, task_state):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    dockerfile = b"FROM python:3.10-slim\nWORKDIR /app\nCOPY . /app/\n"
    (task_dir / "Dockerfile").write_bytes(dockerfile)
    (task_dir / "input.json").write_bytes(b'{"value": 37}\n')
    sandbox_dir = tmp_path / "sandbox"
    (sandbox_dir / "app").mkdir(parents=True)
    (sandbox_dir / "tmp").mkdir()
    submitted = []
    lifecycle = {"cancelled": False}
    job_name = JobName.from_wire("/test/sandbox-job")
    status = task_status_from_proto(
        job_pb2.TaskStatus(task_id=job_name.task(0).to_wire(), state=task_state, error="sandbox failed")
    )

    def submit(**kwargs):
        submitted.append(kwargs)
        return Job(client, job_name)

    def exec_in_container(request, **kwargs):
        # Model the remote exec boundary with a local filesystem rooted under tmp_path.
        script = request.command[2]
        for root in ("/logs", "/tmp", "/app"):
            script = script.replace(root, str(sandbox_dir / root.lstrip("/")))
        result = subprocess.run([*request.command[:2], script], cwd=sandbox_dir / "app", capture_output=True, text=True)
        stdout = result.stdout.replace(str(sandbox_dir), "")
        return SimpleNamespace(error="", stdout=stdout, stderr=result.stderr, exit_code=result.returncode)

    endpoint = SimpleNamespace(url="http://controller", credentials=None, close=lambda: None)
    client = SimpleNamespace(
        submit=submit,
        list_tasks=lambda name: [status],
        task_status=lambda name: status,
        cancel_job=lambda name: lifecycle.update(cancelled=True),
        shutdown=lambda: None,
    )
    monkeypatch.setattr(iris_environment, "connect_controller", lambda **kwargs: endpoint)
    monkeypatch.setattr(iris_environment.IrisClient, "remote", lambda *args, **kwargs: client)
    monkeypatch.setattr(
        iris_environment,
        "ControllerServiceClientSync",
        lambda **kwargs: SimpleNamespace(exec_in_container=exec_in_container),
    )
    image = "registry.example/bfcl@sha256:" + "a" * 64
    task_config = EnvironmentConfig()
    original_config = task_config.model_dump()
    sandbox = iris_environment.IrisEnvironment(
        environment_dir=task_dir,
        environment_name="bfcl-test",
        session_id="bfcl-test",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=task_config,
        controller_url="http://controller",
        prebuilt_images={hashlib.sha256(dockerfile).hexdigest(): image},
    )
    if task_state == job_pb2.TASK_STATE_FAILED:
        with pytest.raises(iris_environment.IrisSandboxError, match="sandbox failed"):
            await sandbox.start(force_build=False)
        assert lifecycle["cancelled"]
        return
    try:
        await sandbox.start(force_build=False)
        assert submitted[0]["task_image"] == image
        assert submitted[0]["container_profile"] == job_pb2.CONTAINER_PROFILE_GVISOR
        assert (sandbox_dir / "app/input.json").read_bytes() == (task_dir / "input.json").read_bytes()
        assert (sandbox_dir / "app/Dockerfile").read_bytes() == dockerfile
        assert task_config.model_dump() == original_config
        result = await sandbox.exec(
            'set -euo pipefail; values=("$BFCL_TEST_VALUE" "$PWD"); printf "%s\\n" "${values[@]}" > shell-result.txt',
            cwd="/app",
            env={"BFCL_TEST_VALUE": "quoted ' value"},
        )
        assert result.return_code == 0, result.stderr
        assert (sandbox_dir / "app/shell-result.txt").read_text().splitlines() == [
            "quoted ' value",
            str(sandbox_dir / "app"),
        ]
    finally:
        await sandbox.stop(delete=True)
    assert lifecycle["cancelled"]
