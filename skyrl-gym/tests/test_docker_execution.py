"""Container execution checks; run explicitly with the docker marker."""

import asyncio
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import pytest_asyncio
import yaml
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.machine import DockerImage, ExitReason, MachineSpec
from taskcompendium.environment import EnvironmentSpec

from skyrl_gym.code_execution import execute_code
from skyrl_gym.python_execution import PythonKernel

pytestmark = [pytest.mark.docker, pytest.mark.asyncio]


@pytest.fixture(scope="session")
def python_image():
    configuration = (
        Path(__file__).resolve().parents[2] / "skyrl-train/skyrl_train/config/task_session_config/default.yaml"
    )
    values = yaml.safe_load(configuration.read_text())
    specification = EnvironmentSpec.model_validate(values["lcb"]["machine"])
    image = f"skyrl-task-test:{uuid4().hex}"
    with TemporaryDirectory(prefix="skyrl-image-") as directory:
        for file in specification.image.files:
            target = Path(directory) / file.path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.content)
        subprocess.run(["docker", "build", "--tag", image, directory], check=True, timeout=300)
    try:
        yield DockerImage(image)
    finally:
        subprocess.run(["docker", "image", "rm", image], check=True, timeout=30)


@pytest_asyncio.fixture
async def docker_machine(python_image):
    machine = await DockerMachineFactory().create(
        MachineSpec(source=python_image, workdir="/workspace", memory_mb=1024, cpus=1)
    )
    try:
        yield machine
    finally:
        await machine.close()
        assert (await docker("inspect", machine.name)).exit_code != 0


async def test_docker_python_keeps_state_after_timeout_and_bounds_output(docker_machine):
    kernel = PythonKernel(docker_machine)
    await kernel.start()
    try:
        timeout = await kernel.execute("value = 42\nwhile True: pass", timeout=0.1)
        assert timeout.reason == ExitReason.TIMED_OUT
        result = await kernel.execute("print(value)", timeout=5.0)
        assert result.stdout == b"42\n"
        output = await kernel.execute("print('x' * 10000)", timeout=5.0, output_limit_bytes=100)
        assert output.stdout == b"x" * 100 and output.stdout_truncated
    finally:
        await kernel.close()


async def test_docker_candidate_cannot_read_worker_reference(docker_machine, tmp_path):
    reference = tmp_path / "private-answer.txt"
    reference.write_text("938171")
    tests = [{"input": "7", "output": "14", "testtype": "functional", "metadata": {"func_name": "solve"}}]
    assert (await execute_code(docker_machine, tests, "def solve(x): return x * 2"))[0] == 1.0
    tests[0]["output"] = reference.read_text()
    candidate = f"def solve(x):\n    try: return int(open({str(reference)!r}).read())\n    except OSError: return -999"
    assert (await execute_code(docker_machine, tests, candidate))[0] == 0.0
    assert reference.read_text() == "938171"


async def test_docker_cancellation_disposes_of_blocked_candidate(docker_machine):
    kernel = PythonKernel(docker_machine)
    await kernel.start()
    pending = asyncio.create_task(kernel.execute("import signal; signal.pause()", timeout=30.0))
    async with asyncio.timeout(5.0):
        while True:
            running = await docker("top", docker_machine.name, "-eo", "args")
            if b"call" in running.stdout:
                break
            await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert (await docker("inspect", docker_machine.name)).exit_code != 0
    await kernel.close()
