"""Container execution checks; run explicitly with the docker marker."""

import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.machine import Command, DockerImage, ExitReason, MachineSpec

from skyrl_gym.code_execution import execute_code
from skyrl_gym.python_execution import PythonKernel

pytestmark = [pytest.mark.docker, pytest.mark.asyncio]


# CI has no Artifact Registry credentials. This public image supplies IPython and the code-grading packages.
PYTHON_IMAGE = (
    "docker.io/igitman/nemo-skills-sandbox@sha256:f8237dd0aafab99a759c11506916dc40a32a11081891f95462e5e576b87c899c"
)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def python_image():
    result = await docker("pull", PYTHON_IMAGE, timeout=600)
    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    return DockerImage(PYTHON_IMAGE)


@pytest_asyncio.fixture
async def docker_machine(python_image):
    # An empty workdir retains the directory declared by the image.
    machine = await DockerMachineFactory().create(MachineSpec(source=python_image, workdir="", memory_mb=1024, cpus=1))
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


async def test_docker_cancellation_stops_candidate_and_preserves_machine_for_grading(docker_machine):
    kernel = PythonKernel(docker_machine)
    await kernel.start()
    ready = f"{kernel.directory}/candidate-ready"
    pending = asyncio.create_task(
        kernel.execute(f"import signal\nopen({ready!r}, 'w').close()\nsignal.pause()", timeout=30.0)
    )
    try:
        async with asyncio.timeout(5.0):
            while True:
                if pending.done():
                    await pending
                    pytest.fail("The candidate exited before cancellation")
                running = await docker_machine.run(Command(("test", "-f", ready), timeout=5.0))
                if running.exit_code == 0:
                    break
                assert running.exit_code == 1
                await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        pid_file = await docker_machine.run(Command(("cat", f"{kernel.directory}/pid"), timeout=5.0))
        assert pid_file.exit_code == 0
        pid = int(pid_file.stdout)
        stopped = await docker_machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    'if [ -f "/proc/$1/stat" ]; then read -r pid comm state rest < "/proc/$1/stat"; '
                    'test "$state" = Z; fi',
                    "kernel-state",
                    str(pid),
                ),
                timeout=5.0,
            )
        )
        assert stopped.exit_code == 0
        assert (await docker_machine.run(Command(("test", "-f", ready), timeout=5.0))).exit_code == 0
    finally:
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await kernel.close()
