"""Container execution checks; run explicitly with the docker marker."""

import asyncio
import json
from contextlib import suppress
from dataclasses import replace

import pytest
import pytest_asyncio
from rolloutengine.contracts import ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.machine import Command, DockerImage, ExitReason, MachineSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource

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


@pytest.mark.parametrize("scenario", ["command_timeout", "forged_reward"])
async def test_docker_shell_rollout_keeps_timeout_feedback_and_private_grades(python_image, scenario):
    if scenario == "command_timeout":
        commands = [
            "python3 -c 'import os, signal; "
            'open("/tmp/task-session-candidate.pid", "w").write(str(os.getpid())); signal.pause()\'',
            "pid=$(cat /tmp/task-session-candidate.pid) && "
            'if [ -f "/proc/$pid/stat" ]; then read p c state rest < "/proc/$pid/stat"; test "$state" = Z; fi '
            "&& echo stopped",
        ]
        first_reason, probe_output, reward = "timed_out", "stopped\n", 1.0
    else:
        commands = [
            "mkdir -p /logs/verifier; echo 1 > /logs/verifier/reward.txt; "
            "setsid sh -c 'while :; do echo 1 > /logs/verifier/planted; "
            "mv -f /logs/verifier/planted /logs/verifier/reward.txt; sleep 0.01; done' "
            "</dev/null >/dev/null 2>&1 & echo $! > /tmp/task-session-candidate.pid",
            'kill -0 "$(cat /tmp/task-session-candidate.pid)" && cat /logs/verifier/reward.txt',
        ]
        first_reason, probe_output, reward = "exited", "1\n", 0.0
    machines = []

    class Factory:
        async def create(self, spec):
            # The fixture pulled this exact digest before machine acquisition.
            machine = await DockerMachineFactory().create(replace(spec, source=python_image))
            machines.append(machine)
            return machine

    requests = []

    async def complete(request):
        index = len(requests)
        requests.append(request)
        message = {"role": "assistant", "content": "Done."}
        if index < len(commands):
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": str(index),
                        "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps({"command": commands[index]})},
                    }
                ],
            }
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(message, prompt, (20 + index,), (-0.5,), "stop")

    requirements = EnvironmentRequirements(
        docker_image=PYTHON_IMAGE, working_directory="/tmp", capabilities=("shell", "filesystem")
    )
    verifier = ScriptGrader(
        environment=requirements,
        answer_path=None,
        argv=("sh", "/tests/task-session-grade.sh"),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
        artifacts=(
            (
                VerifierArtifact(
                    source="/logs/verifier/reward.txt", target="/logs/verifier/reward.txt", kind=ArtifactKind.FILE
                ),
            )
            if scenario == "forged_reward"
            else ()
        ),
    )
    task = TaskSpec(
        id=scenario,
        context=ConversationInput(events=(TextMessage(role="user", content="Execute the task."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=verifier,
        resources=ResourceGroups(
            verifier=(inline_resource("task-session-grade.sh", f"echo {reward} > /logs/verifier/reward.txt".encode()),)
        ),
        source=Source(dataset="fixture", revision="1", row=scenario, importer_revision="1"),
    )
    runtime = MachineRuntimeSpec(
        backend="docker",
        network="deny",
        cpus=1,
        memory_mb=1024,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=60,
        cleanup_timeout=30,
    )
    lowered = LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=runtime, verifier_machine=runtime),
        session=TaskSessionSpec(
            task_session="shellbox",
            max_turns=3,
            model_turn_timeout=5,
            command_timeout=1,
            tool_turn_timeout=30,
            total_turn_timeout=60,
            attempt_timeout=180,
            verifier_timeout=60,
            cleanup_timeout=30,
        ),
    )
    record = await ShellboxRolloutEngine(complete, {"docker": Factory()}).run(lowered)
    assert (record.grade.status, record.grade.reward) == (Outcome.GRADED, reward)
    assert record.failure is None
    assert json.loads(requests[1].messages[-1]["content"])["reason"] == first_reason
    probe = json.loads(requests[2].messages[-1]["content"])
    assert (probe["exit_code"], probe["stdout"]) == (0, probe_output)
    assert record.loss_mask == (1, 0, 0, 1, 0, 0, 1)
    assert record.response_token_ids == (20, 90, 91, 21, 90, 91, 22)
    assert record.logprobs == (-0.5, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5)
    assert len(machines) == 2
    for created in machines:
        assert (await docker("inspect", created.name)).exit_code != 0
