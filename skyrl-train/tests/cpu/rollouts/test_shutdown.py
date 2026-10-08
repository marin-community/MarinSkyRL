import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
import threading

import numpy as np
import pytest
from loguru import logger
from omegaconf import OmegaConf
from skyrl_gym.task_sessions import AnswerTaskSession
from taskcompendium.grading_result import GradeResult, Outcome
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ShellSimBuiltins
from taskcompendium.shell_verifier import ArtifactKind, ExitCodeReward, ShellVerifierSpec, VerifierArtifact
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    Source,
    EnvironmentRequirements,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.group_grader import GroupGraderSpec
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_gym.source_task import source_task
from rolloutengine.spec import LoweredTaskSpec
from tests.cpu.task_specs import lowered_task
from rolloutengine.contracts import ModelTurn, Transition
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryID
from tests.cpu.rollouts.engine_fakes import InferenceClient, Tokenizer, Writer


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_one_machine_cleanup_failure_does_not_abort_the_buffer_group(task_inputs, projection_type):
    config, original = task_inputs
    task = LoweredTaskSpec.model_validate_json(original["env_extras"][0]["lowered_task_spec"]).task
    task = task.model_copy()
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": original["env_classes"] * 2,
        "env_extras": [{"lowered_task_spec": lowered_task(task, "shellbox", backend="shellsim").model_dump_json()}] * 2,
        "trajectory_ids": [TrajectoryID("arithmetic", 0), TrajectoryID("arithmetic", 1)],
    }
    machines = []

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()
            raise OSError("Remote cleanup response failed")

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            return Machine(machine) if len(machines) == 1 else machine

    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InferenceClient(),
        {"shellsim": Factory()},
        shutdown_timeout=30,
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    finally:
        with pytest.raises(ExceptionGroup, match="Task worker shutdown failed"):
            await worker.shutdown()
    assert len(writer.groups) == 1
    batch = writer.groups[0][1].trajectory_batch
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[1, 1], [1, 1]]
    assert batch["exclude_from_baseline"] == [False, False]
    assert batch["rewards"] == [1.0, 1.0]
    errors = [result.diagnostics.get("cleanup_errors", []) for result in batch["verification_results"]]
    assert sum(len(items) for items in errors) == 1
    assert {"operation": "machine_close", "exception_type": "OSError"} in next(items for items in errors if items)
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


@dataclass
class DeletionMachine:
    machine: object
    deletions: list
    deleted: asyncio.Event

    async def run(self, command):
        return await self.machine.run(command)

    async def upload(self, source, target):
        await self.machine.upload(source, target)

    async def download(self, source, target):
        await self.machine.download(source, target)

    async def close(self):
        await self.machine.close()
        self.deletions.append(self.machine)
        self.deleted.set()


@pytest.mark.asyncio
async def test_worker_shutdown_cancels_model_and_deletes_its_machine(task_inputs):
    config, request = task_inputs
    lowered = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"])
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(lowered.task, backend="fixture").model_dump_json()
    started = asyncio.Event()
    deleted = asyncio.Event()
    deletions = []

    class Client(InferenceClient):
        async def generate(self, request):
            started.set()
            await asyncio.Event().wait()

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(spec)
            return DeletionMachine(machine, deletions, deleted)

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"fixture": Factory()},
        shutdown_timeout=30,
    )
    async with asyncio.timeout(5):
        pending = asyncio.create_task(worker.run(request))
        await started.wait()
        await worker.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert len(deletions) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await deletions[0].run(Command(("true",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_during_shutdown", [True, False])
async def test_worker_shutdown_owns_late_creation_and_bounds_a_blocked_provider(task_inputs, complete_during_shutdown):
    config, request = task_inputs
    lowered = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"])
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        lowered.task, backend="fixture-provider"
    ).model_dump_json()
    started = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()
    deletions = []
    diagnostics = []

    class Factory:
        async def create(self, spec):
            started.set()
            await release.wait()
            machine = await ShellSimMachineFactory().create(spec)
            return DeletionMachine(machine, deletions, deleted)

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {"fixture-provider": Factory()},
        shutdown_timeout=0.05 if not complete_during_shutdown else 5,
    )
    log_sink = logger.add(lambda message: diagnostics.append(message.record["extra"]), level="ERROR")
    try:
        async with asyncio.timeout(5):
            rollout = asyncio.create_task(worker.run(request))
            await started.wait()
            shutdown = asyncio.create_task(worker.shutdown())
            with pytest.raises(asyncio.CancelledError):
                await rollout
            if complete_during_shutdown:
                release.set()
                await shutdown
            else:
                with pytest.raises(ExceptionGroup, match="Task worker shutdown failed"):
                    await shutdown
                assert {"provider": "fixture-provider", "open_creations": 1, "open_machines": 0} in diagnostics
                release.set()
            await deleted.wait()
            await worker.shutdown()
    finally:
        release.set()
        logger.remove(log_sink)
    assert len(deletions) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await deletions[0].run(Command(("true",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize("timeout_phase", ["model", "advance"])
@pytest.mark.parametrize("deadline_source", ["task", "train", "eval"])
async def test_agent_deadlines_grade_the_workspace_and_commit_training_tokens(
    task_inputs, projection_type, timeout_phase, deadline_source
):
    config, request = task_inputs
    config.error_handling = {
        "enable_error_classification": True,
        "mask_exceptions": ["AgentTimeoutError"],
        "preserve_logprobs_on_timeout": False,
    }
    task = TaskSpec(
        id="deadline",
        context=ConversationInput(events=(TextMessage(role="user", content="Write the answer file."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.FILE,
        verifier=VerifierSpec(
            kind="shell",
            environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
            parameters_json=ShellVerifierSpec(
                argv=("test", "-f", "/workspace/answer"),
                reward=ExitCodeReward(),
                artifacts=(
                    VerifierArtifact(source="/workspace/answer", target="/workspace/answer", kind=ArtifactKind.FILE),
                ),
            ).model_dump_json(),
        ),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    settings = None
    if deadline_source != "task":
        task = task.model_copy(update={"tags": ("harbor",)})
        settings = HarborTaskSettings.from_config(
            OmegaConf.create(
                {
                    "harbor": {
                        "override_timeout_sec": 1 if deadline_source == "train" else None,
                        "eval_timeout_override_sec": 1 if deadline_source == "eval" else 900,
                        "timeout_multiplier": 2,
                        "max_timeout_sec": 1,
                        "max_retries": 0,
                    }
                }
            )
        )
        request["batch_metadata"] = BatchMetadata(0, "eval" if deadline_source == "eval" else "train")
    request["env_extras"] = [
        {
            "lowered_task_spec": lowered_task(
                task, backend="shellsim", verifier_backend="shellsim", total_turn_timeout=1
            ).model_dump_json()
        }
    ]
    machines = []

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            result = await self.machine.run(command)
            if timeout_phase == "advance" and command.argv == ("sh", "-c", "echo 12 > /workspace/answer"):
                await asyncio.Future()
            return result

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            return Machine(machine)

    class Client(InferenceClient):
        def __init__(self):
            self.generated = False

        async def generate(self, request):
            if self.generated:
                await asyncio.Future()
            self.generated = True
            output = await super().generate(request)
            output["assistant_messages"] = [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "write",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"command":"echo 12 > /workspace/answer"}'},
                        }
                    ],
                }
            ]
            return output

    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        Client(),
        {"shellsim": Factory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["exclude_from_baseline"] == [False]
    assert batch["stop_reasons"] == ["total_turn_timeout"]
    assert not batch.get("exception_types")
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


@pytest.fixture(params=[0, 1])
def verifier_executor(request):
    if request.param == 0:
        yield None
    else:
        with ThreadPoolExecutor(max_workers=request.param) as executor:
            yield executor


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2])
@pytest.mark.parametrize("verifier_fails", [False, True])
async def test_cancellation_returns_before_the_verifier_but_cleanup_keeps_ownership(
    python_tool_task, verifier_executor, cancellations, verifier_fails
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    events = []

    def execute(turn, config, extras):
        events.append("start")
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        events.append("finish")
        if verifier_fails:
            raise RuntimeError("Verifier failed after cancellation")
        return Transition(done=True, reward=1.0, grade=GradeResult(Outcome.GRADED, 1.0))

    task, message = python_tool_task
    session = AnswerTaskSession(task, None, grader=execute, executor=verifier_executor)
    await session.prepare()
    operation = asyncio.create_task(session.advance(ModelTurn(message, (1, 2), (3, 4), None, "stop")))
    cleanup = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=5)
        assert events == ["start"]

        async def close():
            try:
                await session.close()
            finally:
                events.append("close")

        cleanup = asyncio.create_task(close())
        cleanup_started = loop.create_future()
        loop.call_soon(cleanup_started.set_result, None)
        await cleanup_started
        for _ in range(cancellations):
            cleanup.cancel()
            completion_at_cancel = loop.create_future()
            # Deliver cleanup cancellation while the verifier thread still owns its resources.
            loop.call_soon(loop.call_soon, lambda: completion_at_cancel.set_result(cleanup.done()))
            assert not await completion_at_cancel
    finally:
        release.set()
        if cleanup is not None:
            with pytest.raises(asyncio.CancelledError) as cancellation:
                await cleanup
        else:
            await session.close()
    assert events == ["start", "finish", "close"]
    if verifier_fails:
        assert isinstance(cancellation.value.__cause__, RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", ["tool_turn_timeout", "attempt_timeout"])
async def test_blocking_grader_cannot_extend_attempt_or_cleanup_deadlines(task_inputs, deadline):
    config, request = task_inputs
    config.error_handling = {
        "enable_error_classification": True,
        "mask_exceptions": ["AgentTimeoutError", "TrialTimeoutError"],
    }
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()

    def grade(turn, config, extras):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        loop.call_soon_threadsafe(finished.set)
        return Transition(done=True, reward=1.0, grade=GradeResult(Outcome.GRADED, 1.0))

    task = source_task(
        request["prompts"][0], {}, {}, Source(dataset="deadline", revision="1", row="0", importer_revision="1")
    )
    selected = lowered_task(task, "blocking", cleanup_timeout=0.01, **{deadline: 0.25})
    request["env_extras"] = [{"lowered_task_spec": selected.model_dump_json()}]
    request["env_classes"] = ["blocking"]
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {},
        sessions={"blocking": partial(AnswerTaskSession, grader=grade)},
        shutdown_timeout=30,
    )
    operation = asyncio.create_task(worker.run(request))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        batch = await asyncio.wait_for(operation, timeout=5)
        assert not release.is_set()
        assert batch["exclude_from_baseline"] == [True]
        assert batch["verification_results"][0].score is None
        assert batch["verification_results"][0].diagnostics["cleanup_errors"] == [
            {"operation": "session_close", "exception_type": "TimeoutError"}
        ]
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        await worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["group_grade", "enqueue"])
async def test_worker_shutdown_cancels_the_whole_request_before_buffer_commit(task_inputs, phase):
    config, request = task_inputs
    started = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def blocking_grader(task, specification, records, eligible, training_phase):
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(timeout=5)
            return records
        finally:
            loop.call_soon_threadsafe(finished.set)

    class BlockingWriter(Writer):
        async def write_rollout(self, lease, group):
            if phase == "enqueue":
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    finished.set()
            await super().write_rollout(lease, group)

    if phase == "group_grade":
        request["env_extras"][0]["group_grader"] = GroupGraderSpec(
            name="fixture", parameters_json="{}"
        ).model_dump_json()
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {},
        group_graders={"fixture": blocking_grader},
        shutdown_timeout=5,
    )
    writer = BlockingWriter()
    pending = asyncio.create_task(
        worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "arithmetic"}, request), writer)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.wait_for(worker.shutdown(), timeout=5)
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert writer.groups == []
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_error", [OSError, asyncio.CancelledError])
async def test_worker_shutdown_reports_late_provider_close_failure_without_unhandled_task(task_inputs, provider_error):
    config, request = task_inputs
    lowered = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"])
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        lowered.task, backend="fixture-provider"
    ).model_dump_json()
    started = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()
    deletions = []
    diagnostics = []
    unhandled = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unhandled.append(context))
    log_sink = logger.add(lambda message: diagnostics.append(message.record["extra"]), level="ERROR")

    class DelayedFailureMachine(DeletionMachine):
        async def close(self):
            started.set()
            await release.wait()
            await super().close()
            raise provider_error("Provider deletion failed after the shutdown deadline")

    class Factory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(spec)
            return DelayedFailureMachine(machine, deletions, deleted)

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {"fixture-provider": Factory()},
        shutdown_timeout=0.02,
    )
    pending = asyncio.create_task(worker.run(request))
    try:
        async with asyncio.timeout(5):
            await started.wait()
            with pytest.raises(ExceptionGroup, match="Task worker shutdown failed"):
                await asyncio.wait_for(worker.shutdown(), timeout=1)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await deleted.wait()
            ready = loop.create_future()
            loop.call_soon(ready.set_result, None)
            await ready
        assert {
            "provider": "fixture-provider",
            "operation": "machine_close",
            "exception_type": provider_error.__name__,
        } in diagnostics
        assert unhandled == []
        assert len(deletions) == 1
        with pytest.raises(RuntimeError, match="closed"):
            await deletions[0].run(Command(("true",)))
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        loop.set_exception_handler(previous_handler)
        logger.remove(log_sink)
