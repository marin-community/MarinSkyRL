"""Task Parquet through canonical execution and the leased buffer writer."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import shutil
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger
from harbor_config.errors import ErrorCategory, error_category
from jinja2 import Environment, StrictUndefined
from omegaconf import OmegaConf
from skyrl_gym.task_records import fold_grades, grade_result
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score
from taskcompendium.grading import numeric_answer
from taskcompendium.grading_result import GradeResult, Outcome
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ExitReason, Result, ShellSimBuiltins, UnsupportedMachineSpec, NetworkPolicy
from taskcompendium.shell_verifier import (
    ArtifactKind,
    ExitCodeReward,
    FileReward,
    MissingArtifactPolicy,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
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
from skyrl_train.dataset.tasks import SourceTaskDataset, TaskDataset, source_row_task
from skyrl_train.dataset.harbor import HarborTaskDataset
from skyrl_train.dataset.nemotron_ultra import NemotronTaskDataset
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.group_grader import GenRMGroupGraderParameters, GroupGraderSpec, task_group_grader
from skyrl_train.rollouts.genrm_grading import grade_genrm_rollouts
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_gym.source_task import source_task
from rolloutengine.spec import LoweredTaskSpec
from skyrl_train.config.utils import get_default_config
from tests.cpu.task_specs import lowered_task, machine_runtime, session_spec
from rolloutengine.contracts import ModelTurn, RolloutContractError, RolloutData, RolloutStep, SessionStart, Transition
from skyrl_train.trajectory_runners.types import BatchMetadata, TokenProvenance, TrajectoryID
from skyrl_train.trajectory_runners.model_clients import DirectModelClient, ModelServerError
from skyrl_train.rollout_observability import observe_rollout_call


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt):
        return [1, 2]


class InferenceClient:
    async def generate(self, request):
        return {
            "assistant_messages": [{"role": "assistant", "content": "12"}],
            "prompt_ids": [[1, 2]],
            "response_ids": [[3, 4]],
            "response_logprobs": [[-0.1, -0.2]],
            "stop_reasons": ["stop"],
            "token_provenance": TokenProvenance.ENGINE,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "verification,eligible,exception_type",
    [
        (VerificationResult.skipped("grading disabled"), True, None),
        (VerificationResult.unavailable("judge unreachable"), False, "VerifierUnavailable"),
        (VerificationResult.error("sandbox lost state"), False, "VerifierRuntimeError"),
    ],
)
async def test_session_verifier_failures_reach_training_eligibility(
    task_inputs, projection_type, verification, eligible, exception_type
):
    def grader(turn, config, extras):
        return Transition(done=True, reward=1.0, grade=grade_result(verification))

    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "verifier").model_dump_json()}]
    request["env_classes"] = ["verifier"]
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InferenceClient(),
        {},
        sessions={"verifier": partial(AnswerTaskSession, grader=grader)},
        shutdown_timeout=30,
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch

    assert batch["verification_results"][0].status is verification.status
    assert batch["verification_results"][0].score is None
    assert batch["loss_masks"] == [[int(eligible), int(eligible)]]
    assert batch["exclude_from_baseline"] == [not eligible]
    assert batch.get("exception_types", [None]) == [exception_type]
    if not eligible:
        assert batch["rewards"] == [[0.0, 0.0]]
    if exception_type == "VerifierRuntimeError":
        assert error_category(batch["exception_types"][0]) is ErrorCategory.INFRASTRUCTURE


@dataclass
class Writer:
    groups: list[tuple[RolloutLease, RolloutGroup]] = field(default_factory=list)

    async def write_rollout(self, lease, group):
        self.groups.append((lease, group))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "projection_type,expected_rewards",
    [(WholeTaskProjection, [[0.0, 0.0], [0.0, 1.0]]), (StepTaskProjection, [[0.0, 0.0], 1.0])],
)
@pytest.mark.parametrize("session", ["nupa", "reasoning_gym"])
async def test_corrupt_source_task_is_masked_without_losing_its_graded_peer(
    task_inputs, projection_type, expected_rewards, session
):
    config, original = task_inputs
    failed = source_task(
        original["prompts"][0],
        {"reward_spec": {"ground_truth": "not JSON"}, "reward_model": {"ground_truth": "not JSON"}},
        {},
        Source(dataset="corrupt", revision="1", row="0", importer_revision="1"),
    )
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": [session, "taskcompendium"],
        "env_extras": [
            {"lowered_task_spec": lowered_task(failed, session).model_dump_json(), "teacher_route": "corrupt"},
            original["env_extras"][0],
        ],
        "trajectory_ids": [TrajectoryID("corrupt", 0), TrajectoryID("arithmetic", 0)],
    }
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InferenceClient(),
        {},
        shutdown_timeout=30,
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": failed.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch
    assert len(writer.groups) == 1
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[0, 0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["rewards"] == expected_rewards
    assert batch["exception_types"] == ["VerifierRuntimeError", None]
    assert [result.status for result in batch["verification_results"]] == [
        VerificationStatus.ERROR,
        VerificationStatus.VERIFIED,
    ]


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
async def test_mixed_harbor_and_native_tasks_keep_separate_providers_and_machine_specs(task_inputs):
    config, original = task_inputs
    lowered = LoweredTaskSpec.model_validate_json(original["env_extras"][0]["lowered_task_spec"])
    native = lowered.task.model_copy(
        update={"environment_requirements": EnvironmentRequirements(docker_image="fixture@sha256:" + "a" * 64)}
    )
    harbor = native.model_copy(update={"id": "harbor", "tags": ("harbor",)})
    machines = {"docker": [], "harbor": []}

    class Factory:
        def __init__(self, provider):
            self.provider = provider

        async def create(self, spec):
            machines[self.provider].append(spec)
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), cpus=None, workdir="/workspace")
            )

    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"override_cpus": 2, "max_retries": 0}}))
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": ["taskcompendium"] * 2,
        "env_extras": [
            {"lowered_task_spec": lowered_task(task, backend="docker").model_dump_json()} for task in (native, harbor)
        ],
        "trajectory_ids": [TrajectoryID(task.id, 0) for task in (native, harbor)],
    }
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {provider: Factory(provider) for provider in machines},
        harbor=settings,
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert batch["unshaped_rewards"] == [1.0, 1.0]
    assert [sum(reward) for reward in batch["rewards"]] == [1.0, 1.0]
    assert batch["exclude_from_baseline"] == [False, False]
    assert len(machines["docker"]) == len(machines["harbor"]) == 1
    assert machines["docker"][0].network is NetworkPolicy.DENY
    assert machines["docker"][0].cpus is None
    assert machines["harbor"][0].cpus == 2


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer("12", tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 2,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = {
        "prompts": [[{"role": "user", "content": "What is six plus six?"}]],
        "env_classes": ["taskcompendium"],
        "env_extras": [
            {
                "lowered_task_spec": lowered_task(task, "shellbox").model_dump_json(),
                "teacher_route": "arithmetic",
                "data_source": "arithmetic",
            }
        ],
        "trajectory_ids": [TrajectoryID("arithmetic", 0)],
        "sampling_params": None,
        "batch_metadata": None,
    }
    return config, request


@pytest.mark.asyncio
async def test_task_replay_preserves_inference_session(task_inputs):
    config, request = task_inputs
    sessions = []

    class Client(InferenceClient):
        async def generate(self, request):
            sessions.extend(request["session_ids"])
            return await super().generate(request)

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {},
        shutdown_timeout=30,
    )
    try:
        await worker.generate(request)
        await worker.generate(request)
        await worker.generate({**request, "trajectory_ids": [TrajectoryID("arithmetic", 1)]})
    finally:
        await worker.shutdown()
    assert sessions == ["arithmetic_0", "arithmetic_0", "arithmetic_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_logprobs", [True, False])
async def test_task_grade_and_exact_tokens_reach_the_leased_buffer(task_inputs, with_logprobs):
    config, request = task_inputs
    if not with_logprobs:
        config.sampling_params.logprobs = None

    class Client(InferenceClient):
        async def generate(self, request):
            output = await super().generate(request)
            if not with_logprobs:
                output["response_logprobs"] = None
            return output

    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {},
        shutdown_timeout=30,
    )
    lease = RolloutLease("lease", policy_step=7, batch_id=8)
    writer = Writer()
    tokens = await runner.run_task(RolloutTask(lease, {"uid": "arithmetic"}, request), writer)
    assert tokens == 2
    committed_lease, group = writer.groups[0]
    assert committed_lease == lease
    assert (group.uid, group.policy_step) == ("arithmetic", 7)
    batch = group.trajectory_batch
    assert batch["prompt_token_ids"] == [[1, 2]]
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[1, 1]]
    if with_logprobs:
        np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    else:
        assert batch["rollout_logprobs"] is None
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["rollout_metrics"]["generate/failed_trajectory_fraction"] == 0.0
    assert batch["teacher_route_keys"] == ["arithmetic"]
    assert batch["data_sources"] == ["arithmetic"]
    assert batch["trajectory_ids"] == [TrajectoryID("arithmetic", 0)]
    assert {
        name: value for name, value in batch["rollout_metrics"].items() if name.startswith("generate/task_rollout/")
    } == {
        "generate/task_rollout/tasks": 1.0,
        "generate/task_rollout/turns": 1.0,
        "generate/task_rollout/multi_turn_tasks": 0.0,
        "generate/task_rollout/tool_tasks": 0.0,
        "generate/task_rollout/generated_tokens": 2.0,
        "generate/task_rollout/missing_logprob_tokens": 0.0 if with_logprobs else 2.0,
    }


class FixtureImageFactory:
    """Execute fixture image commands on the built-in filesystem."""

    async def create(self, spec):
        return await ShellSimMachineFactory().create(
            replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
        )


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


@pytest.mark.asyncio
async def test_model_programming_failure_does_not_commit_a_partial_group(task_inputs):
    first_response = asyncio.Event()

    class FailedClient:
        async def generate(self, request):
            if not first_response.is_set():
                response = await InferenceClient().generate(request)
                first_response.set()
                return response
            raise ValueError("Invalid inference response")

    writer = Writer()
    config, request = task_inputs
    request.update(
        prompts=request["prompts"] * 2,
        env_classes=request["env_classes"] * 2,
        env_extras=request["env_extras"] * 2,
        trajectory_ids=[TrajectoryID("arithmetic", index) for index in range(2)],
    )
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        FailedClient(),
        {},
        shutdown_timeout=30,
    )
    with pytest.raises(ExceptionGroup) as failure:
        await runner.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "task"}, request), writer)
    assert writer.groups == []
    assert isinstance(failure.value.exceptions[0], ValueError)


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "template,cause_type",
    [("{{ missing }}", "UndefinedError"), ("{% if %}", "TemplateSyntaxError")],
)
async def test_model_template_failure_is_masked_without_losing_its_graded_peer(
    task_inputs, projection_type, template, cause_type
):
    class TemplateClient(InferenceClient):
        async def generate(self, request):
            Environment(undefined=StrictUndefined).from_string(request["prompts"][0][0]["content"]).render()
            return await super().generate(request)

    config, original = task_inputs
    config.error_handling = {"enable_error_classification": True}
    prompt = [{"role": "user", "content": template}]
    failed = source_task(
        prompt,
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="template", revision="1", row="0", importer_revision="1"),
    )
    request = {
        **original,
        "prompts": [prompt, original["prompts"][0]],
        "env_classes": ["gsm8k", "taskcompendium"],
        "env_extras": [
            {"lowered_task_spec": lowered_task(failed, "gsm8k").model_dump_json(), "teacher_route": "template"},
            original["env_extras"][0],
        ],
        "trajectory_ids": [TrajectoryID(failed.id, 0), TrajectoryID("arithmetic", 0)],
    }
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(config, projection_type(projection), TemplateClient(), {}, shutdown_timeout=30)
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": failed.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch
    assert batch["response_ids"] == [[0], [3, 4]]
    assert batch["loss_masks"] == [[0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["exception_types"] == ["TemplateError", None]
    assert batch["error_treatments"] == ["mask", None]
    assert batch["unshaped_rewards"] == [0.0, 1.0]
    failed_result, peer_result = batch["verification_results"]
    assert failed_result.status is VerificationStatus.ERROR
    assert failed_result.diagnostics["cause_error_type"] == cause_type
    assert peer_result.status is VerificationStatus.VERIFIED
    assert peer_result.score == 1.0


@dataclass
class ConversationClient:
    responses: list[str]
    requests: list[dict] = field(default_factory=list)
    messages: list[dict] | None = None

    async def generate(self, request):
        index = len(self.requests)
        self.requests.append(request)
        continuation = request["chat_continuations"][0]
        prompt = [1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]
        tokens = [3 + index * 2, 4 + index * 2]
        response = self.responses[index]
        return {
            "assistant_messages": [
                {"role": "assistant", "content": response} if self.messages is None else self.messages[index]
            ],
            "responses": [response],
            "prompt_ids": [prompt],
            "response_ids": [tokens],
            "response_logprobs": [[-0.1, -0.2]],
            "stop_reasons": ["stop"],
            "token_provenance": TokenProvenance.ENGINE,
            "student_topk_indices": [[[token, 99] for token in tokens]],
            "behavior_topk_logprobs": [[[-0.1, -0.2], [-0.2, -0.3]]],
            "routed_experts": [np.ones((2, 1, 1), dtype=np.uint8)],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_disabled_harbor_verification_keeps_tokens_but_masks_training(
    tmp_path, task_inputs, phase, projection_type
):
    config, request = task_inputs
    source = tmp_path / "private"
    (source / "tests").mkdir(parents=True)
    (source / "instruction.md").write_text("Complete the task.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:'
        + "a" * 64
        + '"\n'
        '[verifier]\nenvironment_mode = "separate"\nuser = "candidate"\n'
    )
    (source / "tests/test.sh").write_text("echo private-grade-only\n")
    native = HarborTaskDataset(
        [str(source)], Tokenizer(), 100, session=session_spec(), cache_dir=tmp_path / "native", num_workers=1
    )
    assert LoweredTaskSpec.model_validate_json(native[0][2]["lowered_task_spec"]).task.resources.verifier
    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"verifier_disable": True}}))
    prepared = HarborTaskDataset(
        [str(source)],
        Tokenizer(),
        100,
        session=session_spec(),
        cache_dir=tmp_path / "disabled",
        verifier_override=settings.verifier_override(),
        num_workers=1,
    )
    prompt, env, extras, uid = prepared[0]
    assert LoweredTaskSpec.model_validate_json(extras["lowered_task_spec"]).task.resources.verifier == ()
    request.update(prompts=[prompt], env_classes=[env], env_extras=[extras], trajectory_ids=[TrajectoryID(uid, 0)])
    request["batch_metadata"] = BatchMetadata(0, phase)
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        ConversationClient(["Done"]),
        {"harbor": ImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert all(grade.status == VerificationStatus.SKIPPED for grade in batch["verification_results"])
    assert all(grade.score is None for grade in batch["verification_results"])
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[0, 0]]
    assert batch["exclude_from_baseline"] == [True]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    assert batch["rollout_metrics"]["generate/task_rollout/turns"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["startup", "model", "verifier", "attempt", "cancel", "cancel_model"])
async def test_harbor_retries_close_failed_attempts_and_commit_only_the_selected_result(task_inputs, failure_stage):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=ShellVerifierSpec(
                    argv=("cat", "/workspace/reward"),
                    artifacts=(
                        VerifierArtifact(
                            source="/workspace/reward",
                            target="/workspace/reward",
                            kind=ArtifactKind.FILE,
                            missing=MissingArtifactPolicy.SKIP,
                        ),
                    ),
                ).model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task, "shellbox", backend="shellsim", verifier_backend="verifier"
    ).model_dump_json()
    attempts = 0
    machines = []
    waits = []
    backoff = asyncio.Event()

    class Factory:
        async def create(self, spec):
            nonlocal attempts
            attempts += 1
            if failure_stage == "startup" and attempts < 3:
                raise TimeoutError("Sandbox startup timed out")
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            if failure_stage not in {"verifier", "cancel"} or attempts == 3:
                await machine.run(Command(("sh", "-c", "echo 1 > /workspace/reward")))
            return machine

    class Client:
        async def generate(self, request):
            continuation = request["chat_continuations"][0]
            if failure_stage == "cancel_model":
                backoff.set()
                await asyncio.Future()
            if failure_stage == "attempt" and attempts < 3:
                await asyncio.Future()
            if failure_stage == "model" and continuation is not None and attempts < 3:
                raise ModelServerError("unavailable", "request-1", 503)
            message = {"role": "assistant", "content": "Done"}
            if failure_stage == "model" and continuation is None:
                message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "write",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps(
                                    {
                                        "command": "test ! -f /workspace/dirty && echo retry > /workspace/dirty",
                                    }
                                ),
                            },
                        }
                    ],
                }
            offset = 0 if continuation is None else 2
            return {
                "assistant_messages": [message],
                "prompt_ids": [[1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]],
                "response_ids": [[10 * attempts + 3 + offset, 10 * attempts + 4 + offset]],
                "response_logprobs": [[-0.1, -0.2]],
                "stop_reasons": ["stop"],
                "token_provenance": TokenProvenance.ENGINE,
            }

    class VerifierFactory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            return machine

    async def wait(delay):
        waits.append(delay)
        for machine in machines:
            with pytest.raises(RuntimeError, match="closed"):
                await machine.run(Command(("true",)))
        backoff.set()
        if failure_stage == "cancel":
            await asyncio.Future()

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_retries": 2,
                    "min_wait_sec": 1,
                    "wait_multiplier": 2,
                    "max_wait_sec": 1.5,
                    "include_exceptions": [
                        "EnvironmentStartTimeoutError",
                        "ModelServerError",
                        "VerifierRuntimeError",
                        "TrialTimeoutError",
                    ],
                    "trial_attempt_timeout_sec": 1 if failure_stage == "attempt" else None,
                    "enable_error_classification": True,
                    "default_error_treatment": "mask",
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": Factory(), "verifier": VerifierFactory()},
        harbor=settings,
        retry_wait=wait,
        shutdown_timeout=30,
    )
    writer = Writer()
    operation = asyncio.create_task(
        worker.run_task(
            RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request),
            writer,
        )
    )
    if failure_stage in {"cancel", "cancel_model"}:
        await asyncio.wait_for(backoff.wait(), timeout=5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert writer.groups == []
        assert attempts == 1
        for machine in machines:
            with pytest.raises(RuntimeError, match="closed"):
                await machine.run(Command(("true",)))
        await worker.shutdown()
        return
    await operation
    assert waits == [1, 1.5]
    assert attempts == 3
    assert len(writer.groups) == 1
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["rollout_metrics"]["rollout_retries"] == 2
    if failure_stage == "model":
        assert batch["response_ids"] == [[33, 34, 90, 91, 35, 36]]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    else:
        assert batch["response_ids"] == [[33, 34]]
        assert batch["loss_masks"] == [[1, 1]]
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "empty",
        "invalid",
        "retry_missing",
        "exhausted",
        "passthrough",
        "grade_timeout_keep",
        "grade_timeout_discard",
    ],
)
async def test_harbor_retry_policy_preserves_terminal_grades(task_inputs, phase, failure):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    failing_command = {
        "missing": "true",
        "retry_missing": "true",
        "empty": "printf '' > /reward.txt",
        "invalid": "echo invalid > /reward.txt",
        "exhausted": "exit 7",
        "passthrough": "echo 1 > /reward.txt",
        "grade_timeout_keep": "echo 1 > /reward.txt",
        "grade_timeout_discard": "echo 1 > /reward.txt",
    }[failure]
    script = (
        failing_command
        if failure == "exhausted"
        else f'if [ "$(cat /workspace/attempt)" -eq 1 ]; then {failing_command}; else echo 1 > /reward.txt; fi'
    )
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", script),
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        artifacts=(VerifierArtifact(source="/workspace/attempt", target="/workspace/attempt", kind=ArtifactKind.FILE),),
    )
    # A command failure without a reward file is a missing-reward failure.
    # Use stdout grading to exercise the retry limit for verifier execution errors.
    if failure == "exhausted":
        verifier = ShellVerifierSpec(argv=("false",), artifacts=verifier.artifacts)
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=verifier.model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task,
        "shellbox",
        backend="shellsim",
        verifier_backend="verifier",
    ).model_dump_json()
    request["batch_metadata"] = BatchMetadata(0, phase)
    attempts = 0
    verifier_starts = 0
    waits = []

    class Factory:
        async def create(self, spec):
            nonlocal attempts
            attempts += 1
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            await machine.run(Command(("sh", "-c", f"echo {attempts} > /workspace/attempt")))
            return machine

    class VerifierFactory:
        async def create(self, spec):
            nonlocal verifier_starts
            verifier_starts += 1
            if failure.startswith("grade_timeout"):
                raise TimeoutError("Verifier machine startup timed out")
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    class Client(InferenceClient):
        async def generate(self, request):
            if failure == "passthrough" and request["chat_continuations"][0] is not None:
                raise TimeoutError("Agent deadline exceeded")
            output = await super().generate(request)
            if failure == "passthrough":
                output["assistant_messages"] = [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "work",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": json.dumps({"command": "echo work"}),
                                },
                            }
                        ],
                    }
                ]
            return output

    async def wait(delay):
        waits.append(delay)

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_retries": 2,
                    "enable_error_classification": True,
                    "default_error_treatment": "mask",
                    "passthrough_exceptions": ["AgentTimeoutError"],
                    "preserve_logprobs_on_timeout": failure != "grade_timeout_discard",
                    **(
                        {"exclude_exceptions": [], "include_exceptions": ["RewardFileNotFoundError"]}
                        if failure == "retry_missing"
                        else {}
                    ),
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": Factory(), "verifier": VerifierFactory()},
        harbor=settings,
        retry_wait=wait,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    retries = {"exhausted": 2, "retry_missing": 1}.get(failure, 0)
    assert attempts == retries + 1
    assert verifier_starts == retries + 1
    assert len(waits) == retries
    assert batch["rollout_metrics"]["rollout_retries"] == retries
    if failure in {"passthrough", "retry_missing"}:
        assert batch["unshaped_rewards"] == [1.0]
        assert batch["rewards"] == ([1.0] if failure == "passthrough" else [[0.0, 1.0]])
        assert batch["loss_masks"] == [[1, 1]]
        assert batch["exclude_from_baseline"] == [False]
    else:
        assert batch["unshaped_rewards"] == [0.0]
        assert batch["verification_results"][0].score is None
        assert batch["loss_masks"] == [[0, 0]]
        assert batch["exclude_from_baseline"] == [True]
    assert batch["response_ids"] == [[3, 4]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason,expected", [("stop", 1.3), ("length", 0.8)])
async def test_harbor_completion_reward_uses_the_engine_stop_reason(task_inputs, stop_reason, expected):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", "echo '===== 1 passed in 0.1s ====='; echo 1 > /reward.txt"),
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=verifier.model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task, "shellbox", backend="shellsim", verifier_backend="shellsim"
    ).model_dump_json()

    class Client(InferenceClient):
        async def generate(self, request):
            result = await super().generate(request)
            result["stop_reasons"] = [stop_reason]
            return result

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "enable_reward_shaping": True,
                    "reward_shaper": "composite_loop",
                    "reward_parser": "pytest",
                    "loop_shaping": {"terminate": {"enabled": True}},
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": FixtureImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["rewards"][0] == pytest.approx([0.0, expected])
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[1, 1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("shaper", ["threshold", "identity_aware_pass_ratio"])
async def test_strict_harbor_parser_masks_only_the_affected_response(task_inputs, shaper):
    config, request = task_inputs
    base = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    tasks = []
    for output in ("unrecognized output", "tests/test_task.py::test_answer PASSED"):
        verifier = ShellVerifierSpec(
            argv=("sh", "-c", f"echo '{output}'; echo 1 > /reward.txt"),
            reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        )
        tasks.append(
            base.model_copy(
                update={
                    "answer_type": AnswerType.STATE,
                    "tags": ("harbor",),
                    "verifier": VerifierSpec(
                        kind="shell",
                        environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                        parameters_json=verifier.model_dump_json(),
                    ),
                }
            )
        )
    request.update(
        prompts=request["prompts"] * 2,
        env_classes=["taskcompendium"] * 2,
        env_extras=[
            {
                "lowered_task_spec": lowered_task(
                    task, "shellbox", backend="shellsim", verifier_backend="shellsim"
                ).model_dump_json()
            }
            for task in tasks
        ],
        trajectory_ids=[TrajectoryID(base.id, index) for index in range(2)],
    )
    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "enable_reward_shaping": True,
                    "reward_shaper": shaper,
                    "reward_parser": "pytest",
                    "reward_shaping_fallback": False,
                    "enable_error_classification": True,
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {"shellsim": FixtureImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    assert batch["unshaped_rewards"] == [1.0, 1.0]
    assert [verdict.score for verdict in batch["verification_results"]] == [1.0, 1.0]
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[0, 0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["exception_types"] == ["VerifierOutputParseError", None]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_harbor_concurrency_does_not_queue_gym_tasks(task_inputs, phase):
    config, request = task_inputs
    base = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    tasks = [base.model_copy(update={"id": "harbor", "tags": ("harbor",)})] * 2 + [base]
    admitted = []
    active_harbor = 0
    maximum_harbor = 0
    ready = asyncio.Event()
    release = asyncio.Event()

    class WaitingClient(InferenceClient):
        async def generate(self, request):
            nonlocal active_harbor, maximum_harbor
            is_harbor = request["prompts"][0][0]["content"].startswith("Terminal task")
            admitted.append(is_harbor)
            if is_harbor:
                active_harbor += 1
                maximum_harbor = max(maximum_harbor, active_harbor)
            if len(admitted) == (2 if phase == "train" else 3):
                ready.set()
            await release.wait()
            if is_harbor:
                active_harbor -= 1
            return await super().generate(request)

    tasks[:2] = [
        task.model_copy(
            update={"context": ConversationInput(events=(TextMessage(role="user", content="Terminal task"),))}
        )
        for task in tasks[:2]
    ]
    request.update(
        prompts=request["prompts"] * 3,
        env_classes=["taskcompendium"] * 3,
        env_extras=[{"lowered_task_spec": lowered_task(task, "shellbox").model_dump_json()} for task in tasks],
        trajectory_ids=[TrajectoryID(str(index), 0) for index in range(3)],
        batch_metadata=BatchMetadata(0, phase),
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        WaitingClient(),
        {},
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"n_concurrent_trials": 2}})),
        concurrent_tasks=2 if phase == "train" else 3,
        concurrent_harbor_tasks=1,
        shutdown_timeout=30,
    )
    operation = asyncio.create_task(worker.run(request))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert admitted.count(False) == 1
        assert maximum_harbor == (1 if phase == "train" else 2)
    finally:
        release.set()
        try:
            batch = await operation
        finally:
            await worker.shutdown()
    assert batch["unshaped_rewards"] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize("model_failure", [False, True])
@pytest.mark.parametrize(
    "alias_file,alias_metadata,identifier",
    [
        ("config.json", {"instance_id": "Project-MONAI__MONAI-123"}, "project-monai__monai-123"),
        (
            "test_info.json",
            {"github_repo": "python-pillow/Pillow", "base_commit": "3ba81234"},
            "python-pillow__Pillow-3ba81234",
        ),
    ],
)
async def test_mixed_nemotron_tasks_run_without_the_original_sources(
    tmp_path, task_inputs, projection_type, alias_file, alias_metadata, identifier, model_failure
):
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "instruction.md").write_text("Repair the terminal task.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    (source / "tests" / alias_file).write_text(json.dumps(alias_metadata))
    (source / "tests/test.sh").write_text(
        "#!/bin/sh\necho 'tests/test_task.py::test_fix PASSED'\necho 0.75 > /logs/verifier/reward.txt\n"
    )
    rows = []
    for index, route in enumerate(("gym", "terminal_bench", "gym")):
        rows.append(
            {
                "prompt": [{"role": "user", "content": "What is six plus six?"}],
                "env_class": "gsm8k_multi_turn",
                "reward_spec": {"ground_truth": "12"},
                "teacher_route": "code" if route == "terminal_bench" else "math",
                "data_source": f"source-{index}",
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr1",
                        "agent": "swe" if route == "terminal_bench" else "arithmetic",
                        "route": route,
                        "terminal_bench_instance_id": identifier if route == "terminal_bench" else None,
                    }
                },
            }
        )
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    prepared = NemotronTaskDataset(
        [str(input_path)],
        Tokenizer(),
        100,
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
        terminal_bench_data=[str(source)],
        cache_dir=tmp_path / "tasks",
        num_workers=1,
    )
    shutil.rmtree(source)
    input_path.unlink()
    prompts, environments, extras, uids = zip(*[prepared[index] for index in range(len(prepared))], strict=True)

    class MixedClient:
        async def generate(self, request):
            if model_failure:
                raise ModelServerError("unavailable", "request-1", 503)
            continuation = request["chat_continuations"][0]
            prompt = [1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]
            output = await InferenceClient().generate(request)
            output["prompt_ids"] = [prompt]
            # Gym must retain its second turn despite the Harbor turn limit of one.
            output["assistant_messages"] = [
                {"role": "assistant", "content": "#### 13" if continuation is None else "#### 12"}
            ]
            return output

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins(), cpus=None))

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_turns": 1,
                    "override_cpus": 2,
                    "enable_reward_shaping": True,
                    "reward_shaper": "pass_ratio",
                    "reward_parser": "pytest",
                    "enable_token_reward_channel": False,
                    "enable_error_classification": True,
                    "default_error_treatment": "zero",
                    "max_retries": 0,
                }
            }
        )
    )
    config, request = task_inputs
    config.error_handling = {"enable_error_classification": True, "default_error_treatment": "mask"}
    request.update(
        prompts=list(prompts),
        env_classes=list(environments),
        env_extras=list(extras),
        trajectory_ids=[TrajectoryID(uid, 0) for uid in uids],
    )
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        MixedClient(),
        {"harbor": ImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "mixed"}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    if model_failure:
        assert batch["exclude_from_baseline"] == [True, False, True]
        assert batch["error_treatments"] == ["mask", "zero", "mask"]
        assert batch["loss_masks"] == [[0], [0], [0]]
        expected_rows = [0, 1, 2]
    elif projection_type is WholeTaskProjection:
        assert batch["unshaped_rewards"] == [1.0, 0.75, 1.0]
        assert [sum(reward) for reward in batch["rewards"]] == [1.1, 1.0, 1.1]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1], [1, 1], [1, 1, 0, 0, 1, 1]]
        expected_rows = [0, 1, 2]
    else:
        assert batch["unshaped_rewards"] == [0.0, 1.0, 0.75, 0.0, 1.0]
        assert [sum(reward) for reward in batch["rewards"]] == [0.1, 1.0, 1.0, 0.1, 1.0]
        assert batch["response_ids"] == [[3, 4]] * 5
        assert batch["loss_masks"] == [[1, 1]] * 5
        expected_rows = [0, 0, 1, 2, 2]
    assert batch["teacher_route_keys"] == [rows[index]["teacher_route"] for index in expected_rows]
    assert batch["data_sources"] == [rows[index]["data_source"] for index in expected_rows]
    assert [identity.instance_id for identity in batch["trajectory_ids"]] == [uids[index] for index in expected_rows]
    assert batch["rollout_metrics"]["nemotron_ultra/coverage/rlvr1/arithmetic"] == 2
    assert batch["rollout_metrics"]["nemotron_ultra/coverage/rlvr1/swe"] == 1
    assert "test_fix" not in json.dumps(prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "phase,treatment,preserve,train,exclude",
    [
        ("model_context", "passthrough", True, True, False),
        ("model_context", "zero", True, True, False),
        ("model_context", "mask", True, False, True),
        ("step", "passthrough", True, True, False),
        ("step", "zero", True, True, False),
        ("step", "passthrough", False, False, False),
        ("model_server", "mask", True, False, True),
        ("model_timeout", "passthrough", True, True, False),
        ("model_timeout", "passthrough", False, False, False),
        ("missing_logprobs", "passthrough", True, False, True),
        ("prepare", "mask", True, False, True),
    ],
)
async def test_interrupted_tasks_keep_verified_turns_and_apply_training_policy(
    task_inputs, projection_type, phase, treatment, preserve, train, exclude
):
    closed = []

    class InterruptedSession:
        def __init__(self, task, machine):
            self.turn = 0
            self.task = task
            self.grades = []

        async def prepare(self):
            if phase == "prepare":
                raise TimeoutError("private initialization details")
            return SessionStart(tuple(request["prompts"][0]), {})

        async def advance(self, turn):
            self.turn += 1
            if phase == "step" and self.turn == 2:
                raise TimeoutError("private environment details")
            grade = GradeResult(Outcome.GRADED, 1.0)
            self.grades.append(grade)
            return Transition(
                done=False, observations=({"role": "user", "content": "Continue"},), reward=1.0, grade=grade
            )

        async def grade(self, messages):
            return fold_grades(self.grades)

        async def close(self):
            closed.append(True)

    class InterruptedClient(ConversationClient):
        async def generate(self, request):
            if self.requests and phase == "model_timeout":
                raise TimeoutError("private serving details")
            if self.requests and phase == "missing_logprobs":
                raise ModelServerError("context_overflow", "request-123", 400)
            if self.requests and phase.startswith("model_"):
                category = "context_overflow" if phase == "model_context" else "constrained_decoding"
                raise ModelServerError(category, "request-123", 400 if phase == "model_context" else 500)
            output = await super().generate(request)
            if phase == "missing_logprobs":
                output["response_logprobs"] = None
            return output

    config, request = task_inputs
    exception_type = {
        "model_context": "ContextLengthExceededError",
        "model_server": "ModelServerError",
        "step": "AgentTimeoutError",
        "prepare": "AgentSetupTimeoutError",
        "model_timeout": "AgentTimeoutError",
        "missing_logprobs": "ContextLengthExceededError",
    }[phase]
    config.error_handling = {
        "enable_error_classification": True,
        f"{treatment}_exceptions": [exception_type],
        "preserve_logprobs_on_timeout": preserve,
    }
    task = source_task(
        request["prompts"][0],
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "interrupted").model_dump_json()}]
    request["env_classes"] = ["interrupted"]
    request["sampling_params"] = {"max_tokens": 3}
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InterruptedClient(["first", "unverified"]),
        {},
        sessions={"interrupted": InterruptedSession},
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert closed == [True]
    assert batch["error_treatments"] == [treatment]
    assert batch["exception_types"] == ["PassthroughWithoutLogprobs" if phase == "missing_logprobs" else exception_type]
    assert batch["exclude_from_baseline"] == [exclude]
    assert "private" not in str(batch["verification_results"])
    if phase != "prepare":
        assert batch["response_ids"] == [[3, 4]]
        np.testing.assert_allclose(
            batch["rollout_logprobs"], [[0.0, 0.0]] if phase == "missing_logprobs" else [[-0.1, -0.2]]
        )
        assert batch["loss_masks"] == ([[1, 1]] if train else [[0, 0]])
        assert batch["unshaped_rewards"] == [1.0]
        assert batch["rewards"] == [[0.0, 1.0 if treatment == "passthrough" else 0.0]]
        assert batch["evidence_messages"][0][-1]["content"] == "first"
        np.testing.assert_array_equal(batch["student_topk_indices"], [[[3, 99], [4, 99]]])
        assert batch["rollout_routed_experts"][0][:, 0, 0].tolist() == [1, 1]
    else:
        assert batch["response_ids"] == [[0]]
        assert batch["loss_masks"] == [[0]]
        assert batch["verification_results"][0].score is None
    if phase in {"model_context", "model_server", "missing_logprobs"}:
        assert batch["server_errors"] == [
            {
                "category": "constrained_decoding" if phase == "model_server" else "context_overflow",
                "request_id": "request-123",
                "status_code": 500 if phase == "model_server" else 400,
            }
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("violation", ["prefix", "logprobs", "provenance"])
async def test_invalid_model_evidence_aborts_the_group_even_with_error_masking(task_inputs, violation):
    class InvalidClient(ConversationClient):
        async def generate(self, request):
            output = await super().generate(request)
            if len(self.requests) == 2:
                if violation == "prefix":
                    output["prompt_ids"][0][0] = 99
                elif violation == "logprobs":
                    output["response_logprobs"][0].pop()
                else:
                    output["token_provenance"] = TokenProvenance.RECONSTRUCTED
            return output

    config, request = task_inputs
    config.error_handling = {"enable_error_classification": True, "default_error_treatment": "mask"}
    task = source_task(
        request["prompts"][0],
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json()}]
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InvalidClient(["#### 13", "#### 12"]),
        {},
        shutdown_timeout=30,
    )
    writer = Writer()
    with pytest.raises(ExceptionGroup) as failure:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    assert isinstance(failure.value.exceptions[0], RolloutContractError)
    assert writer.groups == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop_reason,last_token,mask", [("length", 4, [0, 0]), ("stop", 99, [1, 1]), ("stop", 4, [0, 0])]
)
async def test_overlong_filter_uses_sampled_end_tokens(task_inputs, stop_reason, last_token, mask):
    class StoppedClient(InferenceClient):
        async def generate(self, request):
            output = await super().generate(request)
            output["stop_reasons"] = [stop_reason]
            output["response_ids"] = [[3, last_token]]
            return output

    config, request = task_inputs
    config.apply_overlong_filtering = True
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        StoppedClient(),
        {},
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    assert batch["response_ids"] == [[3, last_token]]
    assert batch["loss_masks"] == [mask]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
async def test_source_tasks_run_without_the_original_dataset(tmp_path, task_inputs):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "What is six plus six?"}],
                    "env_class": "gsm8k",
                    "reward_spec": {"ground_truth": "12"},
                    "teacher_route": "math",
                    "data_source": "arithmetic",
                    "extra_info": {"grade": 3},
                }
            ]
        ),
        source,
    )
    prepared = SourceTaskDataset(
        [str(source)],
        Tokenizer(),
        100,
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
        num_workers=1,
    )
    source.unlink()
    prompt, env, extras, uid = prepared[0]
    config, _ = task_inputs
    client = ConversationClient(["#### 12"])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {},
        shutdown_timeout=30,
    )
    request = {
        "prompts": [prompt],
        "env_classes": [env],
        "env_extras": [extras],
        "trajectory_ids": [TrajectoryID(uid, 0)],
        "sampling_params": None,
        "batch_metadata": None,
    }
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": uid}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["teacher_route_keys"] == ["math"]
    assert batch["data_sources"] == ["arithmetic"]
    assert extras["extra_info"] == {"grade": 3}
    assert "ground_truth" not in json.dumps(client.requests[0]["prompts"])
    assert batch["response_ids"] == [[3, 4]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
async def test_harbor_source_materialization_runs_without_the_original_directory(tmp_path, task_inputs):
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "instruction.md").write_text("Write 12 to /logs/artifacts/answer.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    (source / "tests/test.sh").write_text(
        '#!/bin/sh\nif [ "$(cat /logs/artifacts/answer)" = "12" ]; then\n'
        "echo 0.75 > /logs/verifier/reward.txt\nelse\necho 0 > /logs/verifier/reward.txt\nfi\n"
    )
    prepared = HarborTaskDataset(
        [str(source)], Tokenizer(), 100, session=session_spec(), cache_dir=tmp_path / "tasks", num_workers=1
    )
    assert prepared.uid(0) == "source"
    shutil.rmtree(source)
    restored = TaskDataset([str(prepared.task_path)], Tokenizer(), 100, num_workers=1)
    prompt, env, extras, uid = restored[0]

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    client = ConversationClient(
        ["", "Done"],
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "answer",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo 12 > /logs/artifacts/answer"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Done"},
        ],
    )
    config, request = task_inputs
    request.update(prompts=[prompt], env_classes=[env], env_extras=[extras], trajectory_ids=[TrajectoryID(uid, 0)])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {"docker": ImageFactory()},
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": uid}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [0.75]
    assert batch["rollout_metrics"]["generate/task_rollout/tool_tasks"] == 1.0
    assert batch["response_ids"] == [[3, 4, 90, 91, 5, 6]]
    assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2, 0, 0, -0.1, -0.2]])
    assert all("reward.txt" not in json.dumps(request["prompts"]) for request in client.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("shaper,expected", [("pass_ratio", [0.5, 1.0]), ("identity_aware_pass_ratio", [0.0, 1.0])])
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_harbor_task_worker_preserves_verdicts_and_shapes_group_rewards(
    task_inputs, shaper, expected, projection_type, phase
):
    class SpanTokenizer(Tokenizer):
        def decode(self, tokens, *, skip_special_tokens):
            return "".join({3: "<think>Plan</think>", 4: "apply_patch file"}[token] for token in tokens)

    config, request = task_inputs
    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "enable_reward_shaping": True,
                    "reward_shaper": shaper,
                    "reward_parser": "pytest",
                    "override_timeout_sec": 30,
                    "verifier_override_timeout_sec": 5,
                    "enable_token_reward_channel": True,
                }
            }
        )
    )
    tasks = []
    for index, verdict in enumerate(("FAILED", "PASSED")):
        verifier = ShellVerifierSpec(
            argv=(
                "sh",
                "-c",
                f"printf 'tests/test_task.py::test_common PASSED\\n"
                f"tests/test_task.py::test_changed {verdict}\\n'; echo 0 > /reward.txt",
            ),
            reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        )
        tasks.append(
            TaskSpec(
                id=f"harbor-{index}",
                context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
                environment_requirements=EnvironmentRequirements(),
                answer_type=AnswerType.STATE,
                verifier=VerifierSpec(
                    kind="shell",
                    environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                    parameters_json=verifier.model_dump_json(),
                ),
                source=Source(dataset="harbor", revision="1", row=str(index), importer_revision="1"),
                tags=("harbor",),
            )
        )
    request.update(
        prompts=[[{"role": "user", "content": "Complete the task."}]] * 2,
        env_classes=["taskcompendium"] * 2,
        env_extras=[
            {
                "lowered_task_spec": lowered_task(
                    task, "shellbox", backend="shellsim", verifier_backend="shellsim"
                ).model_dump_json()
            }
            for task in tasks
        ],
        trajectory_ids=[TrajectoryID("harbor", index) for index in range(2)],
        batch_metadata=BatchMetadata(0, phase),
    )
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, SpanTokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InferenceClient(),
        {"shellsim": FixtureImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "harbor"}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [0.0, 0.0]
    assert [sum(reward) for reward in batch["rewards"]] == expected
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[1, 1], [1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2], [-0.1, -0.2]])
    if phase == "train":
        assert batch["response_span_tags"] == [[1, 3], [1, 3]]
        assert batch["token_level_shaping"] == [[0.0, 0.0], [0.0, 0.0]]
    else:
        assert "response_span_tags" not in batch
        assert "token_level_shaping" not in batch


@pytest.fixture
def genrm_judge_server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            metadata = payload["metadata"]
            server.comparisons.append(metadata)
            first_better = metadata["response_1"] == "better"
            scores = (
                {"score_1": 5, "score_2": 1, "ranking": 1}
                if first_better
                else {"score_1": 1, "score_2": 5, "ranking": 6}
            )
            body = json.dumps(
                {
                    "status": server.status,
                    "output": [
                        {
                            "type": "message",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(scores)}],
                        }
                    ],
                }
            ).encode()
            self.send_response(server.http_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.comparisons = []
    server.status = "completed"
    server.http_status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("valid_peers", [0, 2])
def test_genrm_ineligible_attempts_have_no_provisional_score(task_inputs, genrm_judge_server, valid_peers):
    _, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    specification = GroupGraderSpec(
        name="nemotron_genrm",
        parameters_json=GenRMGroupGraderParameters(
            principle="Prefer the correct answer.",
            agent="genrm_simple_agent",
            config={
                "judge": {"base_url": f"http://127.0.0.1:{genrm_judge_server.server_port}", "model": "judge"},
                "reasoning_bonus": 0,
                "answer_bonus": 0,
                "group_reasoning_length_penalty_coeff": 0,
                "group_answer_length_penalty_coeff": 0,
            },
        ).model_dump_json(),
    )
    pending_grade = GradeResult(Outcome.GRADED, 3.0, passed=True, score_min=1.0, score_max=5.0)
    records = []
    for text in ("better", "worse", "excluded"):
        message = {"role": "assistant", "content": text}
        turn = ModelTurn(message, (1, 2), (3, 4), (-0.1, -0.2), "stop", text=text)
        records.append(
            RolloutData(
                task_id=task.id,
                messages=(message,),
                prompt_token_ids=(1, 2),
                response_token_ids=(3, 4),
                loss_mask=(1, 1),
                logprobs=(-0.1, -0.2),
                grade=pending_grade,
                stop_reason="stop",
                steps=(RolloutStep(turn, Transition(done=True, reward=3.0, grade=pending_grade), 1, (message,)),),
            )
        )
    result = grade_genrm_rollouts(task, specification, records, [index < valid_peers for index in range(3)], "train")

    assert [record.grade.reward for record in result] == ([5.0, 1.0, None] if valid_peers else [None] * 3)
    for record in result[valid_peers:]:
        assert record.grade.status == Outcome.UNAVAILABLE
        assert record.steps[0].transition.reward is None
    assert all(
        "excluded" not in (comparison["response_1"], comparison["response_2"])
        for comparison in genrm_judge_server.comparisons
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("verifyit_enabled", [False, True])
@pytest.mark.parametrize(
    "phase,grading,failed_peer,failed_judge",
    [
        ("train", "verify", False, False),
        ("eval", "verify", False, False),
        ("train", "skip", False, False),
        ("train", "verify", True, False),
        ("train", "verify", True, True),
    ],
)
async def test_genrm_final_grades_and_credit_reach_training_batch(
    task_inputs, genrm_judge_server, phase, grading, failed_peer, failed_judge, verifyit_enabled
):
    comparisons = genrm_judge_server.comparisons
    if failed_judge:
        if verifyit_enabled:
            genrm_judge_server.status = "in_progress"
        else:
            genrm_judge_server.http_status = 400
    config, request = task_inputs
    count = 3 if failed_peer else 2
    task = source_task(
        request["prompts"][0],
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "genrm_simple_agent",
                    "record_json": json.dumps({"principle": "Prefer the correct answer."}),
                    "request_json": "{}",
                }
            }
        },
        config={
            "grading": grading,
            "verifyit_enabled": verifyit_enabled,
            "genrm": {
                "genrm_parse_retries": 0,
                "default_score": 0,
                "reasoning_bonus": 0,
                "answer_bonus": 0,
                "group_reasoning_length_penalty_coeff": 0,
                "group_answer_length_penalty_coeff": 0,
                "judge": {"base_url": f"http://127.0.0.1:{genrm_judge_server.server_port}", "model": "judge"},
            },
        },
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    group_grader = task_group_grader(lowered_task(task, "nemotron_ultra"))
    request.update(
        prompts=request["prompts"] * count,
        env_classes=["nemotron_ultra"] * count,
        env_extras=[
            {
                "lowered_task_spec": lowered_task(task, "nemotron_ultra").model_dump_json(),
                "group_grader": None if group_grader is None else group_grader.model_dump_json(),
            }
        ]
        * count,
        trajectory_ids=[TrajectoryID(task.id, index) for index in range(count)],
        batch_metadata=BatchMetadata(global_step=0, training_phase=phase),
    )

    class CohortClient(ConversationClient):
        async def generate(self, request):
            if failed_peer and len(self.requests) == 1:
                self.requests.append(request)
                raise ModelServerError("constrained_decoding", "failed-peer", 500)
            return await super().generate(request)

    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        CohortClient(["better", "failed", "worse"] if failed_peer else ["better", "worse"]),
        {},
        shutdown_timeout=30,
    )
    rollouts = await runner.generate(request)
    batch = await runner.training_batch(request, rollouts)
    if failed_peer:
        failed_indices = [index for index, rollout in enumerate(rollouts) if rollout.failure is not None]
        assert len(failed_indices) == 1
        assert batch["rollout_metrics"]["generate/failed_trajectory_fraction"] == pytest.approx(
            1.0 if failed_judge else 1 / count
        )
        failed_index = failed_indices[0]
        assert batch["exception_types"][failed_index] == "ModelServerError"
        assert batch["verification_results"][failed_index].score is None
        assert batch["response_ids"][failed_index] == [0]
        assert batch["loss_masks"][failed_index] == [0]
        assert batch["exclude_from_baseline"][failed_index] is True
        if failed_judge:
            # A failed comparison can cancel pending judge requests.
            assert comparisons
            assert all({pair["response_1"], pair["response_2"]} == {"better", "worse"} for pair in comparisons)
            assert all(not any(mask) for mask in batch["loss_masks"])
            assert all(grade.score is None for grade in batch["verification_results"])
            assert batch["rewards"] == [0.0, 0.0, 0.0]
        else:
            assert {pair["response_1"] for pair in comparisons} == {"better", "worse"}
            for index, rollout in enumerate(rollouts):
                if index == failed_index:
                    assert batch["rewards"][index] == [0.0]
                else:
                    reward = {"better": 5.0, "worse": 1.0}[rollout.steps[-1].turn.text]
                    assert batch["rewards"][index] == [0.0, reward]
                    assert batch["loss_masks"][index] == [1, 1]
        return
    if phase == "train" and grading == "verify":
        expected_rewards = [{"better": 5.0, "worse": 1.0}[rollout.steps[-1].turn.text] for rollout in rollouts]
        assert [rollout.grade.reward for rollout in rollouts] == expected_rewards
        assert batch["rewards"] == [[0.0, reward] for reward in expected_rewards]
        assert batch["unshaped_rewards"] == expected_rewards
        assert [normalized_verifier_score(result) for result in batch["verification_results"]] == [
            {"better": 1.0, "worse": 0.0}[rollout.steps[-1].turn.text] for rollout in rollouts
        ]
        assert batch["loss_masks"] == [[1, 1], [1, 1]]
        assert {pair["response_1"] for pair in comparisons} == {"better", "worse"}
    else:
        assert comparisons == []
        assert [rollout.grade.reward for rollout in rollouts] == [None, None]
        assert batch["rewards"] == ([[0.0, 0.0], [0.0, 0.0]] if grading == "skip" else [0.0, 0.0])
        expected_mask = [1, 1] if grading == "skip" else [0, 0]
        assert batch["loss_masks"] == [expected_mask, expected_mask]
        assert batch["exclude_from_baseline"] == [grading != "skip"] * 2


@pytest.mark.asyncio
async def test_task_group_grader_preserves_separate_samples_and_private_inputs(task_inputs):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    task = task.model_copy(
        update={
            "verifier": VerifierSpec(
                kind="skipped", parameters_json=json.dumps({"reason": "Group grading supplies the final score"})
            ),
        }
    )
    group_grader = GroupGraderSpec(name="group_total", parameters_json=json.dumps({"private_offset": 10}))
    request.update(
        prompts=request["prompts"] * 4,
        env_classes=["taskcompendium"] * 4,
        env_extras=[
            {
                "lowered_task_spec": lowered_task(task, "shellbox").model_dump_json(),
                "group_grader": group_grader.model_dump_json(),
            }
        ]
        * 4,
        trajectory_ids=[TrajectoryID(group, sample) for group, sample in [("a", 0), ("b", 0), ("a", 1), ("b", 1)]],
    )

    def group_total(task, specification, records, eligible, _phase):
        parameters = json.loads(specification.parameters_json)
        total = sum(int(record.steps[-1].turn.text) for record, valid in zip(records, eligible, strict=True) if valid)
        return [
            replace(
                record,
                grade=GradeResult(
                    Outcome.GRADED,
                    total + parameters["private_offset"],
                    score_max=100,
                ),
            )
            for record in records
        ]

    model = ConversationClient(["1", "2", "3", "4"])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        model,
        {},
        concurrent_tasks=1,
        group_graders={"group_total": group_total},
        shutdown_timeout=30,
    )
    rollouts = await worker.generate(request)
    batch = await worker.training_batch(request, rollouts)
    assert batch["rewards"] == [14, 16, 14, 16]
    assert batch["loss_masks"] == [[1, 1]] * 4
    assert [record.response_token_ids for record in rollouts] == [(3, 4), (5, 6), (7, 8), (9, 10)]
    assert all("private_offset" not in str(item["prompts"]) for item in model.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment,extras,responses,rewards",
    [
        ("cat_count", {"extra_info": {"n": 2}}, ["cat cat"], [0.0, 1.0]),
        ("gsm8k", {"extra_info": None, "reward_spec": {"ground_truth": "12"}}, ["#### 12"], [0.0, 1.0]),
        ("gsm8k", {"reward_spec": {"ground_truth": "12"}}, ["#### 13"], [0.0, 0.0]),
        ("mcq", {"reward_model": {"ground_truth": "B"}}, [r"\boxed{B}"], [0.0, 1.0]),
        (
            "gsm8k_multi_turn",
            {"reward_spec": {"ground_truth": "12"}},
            ["#### 13", "#### 12"],
            [0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        ),
    ],
)
async def test_unified_gym_tasks_preserve_grading_and_turn_credit(task_inputs, environment, extras, responses, rewards):
    config, request = task_inputs
    task = source_row_task(
        {"prompt": request["prompts"][0], "env_class": environment, **extras},
        0,
        source_name="fixture",
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = [environment]
    model = ConversationClient(responses)
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        model,
        {},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert batch["rewards"] == [rewards]
    expected_grade = float(responses[-1] == "#### 12") if environment == "gsm8k_multi_turn" else sum(rewards)
    assert batch["unshaped_rewards"] == [expected_grade]
    assert batch["verification_results"][0].score == expected_grade
    if environment == "cat_count":
        assert batch["verification_results"][0].passed is True
        assert batch["env_metrics"][0]["exact_n2"] == 1.0
    expected_mask = [1, 1] if len(responses) == 1 else [1, 1, 0, 0, 1, 1]
    assert batch["loss_masks"] == [expected_mask]
    assert batch["rollout_routed_experts"][0][:, 0, 0].tolist() == expected_mask
    assert len(batch["student_topk_indices"][0]) == len(expected_mask)
    assert all("ground_truth" not in str(item["prompts"]) for item in model.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row_limits,session_config,turns",
    [
        ({"max_turns": 2}, {}, 2),
        ({"extra_info": {"max_turns": 2}}, {}, 2),
        ({"max_turns": None, "extra_info": {"max_turns": 2}}, {}, 2),
        ({"max_turns": 2, "extra_info": {"max_turns": 1}}, {}, 2),
        ({"max_turns": 2}, {"session": {"max_turns": 1}}, 1),
        ({}, {}, 1),
    ],
)
async def test_source_row_turn_limit_controls_correction_and_terminal_grade(
    task_inputs, row_limits, session_config, turns
):
    config, request = task_inputs
    task = source_row_task(
        {
            "prompt": request["prompts"][0],
            "env_class": "gsm8k_multi_turn",
            "reward_spec": {"ground_truth": "12"},
            **row_limits,
        },
        0,
        source_name="fixture",
        environment_configs={
            "session": session_spec(max_turns=1).model_dump(exclude={"task_session"}),
            "gsm8k_multi_turn": session_config,
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    model = ConversationClient(["#### 13", "#### 12"])
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), model, {}, shutdown_timeout=30
    )
    batch = await worker.run(request)
    assert len(model.requests) == turns
    assert batch["unshaped_rewards"] == [float(turns == 2)]
    assert batch["verification_results"][0].passed is (turns == 2)
    assert sum(batch["rewards"][0]) == pytest.approx(0.2 / turns + float(turns == 2))


@pytest.mark.asyncio
async def test_materialized_turn_limit_precedes_source_row_metadata(task_inputs):
    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        {"reward_spec": {"ground_truth": "12"}, "max_turns": 1},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    model = ConversationClient(["#### 13", "#### 12"])
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), model, {}, shutdown_timeout=30
    )
    batch = await worker.run(request)
    assert len(model.requests) == 2
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["verification_results"][0].passed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("reward_key", ["reward_spec", "reward_model"])
async def test_aime_rollout_preserves_length_reward_and_phase_metrics(
    task_inputs, delivered_telemetry, enable_thinking, reward_key
):
    config, request = task_inputs
    config.chat_template_kwargs = {"enable_thinking": enable_thinking}
    task = source_task(
        request["prompts"][0],
        extras={reward_key: {"ground_truth": "12"}},
        config={"length_penalty_weight": 1.0, "min_response_length": 0, "evaluation_token_budget": 1},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "aime").model_dump_json()}]
    request["sampling_params"] = {"max_tokens": 4, "logprobs": 0}
    request["env_classes"] = ["aime"]
    engine = AsyncMock()
    engine.model_name = "fixture"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = r"Answer: \boxed{12}"

    async def tokenize(payload):
        thinking = payload["json"].get("chat_template_kwargs", {}).get("enable_thinking", True)
        return {"tokens": [1, 2, 20] if thinking else [1, 2]}

    engine.tokenize.side_effect = tokenize

    async def serve(payload):
        assert payload["json"]["max_completion_tokens"] == 4
        assert payload["json"]["chat_template_kwargs"]["enable_thinking"] is enable_thinking
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": r"Answer: \boxed{12}"},
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ]
        }

    engine.chat_completion.side_effect = serve
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        DirectModelClient(engine),
        {},
        max_verifier_workers=1,
        shutdown_timeout=30,
    )
    try:
        with observe_rollout_call(step=3, mode="async", enabled=True):
            batch = await runner.run(request)
    finally:
        await runner.shutdown()
    assert batch["rewards"][0] == pytest.approx([0.0, 0.5])
    assert batch["prompt_token_ids"] == [[1, 2, 20] if enable_thinking else [1, 2]]
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["loss_masks"] == [[1, 1]]
    assert batch["verification_results"][0].diagnostics["over_evaluation_budget"] is True
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    phases = {
        row["attributes"]["phase"]: row["attributes"].get("parent")
        for row in delivered_telemetry.select("phase_duration_seconds", root="rollout_call", step="3")
    }
    assert phases == {
        "rollout_call": None,
        "rollout_collect": "rollout_call",
        "rollout_tokenize": "rollout_collect",
        "rollout_assemble": "rollout_call",
        "rollout_finalize": "rollout_call",
        "rollout_call_residual": "rollout_call",
    }
    waits = {row["attributes"]["wait"]: row["value"] for row in delivered_telemetry.select("rollout_waits", step="3")}
    assert waits["model_client_await"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "max_model_len,max_input_length,output_limit,expected_budgets",
    [
        (5, 100, None, [3]),
        (5, 100, 2, [2]),
        (2, 100, None, []),
        (None, 4, None, [10]),
        (None, 1, None, []),
        (100, 4, None, [10]),
        (100, 1, None, []),
    ],
)
async def test_context_limits_preserve_only_completed_gym_turns(
    task_inputs, projection_type, max_model_len, max_input_length, output_limit, expected_budgets
):
    config, request = task_inputs
    config.max_input_length = max_input_length
    config.engine_init_kwargs = {"max_model_len": max_model_len}
    config.sampling_params.logprobs = 0
    request["sampling_params"] = None if output_limit is None else {"max_tokens": output_limit}
    task = source_task(
        request["prompts"][0],
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    engine = AsyncMock()
    engine.model_name = "fixture"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "#### 13"
    budgets = []

    async def tokenize(payload):
        tokens = []
        for index, message in enumerate(payload["json"]["messages"]):
            if message["role"] == "assistant":
                tokens.extend([3, 4, 99] if message["content"] else [99])
            else:
                tokens.extend([1, 2] if index == 0 else [90, 91])
        return {"tokens": tokens}

    async def serve(payload):
        budgets.append(payload["json"]["max_completion_tokens"])
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "#### 13"},
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ]
        }

    engine.tokenize.side_effect = tokenize
    engine.chat_completion.side_effect = serve
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(config, projection_type(projection), DirectModelClient(engine), {}, shutdown_timeout=30)
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert budgets == expected_budgets
    assert batch["stop_reasons"] == ["length"]
    assert batch["prompt_token_ids"] == [[1, 2]]
    if expected_budgets:
        assert batch["response_ids"] == [[3, 4]]
        assert batch["loss_masks"] == [[1, 1]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
        assert batch["unshaped_rewards"] == [0.0]
        assert batch["rewards"] == [[0.0, 0.1]]
        assert batch["evidence_messages"][0][-1] == {"role": "assistant", "content": "#### 13"}
    else:
        assert batch["response_ids"] == [[0]]
        assert batch["loss_masks"] == [[0]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[0.0]])
        assert batch["verification_results"][0].score is None
        assert batch["exclude_from_baseline"] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_native_rewards_and_credit_remain_aligned_across_tool_observations(task_inputs, projection_type):
    class CreditSession:
        def __init__(self, task, machine):
            self.turn = 0
            self.grades = []

        async def prepare(self):
            return SessionStart(tuple(request["prompts"][0]), {})

        async def advance(self, turn):
            self.turn += 1
            grade = GradeResult(Outcome.GRADED, float(self.turn))
            self.grades.append(grade)
            return Transition(
                observations=({"role": "user", "content": "Continue"},),
                done=self.turn == 2,
                grade=grade,
                reward=0.5 * self.turn,
                token_rewards=(0.2 * self.turn, 0.3 * self.turn),
                token_credit=(-0.1 * self.turn, 0.0),
                reward_components={"penalty": -0.5 * self.turn},
            )

        async def grade(self, messages):
            return fold_grades(self.grades)

        async def close(self):
            pass

    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "credit").model_dump_json()}]
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    runner = TaskRolloutWorker(
        config,
        projection_type(projection),
        ConversationClient(["first", "second"]),
        {},
        sessions={"credit": CreditSession},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    if projection_type is WholeTaskProjection:
        assert batch["rewards"] == [[0.2, 0.3, 0.0, 0.0, 0.4, 0.6]]
        assert batch["token_level_shaping"] == [[-0.1, 0.0, 0.0, 0.0, -0.2, 0.0]]
        assert batch["unshaped_rewards"] == [1.0]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    else:
        assert batch["rewards"] == [[0.2, 0.3], [0.4, 0.6]]
        assert batch["token_level_shaping"] == [[-0.1, 0.0], [-0.2, 0.0]]
        assert batch["unshaped_rewards"] == [1.0, 2.0]
        assert batch["loss_masks"] == [[1, 1], [1, 1]]


@pytest.mark.asyncio
async def test_lean_refinement_discards_the_failed_attempt(task_inputs, task_machine):
    config, request = task_inputs
    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "task_session",
                "agent": "math_formal_lean_refinement_agent",
                "record_json": json.dumps({"header": "", "formal_statement": "theorem equality : 1 = 1 := by sorry"}),
                "request_json": "{}",
            }
        }
    }
    task = source_task(
        request["prompts"][0],
        extras,
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [
        {"lowered_task_spec": lowered_task(task, "nemotron_ultra", backend="shellsim").model_dump_json()}
    ]
    request["env_classes"] = ["nemotron_ultra"]
    client = ConversationClient(["", "by rfl"])
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert batch["response_ids"] == [[5, 6]]
    assert batch["loss_masks"] == [[1, 1]]
    assert batch["rewards"] == [[0.0, 1.0]]
    assert client.requests[1]["chat_continuations"] == [None]
    assert task_machine.closed


@pytest.mark.asyncio
async def test_step_projection_preserves_served_prompts_grades_and_teacher_routes(task_inputs):
    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        extras={"reward_spec": {"ground_truth": "12"}},
        config={},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [
        {"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json(), "teacher_route": "arithmetic"}
    ]
    request["env_classes"] = ["gsm8k_multi_turn"]
    runner = TaskRolloutWorker(
        config,
        StepTaskProjection(StepWiseTrajectoryProjection(config, Tokenizer())),
        ConversationClient(["#### 13", "#### 12"]),
        {},
        shutdown_timeout=30,
    )
    writer = Writer()
    await runner.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["prompt_token_ids"] == [[1, 2], [1, 2, 3, 4, 90, 91]]
    assert batch["response_ids"] == [[3, 4], [5, 6]]
    assert batch["loss_masks"] == [[1, 1], [1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2], [-0.1, -0.2]])
    assert batch["rewards"] == [[0.0, 0.1], [0.0, 1.0]]
    assert batch["unshaped_rewards"] == [0.0, 1.0]
    assert [grade.score for grade in batch["verification_results"]] == [0.0, 1.0]
    assert [item.step for item in batch["trajectory_ids"]] == [0, 1]
    assert batch["is_last_step"] == [False, True]
    assert batch["teacher_route_keys"] == ["arithmetic", "arithmetic"]
    assert [messages[-1]["content"] for messages in batch["evidence_messages"]] == ["#### 13", "#### 12"]
    np.testing.assert_array_equal(batch["student_topk_indices"], [[[3, 99], [4, 99]], [[5, 99], [6, 99]]])
    assert all(routes[:, 0, 0].tolist() == [1, 1] for routes in batch["rollout_routed_experts"])


@pytest.fixture
def task_machine():
    """Supply compiler and interpreter I/O for the training-projection tests."""

    class Machine:
        closed = False
        output = "4"

        async def create(self, spec):
            self.output = spec.env.get("PYTHON_OUTPUT", "4")
            return self

        async def upload(self, source, target):
            pass

        async def run(self, command):
            data = json.loads(command.stdin) if command.stdin else {}
            if "proof" in data:
                output = {
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "reason": "exited",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
            elif "code" in data:
                output = {
                    "exit_code": 0,
                    "stdout": self.output,
                    "stderr": "",
                    "reason": "exited",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
            else:
                output = {}
            return Result(0, json.dumps(output).encode(), b"", False, False, ExitReason.EXITED)

        async def close(self):
            self.closed = True

    return Machine()


@pytest.mark.asyncio
@pytest.mark.parametrize("per_agent", [False, True])
async def test_source_machine_selection_controls_tool_results(task_inputs, task_machine, per_agent):
    config, request = task_inputs
    environment = {
        "machine": {
            "requirements": {"environment_variables": {"PYTHON_OUTPUT": "4"}},
            "runtime": machine_runtime("shellsim").model_dump(),
        },
        "machines": {
            "ns_tools_simple_agent": {
                "requirements": {"environment_variables": {"PYTHON_OUTPUT": "7"}},
                "runtime": machine_runtime("shellsim").model_dump(),
            }
        }
        if per_agent
        else {},
    }
    expected = "7" if per_agent else "4"
    row = {
        "prompt": request["prompts"][0],
        "env_class": "nemotron_ultra",
        "extra_info": {
            "nemotron_ultra": {
                "route": "task_session",
                "agent": "ns_tools_simple_agent",
                "record_json": json.dumps({"expected_answer": expected}),
                "request_json": "{}",
            }
        },
    }
    task = source_row_task(
        row,
        0,
        source_name="source",
        environment_configs={
            "session": session_spec().model_dump(exclude={"task_session"}),
            "nemotron_ultra": environment,
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "python-1", "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"2+2"}'}}
        ],
    }
    model = ConversationClient(["", expected], messages=[message, {"role": "assistant", "content": expected}])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        model,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert model.requests[1]["prompts"][0][-1] == {
        "role": "tool",
        "tool_call_id": "python-1",
        "content": expected,
    }
    assert batch["verification_results"][0].score == 1.0
    assert batch["exclude_from_baseline"] == [False]
    assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    assert task_machine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "session,operation,cause_type",
    [
        ("searchcode", "prepare", "RuntimeError"),
        ("nemotron_ultra", "prepare", "RuntimeError"),
        ("searchcode", "start", "UnsupportedMachineSpec"),
        ("searchcode", "start", "RuntimeError"),
    ],
)
async def test_python_startup_failure_is_masked_without_losing_its_graded_peer(
    task_inputs, task_machine, projection_type, session, operation, cause_type
):
    if operation == "start":
        startup_error = UnsupportedMachineSpec if cause_type == "UnsupportedMachineSpec" else RuntimeError
        task_machine.create = AsyncMock(side_effect=startup_error("The task provider did not start the machine"))
    task_machine.run = AsyncMock(
        return_value=Result(1, b"", b"ModuleNotFoundError: No module named 'IPython'", False, False, ExitReason.EXITED)
    )
    config, original = task_inputs
    config.error_handling = {"enable_error_classification": True}
    extras = (
        {"reward_spec": {"ground_truth": "12"}}
        if session == "searchcode"
        else {
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "ns_tools_simple_agent",
                    "record_json": json.dumps({"expected_answer": "12"}),
                    "request_json": "{}",
                }
            }
        }
    )
    failed = source_task(
        original["prompts"][0],
        extras,
        {},
        Source(dataset="python-startup", revision="1", row="0", importer_revision="1"),
    )
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": [session, "taskcompendium"],
        "env_extras": [
            {
                "lowered_task_spec": lowered_task(failed, session, backend="shellsim").model_dump_json(),
                "teacher_route": "python-startup",
            },
            original["env_extras"][0],
        ],
        "trajectory_ids": [TrajectoryID(failed.id, 0), TrajectoryID("arithmetic", 0)],
    }
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config, projection_type(projection), InferenceClient(), {"shellsim": task_machine}, shutdown_timeout=30
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": failed.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch
    assert batch["response_ids"] == [[0], [3, 4]]
    assert batch["loss_masks"] == [[0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["exception_types"] == [
        "TaskMachineError" if operation == "start" and cause_type == "RuntimeError" else "VerifierRuntimeError",
        None,
    ]
    assert batch["error_treatments"] == ["mask", None]
    assert batch["unshaped_rewards"] == [0.0, 1.0]
    failed_result, peer_result = batch["verification_results"]
    assert failed_result.status is VerificationStatus.ERROR
    assert failed_result.diagnostics["operation"] == operation
    assert failed_result.diagnostics["cause_error_type"] == cause_type
    assert peer_result.status is VerificationStatus.VERIFIED
    assert peer_result.score == 1.0
    if operation == "prepare":
        assert task_machine.closed


@pytest.fixture
def python_tool_task(task_inputs):
    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "ns_tools_simple_agent",
                    "record_json": json.dumps({"question": "What is 2 + 2?", "expected_answer": "4"}),
                    "request_json": "{}",
                }
            }
        },
        config={},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "python-1",
                "type": "function",
                "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"2 + 2"}'},
            }
        ],
    }
    task = lowered_task(task, "nemotron_ultra", backend="shellsim")
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    return task, message


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_python_tool_observations_preserve_training_and_release_session(
    python_tool_task, task_inputs, task_machine, projection_type
):
    config, request = task_inputs
    _, message = python_tool_task
    model = ConversationClient(["", "4"], messages=[message, {"role": "assistant", "content": "4"}])
    projection = (
        WholeTrajectoryProjection(config, Tokenizer())
        if projection_type is WholeTaskProjection
        else StepWiseTrajectoryProjection(config, Tokenizer())
    )
    runner = TaskRolloutWorker(
        config,
        projection_type(projection),
        model,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert task_machine.closed
    assert model.requests[1]["prompts"][0][-1] == {"role": "tool", "tool_call_id": "python-1", "content": "4"}
    if projection_type is WholeTaskProjection:
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
        assert batch["rewards"] == [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        assert batch["exclude_from_baseline"] == [False]
    else:
        assert batch["loss_masks"] == [[1, 1], [1, 1]]
        assert batch["rewards"] == [[0.0, 0.0], [0.0, 1.0]]
        assert batch["exclude_from_baseline"] == [False, False]


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
async def test_worker_keeps_blocking_inference_available_during_rollout(task_inputs):
    class BlockingClient(InferenceClient):
        async def generate(self, request):
            await asyncio.to_thread(threading.get_ident)
            return await super().generate(request)

    config, request = task_inputs
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        BlockingClient(),
        {},
        concurrent_tasks=1,
        shutdown_timeout=30,
    )
    writer = Writer()
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    try:
        await asyncio.wait_for(
            worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "arithmetic"}, request), writer),
            timeout=5,
        )
        assert writer.groups[0][1].trajectory_batch["unshaped_rewards"] == [1.0]
    finally:
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
