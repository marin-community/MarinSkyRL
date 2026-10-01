"""Task Parquet through canonical execution and the leased buffer writer."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import json
import shutil
import threading
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import pytest
from omegaconf import OmegaConf
from skyrl_gym.envs.base_text_env import BaseTextEnv
from skyrl_gym.envs.registration import EnvSpec, registry
from skyrl_gym.verification import RewardResult, VerificationResult, VerificationStatus, normalized_verifier_score
from taskcompendium.grading import numeric_answer
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.image import DockerfileSource
from shellbox.machine import Command, ShellSimBuiltins
from taskcompendium.environment import (
    EnvironmentKind,
    EnvironmentSpec,
    FileReward,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
)
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    Source,
    StageRewardStrategy,
    StageVerifierSpec,
    EnvironmentRequirements,
    TaskSpec,
    TaskStage,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)

from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.dataset.tasks import GymTaskDataset, TaskDataset
from skyrl_train.dataset.harbor import HarborTaskDataset
from skyrl_train.dataset.nemotron_ultra import NemotronTaskDataset
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from taskcompendium.importers.skyrl import gym_task, read_gym_tasks
from taskcompendium.parquet import read_tasks, write_tasks
from taskcompendium.rollout import ModelTurn, RolloutContractError
from skyrl_train.rollouts.gym_tasks import GymTaskSession
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


@dataclass
class Writer:
    groups: list[tuple[RolloutLease, RolloutGroup]] = field(default_factory=list)

    async def write_rollout(self, lease, group):
        self.groups.append((lease, group))


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(12, tolerance_abs=0, tolerance_rel=0),
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
            {"task_spec": task.model_dump_json(), "teacher_route": "arithmetic", "data_source": "arithmetic"}
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
        command_timeout=5,
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
        command_timeout=5,
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


@pytest.mark.asyncio
async def test_model_failure_does_not_commit_a_partial_group(task_inputs):
    first_response = asyncio.Event()

    class FailedClient:
        async def generate(self, request):
            if not first_response.is_set():
                response = await InferenceClient().generate(request)
                first_response.set()
                return response
            raise ConnectionError("Inference endpoint unavailable")

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
        command_timeout=5,
    )
    with pytest.raises(ExceptionGroup) as failure:
        await runner.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "task"}, request), writer)
    assert writer.groups == []
    assert isinstance(failure.value.exceptions[0], ConnectionError)


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
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_disabled_harbor_verification_keeps_stage_tokens_without_a_score(
    task_inputs, staged, phase, projection_type
):
    config, request = task_inputs
    task = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
    broken = VerifierSpec(
        kind=VerifierKind.SHELL,
        parameters_json=ShellVerifierSpec(argv=("false",), timeout=5).model_dump_json(),
    )
    task = task.model_copy(
        update={
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "metadata": {"harbor": {}},
            "verifier": VerifierSpec(
                kind=VerifierKind.STAGED,
                parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
            )
            if staged
            else broken,
            "stages": (
                TaskStage(name="first", verifier=broken, minimum_rewards={"reward": 1}),
                TaskStage(
                    name="second",
                    verifier=broken,
                    context=ConversationInput(events=(TextMessage(role="user", content="Continue."),)),
                ),
            )
            if staged
            else (),
        }
    )
    request["env_extras"][0]["task_spec"] = task.model_dump_json()
    request["batch_metadata"] = BatchMetadata(0, phase)
    client = ConversationClient(["Done", "Done"] if staged else ["Done"])
    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"verifier_disable": True}}))
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        client,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        command_timeout=5,
        harbor=settings,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert all(grade.status == VerificationStatus.SKIPPED for grade in batch["verification_results"])
    assert batch["rollout_metrics"]["generate/task_rollout/tasks"] == 1
    assert batch["rollout_metrics"]["generate/task_rollout/turns"] == (2 if staged else 1)
    assert batch["rollout_metrics"]["generate/task_rollout/multi_turn_tasks"] == int(staged)
    assert batch["rollout_metrics"]["generate/task_rollout/generated_tokens"] == (4 if staged else 2)
    assert all(grade.score is None for grade in batch["verification_results"])
    if staged and projection_type is StepTaskProjection:
        assert batch["response_ids"] == [[3, 4], [5, 6]]
        assert batch["loss_masks"] == [[1, 1], [1, 1]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2], [-0.1, -0.2]])
    else:
        assert batch["response_ids"] == ([[3, 4, 90, 91, 5, 6]] if staged else [[3, 4]])
        assert batch["loss_masks"] == ([[1, 1, 0, 0, 1, 1]] if staged else [[1, 1]])
        np.testing.assert_allclose(
            batch["rollout_logprobs"], ([[-0.1, -0.2, 0, 0, -0.1, -0.2]] if staged else [[-0.1, -0.2]])
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["startup", "model", "verifier", "attempt", "cancel", "cancel_model"])
async def test_harbor_retries_close_failed_attempts_and_commit_only_the_selected_result(task_inputs, failure_stage):
    config, request = task_inputs
    task = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "metadata": {"harbor": {}},
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(
                    argv=("cat", "/workspace/reward"),
                    timeout=5,
                ).model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["task_spec"] = task.model_dump_json()
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
            machine = await ShellSimMachineFactory().create(spec)
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
        {EnvironmentKind.SHELLSIM: Factory()},
        command_timeout=5,
        harbor=settings,
        retry_wait=wait,
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
    task = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
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
        timeout=5,
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    # A command failure without a reward file is a missing-reward failure.
    # Use stdout grading to exercise the retry limit for verifier execution errors.
    if failure == "exhausted":
        verifier = ShellVerifierSpec(argv=("false",), timeout=5)
    if failure.startswith("grade_timeout"):
        verifier = verifier.model_copy(update={"environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM)})
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "metadata": {"harbor": {}},
            "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
        }
    )
    request["env_extras"][0]["task_spec"] = task.model_dump_json()
    request["batch_metadata"] = BatchMetadata(0, phase)
    attempts = 0
    waits = []

    class Factory:
        async def create(self, spec):
            nonlocal attempts
            attempts += 1
            if failure.startswith("grade_timeout") and attempts == 2:
                raise TimeoutError("Verifier machine startup timed out")
            machine = await ShellSimMachineFactory().create(spec)
            await machine.run(Command(("sh", "-c", f"echo {attempts} > /workspace/attempt")))
            return machine

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
        {EnvironmentKind.SHELLSIM: Factory()},
        command_timeout=5,
        harbor=settings,
        retry_wait=wait,
    )
    batch = await worker.run(request)
    retries = {"exhausted": 2, "retry_missing": 1}.get(failure, 0)
    assert attempts == (2 if failure.startswith("grade_timeout") else retries + 1)
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
        assert batch["loss_masks"] == ([[]] if failure == "grade_timeout_discard" else [[0, 0]])
        assert batch["exclude_from_baseline"] == [True]
    assert batch["response_ids"] == ([[]] if failure == "grade_timeout_discard" else [[3, 4]])
    np.testing.assert_allclose(
        batch["rollout_logprobs"], ([[]] if failure == "grade_timeout_discard" else [[-0.1, -0.2]])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason,expected", [("stop", 1.3), ("length", 0.8)])
async def test_harbor_completion_reward_uses_the_engine_stop_reason(task_inputs, stop_reason, expected):
    config, request = task_inputs
    task = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", "echo '===== 1 passed in 0.1s ====='; echo 1 > /reward.txt"),
        timeout=5,
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
            "metadata": {"harbor": {}},
            "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
        }
    )
    request["env_extras"][0]["task_spec"] = task.model_dump_json()

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
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        command_timeout=5,
        harbor=settings,
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
    base = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
    tasks = []
    for output in ("unrecognized output", "tests/test_task.py::test_answer PASSED"):
        verifier = ShellVerifierSpec(
            argv=("sh", "-c", f"echo '{output}'; echo 1 > /reward.txt"),
            timeout=5,
            reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        )
        tasks.append(
            base.model_copy(
                update={
                    "answer_type": AnswerType.STATE,
                    "environment": EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
                    "metadata": {"harbor": {}},
                    "verifier": VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
                }
            )
        )
    request.update(
        prompts=request["prompts"] * 2,
        env_classes=["taskcompendium"] * 2,
        env_extras=[{"task_spec": task.model_dump_json()} for task in tasks],
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
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        command_timeout=5,
        harbor=settings,
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
    base = TaskSpec.model_validate_json(request["env_extras"][0]["task_spec"])
    tasks = [base.model_copy(update={"id": "harbor", "metadata": {"harbor": {}}})] * 2 + [base]
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
        env_extras=[{"task_spec": task.model_dump_json()} for task in tasks],
        trajectory_ids=[TrajectoryID(str(index), 0) for index in range(3)],
        batch_metadata=BatchMetadata(0, phase),
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        WaitingClient(),
        {},
        command_timeout=5,
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"n_concurrent_trials": 2}})),
        concurrent_tasks=2 if phase == "train" else 3,
        concurrent_harbor_tasks=1,
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
async def test_mixed_nemotron_tasks_run_from_portable_parquet(
    tmp_path, task_inputs, projection_type, alias_file, alias_metadata, identifier, model_failure
):
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "instruction.md").write_text("Repair the terminal task.")
    (source / "task.toml").write_text('[environment]\nworkdir = "/workspace"\nallow_internet = false\n')
    (source / "environment/Dockerfile").write_text("FROM busybox\n")
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
        environment_configs={},
        terminal_bench_data=[str(source)],
        cache_dir=tmp_path / "tasks",
        num_workers=1,
    )
    shutil.rmtree(source)
    input_path.unlink()
    restored = TaskDataset([str(prepared.task_path)], Tokenizer(), 100, num_workers=1)
    prompts, environments, extras, uids = zip(*[restored[index] for index in range(len(restored))], strict=True)

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
        {EnvironmentKind.DOCKER: ImageFactory()},
        command_timeout=5,
        harbor=settings,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "mixed"}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    if model_failure:
        assert batch["exclude_from_baseline"] == [True, False, True]
        assert batch["error_treatments"] == ["mask", "zero", "mask"]
        assert batch["loss_masks"] == [[], [], []]
        expected_rows = [0, 1, 2]
    elif projection_type is WholeTaskProjection:
        assert batch["unshaped_rewards"] == [0.55, 0.75, 0.55]
        assert [sum(reward) for reward in batch["rewards"]] == [1.1, 1.0, 1.1]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1], [1, 1], [1, 1, 0, 0, 1, 1]]
        expected_rows = [0, 1, 2]
    else:
        assert batch["unshaped_rewards"] == [0.1, 1.0, 0.75, 0.1, 1.0]
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
@pytest.mark.parametrize("last_grade", ["0", "invalid", "timeout"])
async def test_staged_tasks_keep_valid_credit_and_candidates_in_the_buffer(task_inputs, projection_type, last_grade):
    config, request = task_inputs
    config.max_turns = 1
    config.error_handling = {
        "enable_error_classification": True,
        "passthrough_exceptions": ["AgentTimeoutError"],
        "preserve_logprobs_on_timeout": False,
    }
    first_context = ConversationInput(events=(TextMessage(role="user", content="Complete the first stage."),))
    task = TaskSpec(
        id="staged",
        context=first_context,
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.STATE,
        environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
        verifier=VerifierSpec(
            kind=VerifierKind.STAGED,
            parameters_json=StageVerifierSpec(strategy=StageRewardStrategy.MEAN).model_dump_json(),
        ),
        stages=(
            TaskStage(
                name="first",
                verifier=VerifierSpec(
                    kind=VerifierKind.SHELL,
                    parameters_json=ShellVerifierSpec(argv=("echo", "1"), timeout=5).model_dump_json(),
                ),
            ),
            TaskStage(
                name="second",
                context=ConversationInput(events=(TextMessage(role="user", content="Complete the second stage."),)),
                verifier=VerifierSpec(
                    kind=VerifierKind.SHELL,
                    parameters_json=ShellVerifierSpec(argv=("echo", last_grade), timeout=5).model_dump_json(),
                ),
            ),
        ),
        source=Source(dataset="fixture", revision="1", row="staged", importer_revision="1"),
    )
    request["prompts"] = [[{"role": "user", "content": "Complete the first stage."}]]
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    projection = (
        WholeTrajectoryProjection(config, Tokenizer())
        if projection_type is WholeTaskProjection
        else StepWiseTrajectoryProjection(config, Tokenizer())
    )

    class StageClient(ConversationClient):
        async def generate(self, request):
            if last_grade == "timeout" and self.requests:
                raise TimeoutError("Model request timed out")
            return await super().generate(request)

    runner = TaskRolloutWorker(
        config,
        projection_type(projection),
        StageClient(["Completed first.", "Completed second."]),
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        command_timeout=5,
    )
    writer = Writer()
    await runner.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "staged"}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    if last_grade == "timeout":
        assert batch["response_ids"] == [[3, 4]]
        assert batch["loss_masks"] == [[1, 1]]
        assert batch["rewards"] == [[0, 1]]
        np.testing.assert_array_equal(batch["student_topk_indices"], [[[3, 99], [4, 99]]])
        assert batch["exclude_from_baseline"] == [False]
        return
    valid = last_grade == "0"
    if projection_type is WholeTaskProjection:
        assert batch["response_ids"] == [[3, 4, 90, 91, 5, 6]]
        assert batch["loss_masks"] == [[1, 1, 0, 0, int(valid), int(valid)]]
        assert batch["rewards"] == ([[0, 0, 0, 0, 0, 0.5]] if valid else [[0, 1, 0, 0, 0, 0]])
        np.testing.assert_array_equal(batch["student_topk_indices"][0][:2], [[3, 99], [4, 99]])
        assert batch["exclude_from_baseline"] == [False]
    else:
        assert batch["response_ids"] == [[3, 4], [5, 6]]
        assert batch["prompt_token_ids"] == [[1, 2], [1, 2, 3, 4, 90, 91]]
        assert batch["loss_masks"] == [[1, 1], [int(valid), int(valid)]]
        assert batch["rewards"] == ([[0, 0], [0, 0.5]] if valid else [[0, 1], [0, 0]])
        assert batch["exclude_from_baseline"] == [False, not valid]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "phase,treatment,preserve,retain,exclude",
    [
        ("model_context", "passthrough", True, True, False),
        ("model_context", "zero", True, True, False),
        ("model_context", "mask", True, True, True),
        ("step", "passthrough", True, True, False),
        ("step", "zero", True, True, False),
        ("step", "passthrough", False, False, True),
        ("model_server", "mask", True, False, True),
        ("model_timeout", "passthrough", True, True, False),
        ("model_timeout", "passthrough", False, False, True),
        ("missing_logprobs", "passthrough", True, False, True),
        ("prepare", "mask", True, False, True),
    ],
)
async def test_interrupted_tasks_keep_verified_turns_and_apply_training_policy(
    task_inputs, monkeypatch, projection_type, phase, treatment, preserve, retain, exclude
):
    closed = []

    class InterruptedEnv(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            self.turn = 0

        def init(self, prompt):
            if phase == "prepare":
                raise TimeoutError("private initialization details")
            return prompt, {}

        def step(self, action):
            self.turn += 1
            if phase == "step" and self.turn == 2:
                raise TimeoutError("private environment details")
            return {
                "observations": [{"role": "user", "content": "Continue"}],
                "reward": 1.0,
                "done": False,
                "metadata": {},
            }

        def close(self):
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

    monkeypatch.setitem(registry, "interrupted", EnvSpec("interrupted", entry_point=InterruptedEnv))
    config, request = task_inputs
    exception_type = {
        "model_context": "ContextLengthExceededError",
        "model_server": "ModelServerError",
        "step": "AgentTimeoutError",
        "prepare": "AgentTimeoutError",
        "model_timeout": "AgentTimeoutError",
        "missing_logprobs": "ContextLengthExceededError",
    }[phase]
    config.error_handling = {
        "enable_error_classification": True,
        f"{treatment}_exceptions": [exception_type],
        "preserve_logprobs_on_timeout": preserve,
    }
    task = gym_task(
        request["prompts"][0],
        "interrupted",
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    request["env_classes"] = ["interrupted"]
    request["sampling_params"] = {"max_tokens": 3}
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config, projection_type(projection), InterruptedClient(["first", "unverified"]), {}, command_timeout=5
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert closed == [True]
    assert batch["error_treatments"] == [treatment]
    assert batch["exception_types"] == [exception_type]
    assert batch["exclude_from_baseline"] == [exclude]
    assert "private" not in str(batch["verification_results"])
    if retain:
        assert batch["response_ids"] == [[3, 4]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
        assert batch["loss_masks"] == ([[0, 0]] if treatment == "mask" else [[1, 1]])
        assert batch["unshaped_rewards"] == [1.0]
        assert batch["rewards"] == [[0.0, 1.0 if treatment == "passthrough" else 0.0]]
        assert batch["evidence_messages"][0][-1]["content"] == "first"
        np.testing.assert_array_equal(batch["student_topk_indices"], [[[3, 99], [4, 99]]])
        assert batch["rollout_routed_experts"][0][:, 0, 0].tolist() == [1, 1]
    else:
        assert batch["response_ids"] == [[]]
        assert batch["loss_masks"] == [[]]
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
    task = gym_task(
        request["prompts"][0],
        "gsm8k_multi_turn",
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InvalidClient(["#### 13", "#### 12"]),
        {},
        command_timeout=5,
    )
    writer = Writer()
    with pytest.raises(ExceptionGroup) as failure:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    assert isinstance(failure.value.exceptions[0], RolloutContractError)
    assert writer.groups == []


@pytest.mark.asyncio
async def test_environment_action_rewrite_cannot_replace_sampled_tokens(task_inputs, monkeypatch):
    closed = []

    class RewritingEnv(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()

        def step(self, action):
            return {
                "observations": [],
                "reward": 1.0,
                "done": True,
                "metadata": {},
                "postprocessed_action": "A different answer",
            }

        def close(self):
            closed.append(True)

    monkeypatch.setitem(registry, "rewrite", EnvSpec("rewrite", entry_point=RewritingEnv))
    config, request = task_inputs
    task = gym_task(
        request["prompts"][0],
        "rewrite",
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {},
        command_timeout=5,
    )
    writer = Writer()
    with pytest.raises(ExceptionGroup) as failure:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    assert isinstance(failure.value.exceptions[0], RolloutContractError)
    assert writer.groups == []
    assert closed == [True]


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
        command_timeout=5,
    )
    batch = await worker.run(request)
    assert batch["response_ids"] == [[3, last_token]]
    assert batch["loss_masks"] == [mask]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
async def test_gym_source_materialization_runs_without_the_original_dataset(tmp_path, task_inputs):
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
    prepared = GymTaskDataset(
        [str(source)],
        Tokenizer(),
        100,
        environment_configs={"gsm8k": {"reward_method": "strict"}},
        cache_dir=tmp_path / "tasks",
        num_workers=1,
    )
    source.unlink()
    restored = TaskDataset([str(prepared.task_path)], Tokenizer(), 100, num_workers=1)
    prompt, env, extras, uid = restored[0]
    config, _ = task_inputs
    client = ConversationClient(["#### 12"])
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), client, {}, command_timeout=5
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
    (source / "task.toml").write_text('[environment]\nworkdir = "/workspace"\nallow_internet = false\n')
    (source / "environment/Dockerfile").write_text("FROM busybox\n")
    (source / "tests/test.sh").write_text(
        '#!/bin/sh\nif [ "$(cat /logs/artifacts/answer)" = "12" ]; then\n'
        "echo 0.75 > /logs/verifier/reward.txt\nelse\necho 0 > /logs/verifier/reward.txt\nfi\n"
    )
    prepared = HarborTaskDataset([str(source)], Tokenizer(), 100, cache_dir=tmp_path / "tasks", num_workers=1)
    assert prepared.uid(0) == "source"
    shutil.rmtree(source)
    restored = TaskDataset([str(prepared.task_path)], Tokenizer(), 100, num_workers=1)
    prompt, env, extras, uid = restored[0]

    class ImageFactory:
        async def create(self, spec):
            assert isinstance(spec.source, DockerfileSource)
            assert spec.source.dockerfile.read_text() == "FROM busybox\n"
            assert not (spec.source.context / "tests").exists()
            return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins()))

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
        {EnvironmentKind.DOCKER: ImageFactory()},
        command_timeout=5,
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
            timeout=10,
            reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        )
        tasks.append(
            TaskSpec(
                id=f"harbor-{index}",
                context=ConversationInput(events=(TextMessage(role="user", content="Complete the task."),)),
                environment_requirements=EnvironmentRequirements(),
                answer_type=AnswerType.STATE,
                environment=EnvironmentSpec(kind=EnvironmentKind.SHELLSIM),
                verifier=VerifierSpec(kind=VerifierKind.SHELL, parameters_json=verifier.model_dump_json()),
                source=Source(dataset="harbor", revision="1", row=str(index), importer_revision="1"),
                metadata={"harbor": {}},
            )
        )
    request.update(
        prompts=[[{"role": "user", "content": "Complete the task."}]] * 2,
        env_classes=["taskcompendium"] * 2,
        env_extras=[{"task_spec": task.model_dump_json()} for task in tasks],
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
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        command_timeout=5,
        harbor=settings,
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


@pytest.mark.asyncio
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
    task_inputs, monkeypatch, phase, grading, failed_peer, failed_judge
):
    comparisons = []

    def judge_response(url, **kwargs):
        metadata = kwargs["json"]["metadata"]
        comparisons.append(metadata)
        if failed_judge:
            raise requests.ConnectionError("Judge unavailable")
        first_better = metadata["response_1"] == "better"
        result = '{"score_1":5,"score_2":1,"ranking":1}' if first_better else '{"score_1":1,"score_2":5,"ranking":6}'
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": result}]}]}
        ).encode()
        return response

    monkeypatch.setattr(requests, "post", judge_response)
    config, request = task_inputs
    count = 3 if failed_peer else 2
    task = gym_task(
        request["prompts"][0],
        environment="nemotron_ultra",
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": "genrm_simple_agent",
                    "record_json": json.dumps({"principle": "Prefer the correct answer."}),
                    "request_json": "{}",
                }
            }
        },
        config={
            "grading": grading,
            "genrm": {
                "num_rollouts_per_prompt": count,
                "genrm_parse_retries": 0,
                "default_score": 0,
                "reasoning_bonus": 0,
                "answer_bonus": 0,
                "group_reasoning_length_penalty_coeff": 0,
                "group_answer_length_penalty_coeff": 0,
                "judge": {"base_url": "http://judge.test", "model": "judge"},
            },
        },
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request.update(
        prompts=request["prompts"] * count,
        env_classes=["nemotron_ultra"] * count,
        env_extras=[{"task_spec": task.model_dump_json()}] * count,
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
        command_timeout=5,
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
        assert batch["loss_masks"][failed_index] == []
        assert batch["exclude_from_baseline"][failed_index] is True
        assert {pair["response_1"] for pair in comparisons} == {"better", "worse"}
        if failed_judge:
            assert all(not any(mask) for mask in batch["loss_masks"])
            assert all(grade.score is None for grade in batch["verification_results"])
            assert batch["rewards"] == [0.0, 0.0, 0.0]
        else:
            for index, rollout in enumerate(rollouts):
                if index == failed_index:
                    assert batch["rewards"][index] == []
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
@pytest.mark.parametrize(
    "environment,extras,responses,rewards",
    [
        ("cat_count", {"extra_info": {"n": 2}}, ["cat cat"], [0.0, 1.0]),
        ("gsm8k", {"reward_spec": {"ground_truth": "12"}}, ["#### 12"], [0.0, 1.0]),
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
async def test_unified_gym_tasks_preserve_grading_and_turn_credit(
    task_inputs, tmp_path, environment, extras, responses, rewards
):
    config, request = task_inputs
    raw_path = str(tmp_path / "source.parquet")
    task_path = str(tmp_path / "tasks.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"prompt": request["prompts"][0], "env_class": environment, **extras}]), raw_path
    )
    write_tasks(
        task_path,
        read_gym_tasks(raw_path, dataset="fixture", revision="1", environment_configs={environment: {}}),
    )
    task = next(read_tasks(task_path))
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    request["env_classes"] = [environment]
    model = ConversationClient(responses)
    runner = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), model, {}, command_timeout=5
    )
    batch = await runner.run(request)
    assert batch["rewards"] == [rewards]
    assert batch["unshaped_rewards"] == [sum(rewards) / len(responses)]
    assert batch["verification_results"][0].score == sum(rewards) / len(responses)
    if environment == "cat_count":
        assert batch["verification_results"][0].passed is True
        assert batch["env_metrics"][0]["exact_n2"] == 1.0
    expected_mask = [1, 1] if len(responses) == 1 else [1, 1, 0, 0, 1, 1]
    assert batch["loss_masks"] == [expected_mask]
    assert batch["rollout_routed_experts"][0][:, 0, 0].tolist() == expected_mask
    assert len(batch["student_topk_indices"][0]) == len(expected_mask)
    assert all("ground_truth" not in str(item["prompts"]) for item in model.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("reward_key", ["reward_spec", "reward_model"])
async def test_aime_rollout_preserves_length_reward_and_phase_metrics(
    task_inputs, delivered_telemetry, enable_thinking, reward_key
):
    config, request = task_inputs
    config.chat_template_kwargs = {"enable_thinking": enable_thinking}
    task = gym_task(
        request["prompts"][0],
        environment="aime",
        extras={reward_key: {"ground_truth": "12"}},
        config={"length_penalty_weight": 1.0, "min_response_length": 0, "evaluation_token_budget": 1},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
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
        command_timeout=5,
        max_env_workers=1,
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
    assert waits == {"model_client_await": 1, "env_await": 3, "env_queue": 3, "env_exec": 3, "env_resume": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "max_model_len,max_input_length,expected_budgets",
    [(5, 100, [3]), (2, 100, []), (None, 4, [10]), (None, 1, [])],
)
async def test_context_limits_preserve_only_completed_gym_turns(
    task_inputs, projection_type, max_model_len, max_input_length, expected_budgets
):
    config, request = task_inputs
    config.max_input_length = max_input_length
    config.engine_init_kwargs = {"max_model_len": max_model_len}
    config.sampling_params.logprobs = 0
    task = gym_task(
        request["prompts"][0],
        "gsm8k_multi_turn",
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
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
    worker = TaskRolloutWorker(config, projection_type(projection), DirectModelClient(engine), {}, command_timeout=5)
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
        assert batch["unshaped_rewards"] == [0.1]
        assert batch["rewards"] == [[0.0, 0.1]]
        assert batch["evidence_messages"][0][-1] == {"role": "assistant", "content": "#### 13"}
    else:
        assert batch["response_ids"] == [[]]
        assert batch["loss_masks"] == [[]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[]])
        assert batch["verification_results"][0].score is None
        assert batch["exclude_from_baseline"] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_native_rewards_and_credit_remain_aligned_across_tool_observations(
    task_inputs, monkeypatch, projection_type
):
    class CreditEnv(BaseTextEnv):
        def __init__(self, env_config, extras):
            super().__init__()
            self.turn = 0

        def step(self, action):
            self.turn += 1
            return {
                "observations": [{"role": "user", "content": "Continue"}],
                "done": self.turn == 2,
                "metadata": {},
                "reward": 0.0,
                "verification": VerificationResult.verified(float(self.turn)),
                "reward_result": RewardResult(
                    unshaped_reward=float(self.turn),
                    optimization_reward=0.5 * self.turn,
                    token_rewards=(0.2 * self.turn, 0.3 * self.turn),
                    token_credit=(-0.1 * self.turn, 0.0),
                    components={"penalty": -0.5 * self.turn},
                ),
            }

    monkeypatch.setitem(registry, "credit", EnvSpec("credit", entry_point=CreditEnv))
    config, request = task_inputs
    task = gym_task(
        request["prompts"][0],
        "credit",
        {},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    runner = TaskRolloutWorker(
        config, projection_type(projection), ConversationClient(["first", "second"]), {}, command_timeout=5
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
async def test_lean_refinement_discards_the_failed_attempt(task_inputs, monkeypatch):
    def compiler_response(*args, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"process_status":"completed","stdout":"","stderr":""}'
        return response

    monkeypatch.setattr(requests, "post", compiler_response)
    config, request = task_inputs
    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "skyrl_gym",
                "agent": "math_formal_lean_refinement_agent",
                "record_json": json.dumps({"header": "", "formal_statement": "theorem equality : 1 = 1 := by sorry"}),
                "request_json": "{}",
            }
        }
    }
    task = gym_task(
        request["prompts"][0],
        "nemotron_ultra",
        extras,
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    client = ConversationClient(["", "by rfl"])
    runner = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), client, {}, command_timeout=5
    )
    batch = await runner.run(request)
    assert batch["response_ids"] == [[5, 6]]
    assert batch["loss_masks"] == [[1, 1]]
    assert batch["rewards"] == [[0.0, 1.0]]
    assert client.requests[1]["chat_continuations"] == [None]


@pytest.mark.asyncio
async def test_step_projection_preserves_served_prompts_grades_and_teacher_routes(task_inputs):
    config, request = task_inputs
    task = gym_task(
        request["prompts"][0],
        environment="gsm8k_multi_turn",
        extras={"reward_spec": {"ground_truth": "12"}},
        config={},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"task_spec": task.model_dump_json(), "teacher_route": "arithmetic"}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    runner = TaskRolloutWorker(
        config,
        StepTaskProjection(StepWiseTrajectoryProjection(config, Tokenizer())),
        ConversationClient(["#### 13", "#### 12"]),
        {},
        command_timeout=5,
    )
    writer = Writer()
    await runner.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["prompt_token_ids"] == [[1, 2], [1, 2, 3, 4, 90, 91]]
    assert batch["response_ids"] == [[3, 4], [5, 6]]
    assert batch["loss_masks"] == [[1, 1], [1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2], [-0.1, -0.2]])
    assert batch["rewards"] == [[0.0, 0.1], [0.0, 1.0]]
    assert batch["unshaped_rewards"] == [0.1, 1.0]
    assert [grade.score for grade in batch["verification_results"]] == [0.1, 1.0]
    assert [item.step for item in batch["trajectory_ids"]] == [0, 1]
    assert batch["is_last_step"] == [False, True]
    assert batch["teacher_route_keys"] == ["arithmetic", "arithmetic"]
    assert [messages[-1]["content"] for messages in batch["evidence_messages"]] == ["#### 13", "#### 12"]
    np.testing.assert_array_equal(batch["student_topk_indices"], [[[3, 99], [4, 99]], [[5, 99], [6, 99]]])
    assert all(routes[:, 0, 0].tolist() == [1, 1] for routes in batch["rollout_routed_experts"])


@pytest.fixture
def python_tool_task(task_inputs):
    config, request = task_inputs
    task = gym_task(
        request["prompts"][0],
        environment="nemotron_ultra",
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
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
    request["env_extras"] = [{"task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    return task, message


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_python_tool_observations_preserve_training_and_release_session(
    python_tool_task, task_inputs, monkeypatch, projection_type
):
    sessions = set()

    def execute(url, **kwargs):
        sessions.add(kwargs["headers"]["X-Session-ID"])
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"process_status":"completed","stdout":"4","stderr":""}'
        return response

    def close(url, **kwargs):
        sessions.remove(kwargs["headers"]["X-Session-ID"])
        response = requests.Response()
        response.status_code = 204
        return response

    monkeypatch.setattr(requests, "post", execute)
    monkeypatch.setattr(requests, "delete", close)
    config, request = task_inputs
    _, message = python_tool_task
    model = ConversationClient(["", "4"], messages=[message, {"role": "assistant", "content": "4"}])
    projection = (
        WholeTrajectoryProjection(config, Tokenizer())
        if projection_type is WholeTaskProjection
        else StepWiseTrajectoryProjection(config, Tokenizer())
    )
    runner = TaskRolloutWorker(config, projection_type(projection), model, {}, command_timeout=5)
    batch = await runner.run(request)
    assert sessions == set()
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
def environment_executor(request):
    if request.param == 0:
        yield None
    else:
        with ThreadPoolExecutor(max_workers=request.param) as executor:
            yield executor


@pytest.mark.asyncio
async def test_cancellation_waits_for_the_tool_before_session_cleanup(
    python_tool_task, monkeypatch, environment_executor
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    events = []

    def execute(url, **kwargs):
        events.append("start")
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        events.append("finish")
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"process_status":"completed","stdout":"4","stderr":""}'
        return response

    def close(url, **kwargs):
        events.append("close")
        response = requests.Response()
        response.status_code = 204
        return response

    monkeypatch.setattr(requests, "post", execute)
    monkeypatch.setattr(requests, "delete", close)
    task, message = python_tool_task
    session = GymTaskSession(task, max_turns=2, executor=environment_executor)
    await session.prepare()
    operation = asyncio.create_task(session.advance(ModelTurn(message, (1, 2), (3, 4), None, "stop")))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        operation.cancel()
        completion_at_cancel = loop.create_future()
        # This callback runs after the operation receives cancellation, while HTTP remains blocked.
        loop.call_soon(lambda: completion_at_cancel.set_result(operation.done()))
        assert not await completion_at_cancel
    finally:
        release.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await operation
        finally:
            await session.close()
    assert events == ["start", "finish", "close"]


@pytest.mark.asyncio
async def test_worker_keeps_blocking_inference_separate_from_rollout_threads(task_inputs):
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
        command_timeout=5,
        concurrent_tasks=1,
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
