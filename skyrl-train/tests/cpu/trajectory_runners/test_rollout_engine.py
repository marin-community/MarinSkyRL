"""Native task execution through the existing SkyRL training boundary."""

import json
from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from omegaconf import OmegaConf
from rolloutengine.contracts import GenerationLimitReached, RolloutContractError
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, NetworkPolicy, ShellSimBuiltins
from taskcompendium.grader import grader_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.shell_verifier import ArtifactKind, ShellVerifierSpec, VerifierArtifact
from taskcompendium.submission import conversation_messages
from verifyit.spec import NumericSpec

from skyrl_gym.verification import VerificationStatus
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.dataset import PromptDataset
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutTask,
)
from skyrl_train.dataset.tasks import LOWERED_TASK_COLUMN, TASKCOMPENDIUM_ENVIRONMENT
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.projections import WholeTrajectoryProjection
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from skyrl_train.trajectory_runners.types import TokenProvenance, TrajectoryID

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
pytestmark = pytest.mark.asyncio


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, _messages, **_kwargs):
        return [10, 11]


@dataclass
class ReplayClient:
    turns: list
    provenance: TokenProvenance = TokenProvenance.ENGINE
    requests: list = field(default_factory=list)

    async def generate(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        message = self.turns[index]
        if isinstance(message, Exception):
            raise message
        continuation = request["chat_continuations"][0]
        prompt = [10, 11] if continuation is None else [*continuation["served_prefix_token_ids"], 90, 91]
        tokens = [20 + index]
        return {
            "prompt_ids": [prompt],
            "response_ids": [tokens],
            "response_logprobs": [[-0.5]],
            "responses": [message.get("content", "")],
            "assistant_messages": [message],
            "stop_reasons": ["stop"],
            "token_provenance": self.provenance,
            "student_topk_indices": [[[tokens[0], 50]]],
            "behavior_topk_logprobs": [[[-0.5, -2.0]]],
            "routed_experts": [np.asarray([[[index + 1, 2]]], dtype=np.uint8)],
        }


@dataclass
class ImageFactory:
    machines: list = field(default_factory=list)
    failure: Exception | None = None
    failure_after: int = 0

    async def create(self, spec):
        if self.failure is not None and len(self.machines) == self.failure_after:
            raise self.failure
        machine = await ShellSimMachineFactory().create(
            replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
        )
        self.machines.append(machine)
        return machine


async def assert_machines_closed(factory):
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


def answer_task():
    return TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=grader_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).verifier,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def file_task(script=b'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi'):
    return answer_task().model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "environment_requirements": EnvironmentRequirements(capabilities=("shell", "filesystem")),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image=FIXTURE_IMAGE),
                parameters_json=ShellVerifierSpec(
                    argv=("sh", "/tests/grade.sh"),
                    artifacts=(
                        VerifierArtifact(
                            source="/workspace/answer", target="/workspace/answer", kind=ArtifactKind.FILE
                        ),
                    ),
                ).model_dump_json(),
            ),
            "resources": ResourceGroups(verifier=(inline_resource("grade.sh", script),)),
        }
    )


def lowered(task):
    machine = MachineRuntimeSpec(
        backend="fixture",
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(
            task_machine=machine if task.answer_type == AnswerType.FILE else None,
            verifier_machine=machine if task.verifier.kind == "shell" else None,
        ),
        session=TaskSessionSpec(
            task_session="shellbox",
            max_turns=3,
            model_turn_timeout=None,
            command_timeout=1,
            tool_turn_timeout=5,
            total_turn_timeout=None,
            attempt_timeout=10,
            verifier_timeout=5,
            cleanup_timeout=5,
        ),
    )


def runner(client, factories=None, *, error_handling=None):
    config = OmegaConf.merge(
        get_default_config().generator,
        {
            "backend": "vllm",
            "max_input_length": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {"logprobs": True, "max_generate_length": 32},
            "error_handling": error_handling or {},
        },
    )
    return TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), client, factories or {}
    )


def request(record):
    return {
        "prompts": [conversation_messages(record.task.context)],
        "env_classes": [TASKCOMPENDIUM_ENVIRONMENT],
        "env_extras": [
            {LOWERED_TASK_COLUMN: record.model_dump_json(), "data_source": "fixture", "teacher_route": "math"}
        ],
        "sampling_params": {"logprobs": True},
        "trajectory_ids": [TrajectoryID("task", 0)],
    }


def shell_turn(command):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "write",
                "type": "function",
                "function": {"name": "shell", "arguments": json.dumps({"command": command})},
            }
        ],
    }


@pytest.mark.parametrize("answer,reward", [("12", 1.0), ("13", 0.0), ("not a number", 0.0)])
async def test_native_answer_returns_trainable_grade_and_exact_evidence(answer, reward):
    record = lowered(answer_task())
    client = ReplayClient([{"role": "assistant", "content": answer}])
    output = await runner(client).run(request(record))

    assert output["rewards"] == [reward]
    assert output["verification_results"][0].status == VerificationStatus.VERIFIED
    assert output["prompt_token_ids"] == [[10, 11]]
    assert output["response_ids"] == [[20]]
    assert output["loss_masks"] == [[1]]
    assert output["exclude_from_baseline"] == [False]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.5])
    np.testing.assert_array_equal(output["student_topk_indices"][0], [[20, 50]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]]])
    assert output["trajectory_ids"] == [TrajectoryID("task", 0)]
    assert output["teacher_route_keys"] == ["math"]
    assert output["data_sources"] == ["fixture"]
    assert "12" not in json.dumps(client.requests[0]["prompts"])


async def test_native_tool_turn_keeps_private_grading_and_masks_observations():
    client = ReplayClient(
        [
            shell_turn("test ! -f /tests/grade.sh && echo 12 > /workspace/answer"),
            {"role": "assistant", "content": "Done."},
        ]
    )
    factory = ImageFactory()
    output = await runner(client, {"fixture": factory}).run(request(lowered(file_task())))

    assert output["rewards"] == [1.0]
    assert output["response_ids"] == [[20, 90, 91, 21]]
    assert output["loss_masks"] == [[1, 0, 0, 1]]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.5, 0, 0, -0.5])
    np.testing.assert_array_equal(output["student_topk_indices"][0][[0, 3]], [[20, 50], [21, 50]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]], [[0, 0]], [[0, 0]], [[2, 2]]])
    assert client.requests[1]["chat_continuations"][0]["served_prefix_token_ids"] == [10, 11, 20]
    observation = client.requests[1]["prompts"][0][-1]
    assert observation["role"] == "tool"
    assert json.loads(observation["content"])["exit_code"] == 0
    assert len(factory.machines) == 2
    await assert_machines_closed(factory)


async def test_native_verifier_failure_is_masked_instead_of_a_wrong_answer():
    factory = ImageFactory()
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    output = await runner(client, {"fixture": factory}).run(request(lowered(file_task(b"exit 7"))))

    assert output["verification_results"][0].status == VerificationStatus.ERROR
    assert output["verification_results"][0].score is None
    assert output["response_ids"] == [[20, 90, 91, 21]]
    assert output["loss_masks"] == [[0, 0, 0, 0]]
    assert output["exclude_from_baseline"] == [True]
    assert output["unshaped_reward_available"] == [False]
    await assert_machines_closed(factory)


@pytest.mark.parametrize(
    "error,exception_type,excluded",
    [
        (ConnectionError("serving unavailable"), "ConnectionError", True),
        (ModelServerError("context_overflow", "request-1", 400), "ContextLengthExceededError", False),
        (TimeoutError("model deadline"), "AgentTimeoutError", False),
    ],
)
async def test_native_model_failure_without_tokens_returns_masked_padding(error, exception_type, excluded):
    output = await runner(
        ReplayClient([error]),
        error_handling={"enable_error_classification": True, "default_error_treatment": "mask"},
    ).run(request(lowered(answer_task())))
    assert output["verification_results"][0].status == VerificationStatus.ERROR
    assert output["verification_results"][0].score is None
    assert len(output["prompt_token_ids"][0]) > 0
    assert len(output["response_ids"][0]) == 1
    assert output["loss_masks"] == [[0]]
    assert output["rewards"] == [0.0]
    assert output["exclude_from_baseline"] == [excluded]
    assert output["exception_types"] == [exception_type]
    np.testing.assert_array_equal(output["rollout_logprobs"][0], [0.0])


@pytest.mark.parametrize(
    "error,exception_type",
    [
        (ModelServerError("context_overflow", "request-1", 400), "ContextLengthExceededError"),
        (TimeoutError("model deadline"), "AgentTimeoutError"),
    ],
)
@pytest.mark.parametrize(
    "policy,reward,mask,excluded,treatment",
    [
        ("default", 0.0, 1, False, "zero"),
        ("disabled", 0.0, 0, True, "mask"),
        ("mask", 0.0, 0, True, "mask"),
        ("passthrough", 1.0, 1, False, "passthrough"),
    ],
)
async def test_native_partial_model_failure_obeys_training_policy(
    error, exception_type, policy, reward, mask, excluded, treatment
):
    handling = {"enable_error_classification": policy != "disabled", "default_error_treatment": "mask"}
    if policy in {"mask", "passthrough"}:
        handling[f"{policy}_exceptions"] = [exception_type]
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), error])
    output = await runner(client, {"fixture": ImageFactory()}, error_handling=handling).run(
        request(lowered(file_task()))
    )

    assert output["rewards"] == [reward]
    assert output["verification_results"][0].score == 1.0
    assert output["unshaped_rewards"] == [1.0]
    assert output["response_ids"] == [[20]]
    assert output["loss_masks"] == [[mask]]
    assert output["exclude_from_baseline"] == [excluded]
    assert output["exception_types"] == [exception_type]
    assert output["error_treatments"] == [treatment]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.5])


@pytest.mark.parametrize("policy,reward", [("zero", 0.0), ("passthrough", 1.0)])
async def test_native_model_timeout_without_recovery_masks_completed_turns(policy, reward):
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), TimeoutError("model deadline")])
    output = await runner(
        client,
        {"fixture": ImageFactory()},
        error_handling={
            "enable_error_classification": True,
            "preserve_logprobs_on_timeout": False,
            f"{policy}_exceptions": ["AgentTimeoutError"],
        },
    ).run(request(lowered(file_task())))

    assert output["response_ids"] == [[20]]
    assert output["loss_masks"] == [[0]]
    assert output["rewards"] == [reward]
    assert output["verification_results"][0].score == 1.0
    assert output["unshaped_rewards"] == [1.0]
    assert output["exclude_from_baseline"] == [False]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.5])


@pytest.mark.parametrize(
    "failure_after,exception_type",
    [(0, "EnvironmentStartTimeoutError"), (1, "VerifierTimeoutError")],
)
async def test_native_machine_timeout_is_masked_by_execution_phase(failure_after, exception_type):
    factory = ImageFactory(failure=TimeoutError("machine startup deadline"), failure_after=failure_after)
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), {"role": "assistant", "content": "Done."}])
    output = await runner(
        client,
        {"fixture": factory},
        error_handling={"enable_error_classification": True, "default_error_treatment": "mask"},
    ).run(request(lowered(file_task())))

    assert output["rewards"] == [0.0]
    assert output["verification_results"][0].score is None
    assert output["response_ids"][0]
    assert not any(output["loss_masks"][0])
    assert output["exclude_from_baseline"] == [True]
    assert output["exception_types"] == [exception_type]
    assert output["error_treatments"] == ["mask"]
    await assert_machines_closed(factory)


async def test_native_serving_failure_after_work_does_not_train_available_grade():
    error = ModelServerError("service_unavailable", "request-2", 503)
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), error])
    output = await runner(
        client,
        {"fixture": ImageFactory()},
        error_handling={"enable_error_classification": True, "default_error_treatment": "mask"},
    ).run(request(lowered(file_task())))

    assert output["verification_results"][0].score == 1.0
    assert output["unshaped_rewards"] == [1.0]
    assert output["rewards"] == [0.0]
    assert output["loss_masks"] == [[0]]
    assert output["exclude_from_baseline"] == [True]
    assert output["server_errors"] == [
        {"category": "service_unavailable", "request_id": "request-2", "status_code": 503}
    ]


async def test_native_passthrough_without_verifier_score_is_masked():
    factory = ImageFactory(failure=TimeoutError("verifier unavailable"), failure_after=1)
    client = ReplayClient([shell_turn("echo 12 > /workspace/answer"), TimeoutError("model deadline")])
    output = await runner(
        client,
        {"fixture": factory},
        error_handling={"enable_error_classification": True, "passthrough_exceptions": ["VerifierTimeoutError"]},
    ).run(request(lowered(file_task())))

    assert output["verification_results"][0].score is None
    assert output["response_ids"] == [[20]]
    assert output["loss_masks"] == [[0]]
    assert output["rewards"] == [0.0]
    assert output["exclude_from_baseline"] == [True]
    assert output["error_treatments"] == ["passthrough"]


async def test_native_empty_failure_keeps_mixed_batch_evidence_aligned():
    batch = request(lowered(answer_task()))
    for key in ("prompts", "env_classes", "env_extras", "trajectory_ids"):
        batch[key] *= 2
    batch["trajectory_ids"] = [TrajectoryID("task", index) for index in range(2)]
    client = ReplayClient([{"role": "assistant", "content": "12"}, ConnectionError("serving unavailable")])
    output = await runner(client).run(batch)

    assert output["rewards"] == [1.0, 0.0]
    assert output["loss_masks"] == [[1], [0]]
    assert output["exclude_from_baseline"] == [False, True]
    assert all(len(tokens) == 1 for tokens in output["response_ids"])
    np.testing.assert_allclose(output["rollout_logprobs"], [[-0.5], [0.0]])
    np.testing.assert_array_equal(output["student_topk_indices"][0], [[20, 50]])
    assert np.all(output["student_topk_indices"][1] < 0)
    np.testing.assert_array_equal(output["behavior_topk_logprobs"][1], [[0.0, 0.0]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][1], [[[0, 0]]])


@pytest.mark.parametrize(
    "scenario,status", [("skipped", VerificationStatus.SKIPPED), ("generation_limit", VerificationStatus.UNAVAILABLE)]
)
async def test_native_no_verdict_is_classified_and_masked(scenario, status):
    task = answer_task()
    turn = {"role": "assistant", "content": "12"}
    if scenario == "skipped":
        task = task.model_copy(
            update={"verifier": VerifierSpec(kind="skipped", parameters_json='{"reason":"fixture"}')}
        )
    else:
        turn = GenerationLimitReached((10, 11))
    output = await runner(ReplayClient([turn])).run(request(lowered(task)))

    assert output["verification_results"][0].status == status
    assert output["verification_results"][0].score is None
    assert not any(output["loss_masks"][0])
    assert output["exclude_from_baseline"] == [True]


async def test_native_reconstructed_tokens_fail_the_transport_contract():
    client = ReplayClient([{"role": "assistant", "content": "12"}], provenance=TokenProvenance.RECONSTRUCTED)
    with pytest.raises(ExceptionGroup) as failure:
        await runner(client).run(request(lowered(answer_task())))
    assert isinstance(failure.value.exceptions[0], RolloutContractError)


async def test_native_dataset_group_round_trips_through_leased_buffer(tmp_path):
    record = lowered(answer_task())
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {"prompt": conversation_messages(record.task.context), LOWERED_TASK_COLUMN: record.model_dump_json()}
        )
    )
    dataset = PromptDataset(str(path), Tokenizer(), max_prompt_length=128, num_workers=1)
    prompt = dataset.collate_fn([dataset[0]])[0]
    batch = {
        "prompts": [prompt["prompt"]] * 2,
        "env_classes": [prompt["env_class"]] * 2,
        "env_extras": [prompt["env_extras"]] * 2,
        "sampling_params": {"logprobs": True},
        "trajectory_ids": [TrajectoryID(prompt["uid"], index) for index in range(2)],
    }
    buffer = RolloutBuffer(RolloutBufferConfig(1, 1, 0, BatchPolicy.FULL_BATCH, None, None))
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=True
        ),
        GroupSelectionPolicy(None),
    )

    class Writer:
        async def write_rollout(self, lease, group):
            await buffer.commit(lease.lease_id, group.prompt, policy.verdict(group), PayloadReference(group))

    await buffer.publish(1)
    lease = await buffer.acquire_lease()
    client = ReplayClient([{"role": "assistant", "content": answer} for answer in ("12", "13")])
    token_count = await runner(client).run_task(RolloutTask(lease, prompt, batch), Writer())
    admission = await buffer.admit(5)
    groups = buffer.payload_refs(admission.batch_id, [group.index for group in admission.admitted])

    assert token_count == 2
    assert admission.selection is not None
    assert len(groups) == 1
    assert groups[0].trajectory_batch["rewards"] == [1.0, 0.0]
    assert groups[0].trajectory_batch["loss_masks"] == [[1], [1]]
    assert groups[0].policy_step == 1
