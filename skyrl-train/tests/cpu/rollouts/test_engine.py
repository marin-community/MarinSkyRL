"""Task Parquet through canonical execution and the leased buffer writer."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading

import numpy as np
import pytest
from harbor_config.errors import ErrorCategory, error_category
from jinja2 import Environment, StrictUndefined
from skyrl_gym.task_records import fold_grades, grade_result
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_gym.verification import VerificationResult, VerificationStatus
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import Source
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_gym.source_task import source_task
from tests.cpu.task_specs import lowered_task
from rolloutengine.contracts import RolloutContractError, SessionStart, Transition
from skyrl_train.trajectory_runners.types import TokenProvenance, TrajectoryID
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from tests.cpu.rollouts.engine_fakes import ConversationClient, InferenceClient, Tokenizer, Writer


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
