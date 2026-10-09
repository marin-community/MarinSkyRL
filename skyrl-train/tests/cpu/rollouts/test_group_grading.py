from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import numpy as np
import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import normalized_verifier_score
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    NoGrader,
    PlainText,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.group_grader import GenRMGroupGraderParameters, GroupGraderSpec, task_group_grader
from skyrl_train.rollouts.genrm_grading import grade_genrm_rollouts
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_gym.source_task import source_task
from rolloutengine.spec import LoweredTaskSpec
from tests.cpu.task_specs import lowered_task
from rolloutengine.contracts import ModelTurn, RolloutData, RolloutStep, Transition
from skyrl_train.trajectory_runners.types import BatchMetadata, TrajectoryID
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from tests.cpu.rollouts.engine_fakes import ConversationClient, FixtureImageFactory, InferenceClient, Tokenizer, Writer


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
        verifier = ScriptGrader(
            environment=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
            answer_path=None,
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
                answer_type=AnswerType.WORKSPACE_STATE,
                answer_format=PlainText(),
                grader=verifier,
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
            "grader": NoGrader(reason="Group grading supplies the final score"),
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
