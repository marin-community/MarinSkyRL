"""One complete Pi payload supplies all ETL inputs; no request-log joins."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from harbor.models.agent.pi_trajectory import PiCollatedTrajectory
from pydantic import ValidationError
from rolloutengine.contracts import ModelTurn, RolloutFailure, SessionStart
from shellbox.machine import NetworkPolicy
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_gym.task_records import skipped_grade
from skyrl_gym.verification import VerificationStatus

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.config.utils import get_default_config
from skyrl_train.distillation_adapters import build_teacher_scoring_work
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import PiRolloutData, TaskRolloutWorker, _agent_session_factory
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.rollouts.engine_fakes import InferenceClient
from tests.cpu.task_specs import machine_runtime

from skyrl_train.rollouts.pi_scenario_etl import (
    HydratedPiTrial,
    PiScenario,
    TokenSource,
    scenario_rows_from_config,
    scenario_training_rows,
    training_rows_to_batch,
)


@pytest.mark.asyncio
async def test_pi_rejects_offline_sandbox_before_installing_agent(task_inputs):
    config, request = task_inputs
    original = request.rows[0].lowered
    lowered = original.model_copy(
        update={"runtime": original.runtime.model_copy(update={"task_machine": machine_runtime("docker")})}
    )
    assert lowered.runtime.task_machine.network is NetworkPolicy.DENY
    row = replace(request.rows[0], lowered=lowered)
    leased = TaskRequestBatch(
        rows=(row,),
        trajectory_ids=request.trajectory_ids,
        batch_metadata=BatchMetadata(global_step=1, training_phase="train"),
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "teacher_demo_pi_base"},
        shutdown_timeout=1,
    )
    with pytest.raises(ValueError, match="Pi installation needs sandbox internet"):
        await worker.generate(leased)


class StudentTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, tools):
        assert tokenize and tools == [{"type": "function", "function": {"name": "bash"}}]
        # Render one message at a time: a teacher action plus its tool result
        # must be a prefix of the next visited student-tokenizer context.
        tokens = [11, *json.dumps(tools, sort_keys=True).encode()]
        for message in messages:
            if message["role"] == "assistant":
                tokens.extend((12, *json.dumps(message, sort_keys=True).encode(), 13))
            else:
                tokens.extend((14, *json.dumps(message, sort_keys=True).encode(), 15))
        if add_generation_prompt:
            tokens.append(12)
        return tokens


USER = {"role": "user", "content": "Make a file"}
ACTION = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"echo 1"}'}},
    ],
}
OBS = {"tool_call_id": "call-1", "content": "one", "authored_by": "student"}


def sampled(request_id, messages, assistant, *, model="student", response_id=777):
    return {
        "request_id": request_id,
        "model": model,
        "messages": messages,
        "assistant": assistant,
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "chat_template_kwargs": {},
        "prompt_token_ids": [20, 21],
        "response_token_ids": [response_id],
        "logprobs": [-0.25],
    }


def trial(*, dagger=False, teacher_suffix=False):
    first = sampled("student-0", [USER], ACTION)
    label = sampled("label-0", [USER], ACTION, model="teacher", response_id=888)
    visited = [USER, ACTION, {"role": "tool", "tool_call_id": "call-1", "content": "one"}]
    second = sampled(
        "teacher-1" if teacher_suffix else "student-1",
        visited,
        {"role": "assistant", "content": "Done"},
        model="teacher" if teacher_suffix else "student",
        response_id=999,
    )
    label_second = (
        deepcopy(second)
        if teacher_suffix
        else sampled("label-1", visited, {"role": "assistant", "content": "Done"}, model="teacher", response_id=888)
    )
    return deepcopy(
        {
            "schema_version": 1,
            "trial_id": "trial-1",
            "policy_step": 3,
            "turns": [
                {
                    "index": 0,
                    "actor": "student",
                    "executed": first,
                    "executed_assistant": first["assistant"],
                    "observations": [OBS],
                    "teacher_label": label if dagger else None,
                },
                {
                    "index": 1,
                    "actor": "teacher" if teacher_suffix else "student",
                    "executed": second,
                    "executed_assistant": second["assistant"],
                    "observations": [],
                    "teacher_label": label_second if dagger else None,
                },
            ],
            "planned_executors": ["student", "teacher" if teacher_suffix else "student"] if dagger else None,
            "verifier_status": "graded",
            "verifier_reward": 0.0,
            "verifier_passed": False,
        }
    )


@pytest.mark.parametrize(
    "preset,teacher_suffix,expected_count",
    [
        (PiScenario.DAGGER_MIXTURE, False, 2),
        (PiScenario.DAGGER_TAKEOVER, True, 1),
    ],
)
def test_dagger_private_labels_become_student_tokenizer_sft(preset, teacher_suffix, expected_count):
    evidence = HydratedPiTrial.model_validate(trial(dagger=True, teacher_suffix=teacher_suffix))
    rows = scenario_training_rows(
        preset, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert len(rows.rows) == expected_count
    assert all(
        row.behavior_logprobs is None and set(row.token_sources) == {TokenSource.TEACHER} and set(row.loss_mask) == {1}
        for row in rows.rows
    )
    assert rows.rows[0].response_ids != evidence.turns[0].teacher_label.response_token_ids
    assert rows.rows[0].prompt_token_ids != evidence.turns[0].executed.prompt_token_ids
    assert rows.rows[0].verifier_reward == 0.0
    batch = training_rows_to_batch(rows)
    assert batch["rollout_logprobs"] is None
    assert batch["loss_masks"] == [[1] * len(row) for row in batch["response_ids"]]
    assert batch["token_sources"] == [[TokenSource.TEACHER] * len(row) for row in batch["response_ids"]]
    assert batch["verification_results"][0].score == 0.0


def test_teacher_trace_supervises_only_teacher_executed_actions():
    raw = trial()
    raw["turns"] = [raw["turns"][1]]
    raw["turns"][0]["index"] = 0
    raw["turns"][0]["actor"] = "teacher"
    raw["turns"][0]["executed"]["messages"] = [USER]
    raw["turns"][0]["executed"]["model"] = "teacher"
    raw["turns"][0]["executed"]["logprobs"] = None
    evidence = HydratedPiTrial.model_validate(raw)
    rows = scenario_training_rows(
        PiScenario.TEACHER_TRACE, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert len(rows.rows) == 1
    assert rows.rows[0].behavior_logprobs is None
    assert set(rows.rows[0].token_sources) == {TokenSource.TEACHER}
    assert training_rows_to_batch(rows)["rollout_logprobs"] is None


def _teacher_trial():
    raw = trial()
    raw["turns"] = [raw["turns"][1]]
    raw["turns"][0]["index"] = 0
    raw["turns"][0]["actor"] = "teacher"
    raw["turns"][0]["executed"]["messages"] = [USER]
    raw["turns"][0]["executed"]["model"] = "teacher"
    raw["turns"][0]["executed"]["logprobs"] = None
    return raw


def test_student_behavior_without_logprobs_remains_inadmissible():
    raw = trial()
    raw["turns"][0]["executed"]["logprobs"] = None
    with pytest.raises(ValueError, match="student and revision behavior requires exact sampled logprobs"):
        HydratedPiTrial.model_validate(raw)


def test_gpu_shaped_teacher_trace_has_tool_observation_and_no_teacher_logprobs():
    raw = trial()
    for turn in raw["turns"]:
        turn["actor"] = "teacher"
        turn["executed"]["model"] = "teacher"
        turn["executed"]["logprobs"] = None
        for observation in turn["observations"]:
            observation["authored_by"] = "teacher"
    raw["turns"][0]["executed"]["request_id"] = "teacher-0"
    raw["turns"][1]["executed"]["request_id"] = "teacher-1"
    raw["turns"][0]["executed"]["response_token_ids"] = [37]
    raw["turns"][1]["executed"]["response_token_ids"] = [49]
    evidence = HydratedPiTrial.model_validate(raw)
    rows = scenario_training_rows(
        PiScenario.TEACHER_TRACE,
        evidence,
        StudentTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    assert len(rows.rows) == 1
    assert rows.rows[0].behavior_logprobs is None
    assert rows.rows[0].token_sources.count(TokenSource.TEACHER) > 2
    assert TokenSource.CONTEXT in rows.rows[0].token_sources
    assert training_rows_to_batch(rows)["rollout_logprobs"] is None


def test_ungraded_teacher_trace_is_sft_without_a_fabricated_verifier_score():
    raw = _teacher_trial()
    raw.update(verifier_status="skipped", verifier_reward=None, verifier_passed=None)
    rows = scenario_training_rows(
        PiScenario.TEACHER_TRACE,
        HydratedPiTrial.model_validate(raw),
        StudentTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    batch = training_rows_to_batch(rows)
    assert batch["loss_masks"] == [[1] * len(batch["response_ids"][0])]
    assert batch["rollout_logprobs"] is None
    assert batch["verification_results"][0].status is VerificationStatus.SKIPPED
    assert batch["unshaped_reward_available"] == [False]
    with pytest.raises(ValueError, match="Only unfiltered"):
        scenario_training_rows(
            PiScenario.VERIFIER_GRPO,
            HydratedPiTrial.model_validate({**raw, "turns": trial()["turns"]}),
            StudentTokenizer(),
            expected_trial_id="trial-1",
            expected_policy_step=3,
        )
    with pytest.raises(ValueError, match="Only unfiltered"):
        scenario_training_rows(
            PiScenario.TEACHER_TRACE_VERIFIED,
            HydratedPiTrial.model_validate(raw),
            StudentTokenizer(),
            expected_trial_id="trial-1",
            expected_policy_step=3,
        )
    failed_grade = HydratedPiTrial.model_validate(
        {**raw, "verifier_status": "graded", "verifier_reward": 0.0, "verifier_passed": False}
    )
    failed_rows = scenario_training_rows(
        PiScenario.TEACHER_TRACE_VERIFIED,
        failed_grade,
        StudentTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    assert not any(training_rows_to_batch(failed_rows)["loss_masks"][0])


@pytest.mark.asyncio
async def test_verified_teacher_sft_rejects_incorrect_trial_without_silent_training(task_inputs):
    cfg = get_default_config().generator
    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "teacher_demo_pi_verified_base"},
        shutdown_timeout=1,
    )
    raw = _teacher_trial()
    trajectory = PiCollatedTrajectory.model_validate(
        {key: raw[key] for key in ("schema_version", "trial_id", "policy_step", "turns", "planned_executors")}
        | {"critique_feedback": None, "critique": None}
    ).model_dump(mode="json")
    base = PiRolloutData(
        task_id="task-1",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(999,),
        loss_mask=(1,),
        logprobs=(-0.25,),
        grade=GradeResult(Outcome.GRADED, 1.0, passed=True),
        stop_reason="stop",
        pi_trajectory=trajectory,
        session_id="trial-1",
    )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows * 2,
        trajectory_ids=(TrajectoryID("task-1", 0), TrajectoryID("task-2", 0)),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    wrong = replace(base, grade=GradeResult(Outcome.GRADED, 0.0, passed=False))
    batch = await worker.training_batch(request, [base, wrong])
    decisions = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=1), rollout_logprobs_required=False
    ).evaluate_batch(batch, ["task-1", "task-2"])
    assert decisions[0].decision.accepted
    assert decisions[1].decision.rejections == (AdmissionRejection.FULLY_MASKED,)
    assert batch["loss_masks"][0] == [1] * len(batch["response_ids"][0])
    assert batch["loss_masks"][1] == [0] * len(batch["response_ids"][1])
    assert batch["rollout_metrics"]["pi/teacher_rejected_trials"] == 1.0
    unfiltered = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "teacher_demo_pi_base"},
        shutdown_timeout=1,
    )
    single = replace(request, rows=request.rows[:1], trajectory_ids=request.trajectory_ids[:1])
    ungraded = await unfiltered.training_batch(single, [replace(base, grade=skipped_grade("No verifier"))])
    assert ungraded["verification_results"][0].status is VerificationStatus.SKIPPED
    assert ungraded["unshaped_reward_available"] == [False]
    assert any(ungraded["loss_masks"][0])
    with pytest.raises(ValueError, match="Only unfiltered"):
        await worker.training_batch(single, [replace(base, grade=skipped_grade("No verifier"))])
    with pytest.raises(ValueError, match="explicit graded"):
        await worker.training_batch(single, [replace(base, grade=GradeResult(Outcome.GRADED, 1.0))])


def test_grpo_judge_reward_is_distinct_from_verifier_reward():
    raw = trial()
    raw["turns"] = [raw["turns"][1]]
    raw["turns"][0]["index"] = 0
    raw["turns"][0]["executed"]["messages"] = [USER]
    raw["judge"] = {
        "request_id": "judge-1",
        "trial_id": "trial-1",
        "student_request_ids": ["student-1"],
        "model": "critic",
        "principle": "Solve the task",
        "prompt_revision": "rlaif_judge_v1",
        "score": 1,
    }
    evidence = HydratedPiTrial.model_validate(raw)
    judged = scenario_training_rows(
        PiScenario.RLAIF, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert [row.optimization_reward for row in judged.rows] == [1.0]
    assert all(row.verifier_reward == 0.0 for row in judged.rows)
    assert all(
        row.behavior_logprobs == (-0.25,) and set(row.token_sources) == {TokenSource.STUDENT} for row in judged.rows
    )
    batch = training_rows_to_batch(judged)
    assert batch["rewards"] == [[1.0]]
    assert batch["unshaped_rewards"] == [0.0]
    assert batch["rollout_logprobs"][0].tolist() == [-0.25]
    assert batch["verification_results"][0].score == 0.0
    assert batch["rollout_metrics"]["pi/judged_score"] == 1.0
    with pytest.raises(ValueError, match="Verifier GRPO"):
        scenario_training_rows(
            PiScenario.VERIFIER_GRPO, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
        )
    with pytest.raises(ValueError, match="continuous served token prefix"):
        scenario_training_rows(
            PiScenario.RLAIF,
            HydratedPiTrial.model_validate(
                trial() | {"judge": {**raw["judge"], "student_request_ids": ["student-0", "student-1"]}}
            ),
            StudentTokenizer(),
            expected_trial_id="trial-1",
            expected_policy_step=3,
        )


@pytest.mark.parametrize("second_prompt", [[20, 21, 777, 301, 302], [501, 502, 503]])
def test_multiturn_grpo_keeps_exact_sampled_turns_without_fabricating_prefix(second_prompt):
    raw = trial()
    raw["turns"][1]["executed"]["prompt_token_ids"] = second_prompt
    evidence = HydratedPiTrial.model_validate(raw)
    result = scenario_training_rows(
        PiScenario.VERIFIER_GRPO, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert [row.prompt_token_ids for row in result.rows] == [(20, 21), tuple(second_prompt)]
    assert [row.response_ids for row in result.rows] == [(777,), (999,)]
    assert all(row.loss_mask == (1,) and row.token_sources == (TokenSource.STUDENT,) for row in result.rows)
    assert all(row.behavior_logprobs == (-0.25,) for row in result.rows)
    assert result.evidence_messages[-1] == {"role": "assistant", "content": "Done"}
    assert result.evidence_messages[-2] == {"role": "tool", "tool_call_id": "call-1", "content": "one"}
    batch = training_rows_to_batch(result)
    assert batch["is_last_step"] == [False, True]
    assert batch["rollout_logprobs"][0].tolist() == [-0.25]
    assert batch["loss_masks"] == [[1], [1]]


def test_distribution_scores_discontinuous_tool_followup_as_two_exact_served_turns():
    raw = trial()
    raw["turns"][1]["executed"]["prompt_token_ids"] = [501, 502, 503]
    result = scenario_training_rows(
        PiScenario.DISTRIBUTION,
        HydratedPiTrial.model_validate(raw),
        StudentTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    batch = training_rows_to_batch(result)
    batch["trajectory_ids"] = [TrajectoryID("task-1", 0)] * 2
    assert batch["prompt_token_ids"] == [[20, 21], [501, 502, 503]]
    assert batch["response_ids"] == [[777], [999]]
    assert batch["rollout_logprobs"][0].tolist() == [-0.25]
    assert batch["is_last_step"] == [False, True]
    admission = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=1), rollout_logprobs_required=True
    ).evaluate(SimpleNamespace(trajectory_batch=batch))
    assert admission.accepted
    work = build_teacher_scoring_work(
        batch,
        route_ids=("default", "default"),
        teacher_id="teacher-a",
        tokenizer_fingerprint="sha256:fixed-vocabulary",
        plan_version="distribution-r1",
        coefficient=1.0,
        route_weights=(1.0, 1.0),
        evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION,
        top_k=2,
    )
    request = work.request
    assert request.prompt_token_ids[request.prompt_mask].tolist() == [20, 21, 501, 502, 503]
    assert request.response_token_ids[request.response_mask].tolist() == [777, 999]
    assert request.trajectory_ids == ("task-1_0", "task-1_0")
    assert request.top_k == 2


def test_multiturn_grpo_group_counts_physical_trials_not_turn_rows():
    batches = []
    for index, reward in enumerate((1.0, 0.0)):
        raw = trial()
        raw["trial_id"] = f"trial-{index}"
        raw["verifier_reward"] = reward
        raw["verifier_passed"] = bool(reward)
        raw["turns"][1]["executed"]["prompt_token_ids"] = [501 + index, 502, 503]
        rows = scenario_training_rows(
            PiScenario.VERIFIER_GRPO,
            HydratedPiTrial.model_validate(raw),
            StudentTokenizer(),
            expected_trial_id=f"trial-{index}",
            expected_policy_step=3,
        )
        batches.append(training_rows_to_batch(rows))
    group = concatenate_trajectory_batches(batches, tis_lcs_alert_threshold=0.005)
    assert group["is_last_step"] == [False, True, False, True]
    admission = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=True
    ).evaluate(SimpleNamespace(trajectory_batch=group))
    assert admission.accepted
    assert group["rewards"][1][-1] == 1.0
    assert group["rewards"][3][-1] == 0.0


def test_sampled_tool_json_and_executed_pi_replay_stay_distinct():
    raw = trial()
    replay = deepcopy(ACTION)
    replay["tool_calls"][0]["function"]["arguments"] = '{"command": "echo 1"}'
    raw["turns"][0]["executed_assistant"] = replay
    raw["turns"][1]["executed"]["messages"][1] = deepcopy(replay)
    raw["turns"][1]["executed"]["prompt_token_ids"] = [20, 21, 777, 301]
    evidence = HydratedPiTrial.model_validate(raw)
    assert evidence.turns[0].executed.assistant["tool_calls"][0]["function"]["arguments"] == '{"command":"echo 1"}'
    rows = scenario_training_rows(
        PiScenario.VERIFIER_GRPO, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert [row.loss_mask for row in rows.rows] == [(1,), (1,)]
    assert rows.rows[1].prompt_token_ids == (20, 21, 777, 301)
    raw["turns"][0]["executed_assistant"]["tool_calls"][0]["function"]["arguments"] = '{"command":"rm -rf /"}'
    with pytest.raises(ValidationError, match="differs from the sampled model action"):
        HydratedPiTrial.model_validate(raw)


def test_judge_must_score_this_complete_trial():
    raw = trial()
    raw["judge"] = {
        "request_id": "judge-1",
        "trial_id": "other-trial",
        "student_request_ids": ["student-0", "student-1"],
        "model": "critic",
        "principle": "Solve the task",
        "prompt_revision": "rlaif_judge_v1",
        "score": 1,
    }
    with pytest.raises(ValidationError, match="this complete student trajectory"):
        HydratedPiTrial.model_validate(raw)


def test_judge_rejects_boolean_and_float_scores():
    raw = trial()
    raw["judge"] = {
        "request_id": "judge-1",
        "trial_id": "trial-1",
        "student_request_ids": ["student-0", "student-1"],
        "model": "critic",
        "principle": "Solve",
        "prompt_revision": "rlaif_judge_v1",
        "score": True,
    }
    with pytest.raises(ValidationError, match="exact binary integer"):
        HydratedPiTrial.model_validate(raw)
    raw["judge"]["score"] = 1.0
    with pytest.raises(ValidationError, match="exact binary integer"):
        HydratedPiTrial.model_validate(raw)


def add_critique(raw, feedback):
    first = raw["turns"][0]
    raw["critique_feedback"] = feedback
    raw["turns"][1]["executed"]["messages"].append({"role": "user", "content": [{"type": "text", "text": feedback}]})
    raw["critique"] = {
        "trial_id": "trial-1",
        "teacher_model": "critic",
        "student_request_id": "student-0",
        "teacher_request_id": "teacher-critique-1",
        "teacher_request": {
            "model": "critic",
            "stream": False,
            "messages": [
                *first["executed"]["messages"],
                first["executed"]["assistant"],
                {"role": "tool", "tool_call_id": "call-1", "content": "one"},
                {"role": "user", "content": "Private principle: diagnose this attempt"},
            ],
        },
        "teacher_response": {"role": "assistant", "content": feedback.split(": ")[-1]},
        "feedback": feedback,
    }


def test_critique_revision_uses_only_revision_tokens_and_verifier_reward():
    raw = trial()
    raw["turns"][1]["actor"] = "revision"
    add_critique(raw, "Private teacher feedback: fix file")
    raw["verifier_reward"] = 1.0
    evidence = HydratedPiTrial.model_validate(raw)
    rows = scenario_training_rows(
        PiScenario.CRITIQUE_REVISION, evidence, StudentTokenizer(), expected_trial_id="trial-1", expected_policy_step=3
    )
    assert len(rows.rows) == 1
    assert rows.rows[0].response_ids == (999,)
    assert rows.rows[0].behavior_logprobs == (-0.25,)
    assert rows.rows[0].token_sources == (TokenSource.REVISION,)
    assert rows.rows[0].optimization_reward == 1.0
    batch = training_rows_to_batch(rows)
    assert batch["response_ids"] == [[999]]
    assert batch["token_sources"] == [[TokenSource.REVISION]]
    assert batch["evidence_messages"][0][-1] == {"role": "assistant", "content": "Done"}


def test_critique_rejects_revision_tool_execution():
    raw = trial()
    raw["turns"][1]["actor"] = "revision"
    add_critique(raw, "Fix the file")
    raw["turns"][1]["executed"]["assistant"] = ACTION
    raw["turns"][1]["executed_assistant"] = ACTION
    raw["turns"][1]["observations"] = [OBS]
    evidence = HydratedPiTrial.model_validate(raw)
    with pytest.raises(ValueError, match="tool-free"):
        scenario_training_rows(
            PiScenario.CRITIQUE_REVISION,
            evidence,
            StudentTokenizer(),
            expected_trial_id="trial-1",
            expected_policy_step=3,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_label",
        "private_leak",
        "missing_observation",
        "actor_mismatch",
        "wrong_identity",
        "wrong_label_context",
        "wrong_tool_author",
        "wrong_tool_template",
        "missing_verifier_status",
    ],
)
def test_dagger_fails_closed_on_incomplete_or_joined_evidence(damage):
    raw = trial(dagger=True)
    if damage == "missing_label":
        raw["turns"][0]["teacher_label"] = None
    elif damage == "private_leak":
        raw["turns"][1]["executed"]["messages"].insert(-1, raw["turns"][0]["teacher_label"]["assistant"])
    elif damage == "missing_observation":
        raw["turns"][0]["observations"] = []
    elif damage == "actor_mismatch":
        raw["turns"][1]["actor"] = "teacher"
    elif damage == "wrong_identity":
        raw["trial_id"] = "other-trial"
    elif damage == "wrong_label_context":
        raw["turns"][0]["teacher_label"]["messages"] = [{"role": "user", "content": "different"}]
    elif damage == "wrong_tool_author":
        raw["turns"][0]["observations"][0]["authored_by"] = "teacher"
    elif damage == "wrong_tool_template":
        raw["turns"][0]["teacher_label"]["tools"] = []
    elif damage == "missing_verifier_status":
        raw["verifier_status"] = "unavailable"
    if damage in ("missing_label", "wrong_identity"):
        evidence = HydratedPiTrial.model_validate(raw)
        with pytest.raises(ValueError):
            scenario_training_rows(
                PiScenario.DAGGER_MIXTURE,
                evidence,
                StudentTokenizer(),
                expected_trial_id="trial-1",
                expected_policy_step=3,
            )
    else:
        with pytest.raises(ValidationError):
            HydratedPiTrial.model_validate(raw)


def test_single_preset_selects_etl_and_rejects_conflicting_methods():
    raw = trial()
    raw["turns"] = [raw["turns"][1]]
    raw["turns"][0]["index"] = 0
    raw["turns"][0]["executed"]["messages"] = [USER]
    rows = scenario_rows_from_config(
        {"preset": "verifier_grpo_base"},
        raw,
        StudentTokenizer(),
        expected_trial_id="trial-1",
        expected_policy_step=3,
    )
    assert rows.scenario is PiScenario.VERIFIER_GRPO
    assert rows.rows[0].optimization_reward == 0.0
    assert rows.rows[0].behavior_logprobs == (-0.25,)
    with pytest.raises(ValueError, match="student-only"):
        scenario_rows_from_config(
            {"preset": "verifier_grpo_base"},
            trial(dagger=True),
            StudentTokenizer(),
            expected_trial_id="trial-1",
            expected_policy_step=3,
        )


def test_worker_binds_preset_to_one_hydrated_trial_without_lookups():
    cfg = get_default_config().generator
    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "dagger_pi_turn_mixture_base"},
        shutdown_timeout=1,
    )
    batch = worker.etl_pi_trial(trial(dagger=True), trial_id="trial-1", policy_step=3)
    assert len(batch["response_ids"]) == 2
    assert batch["rollout_logprobs"] is None
    with pytest.raises(ValueError, match="leased identity"):
        worker.etl_pi_trial(trial(dagger=True), trial_id="other", policy_step=3)
    with pytest.raises(ValueError, match="scenario.preset"):
        TaskRolloutWorker(
            cfg,
            WholeTaskProjection(cfg, StudentTokenizer()),
            InferenceClient(),
            {},
            pi_scenario={"preset": "dagger_pi_turn_mixture_base", "plugin_path": "x.js"},
            shutdown_timeout=1,
        )


@pytest.mark.asyncio
async def test_installed_session_returns_one_collated_payload_with_its_model_turn():
    hydrated = {"schema_version": 1, "turns": ["one"]}
    holder = [None]

    class Runtime:
        async def run(self, machine, complete, *, lowered, session_id, policy_step):
            assert session_id == "trial-1" and policy_step == 3
            await complete(SessionStart(({"role": "user", "content": "Solve"},), {}))
            return hydrated

    async def grader(messages, *, stop_reason):
        return GradeResult(Outcome.GRADED, 1.0, passed=True)

    context = SimpleNamespace(machine=None, lowered=None, require_task_grader=lambda: grader)
    session = _agent_session_factory(Runtime(), "trial-1", 3, holder, context)
    assert (await session.prepare()).messages == ({"role": "user", "content": "Solve"},)
    transition = await session.advance(
        ModelTurn(
            {"role": "assistant", "content": "Done"},
            (1,),
            (2,),
            (-0.2,),
            "stop",
        )
    )
    assert transition.done
    assert holder[0] is hydrated
    await session.close()


@pytest.mark.asyncio
async def test_worker_training_batch_transforms_graded_pi_payload_without_logs(task_inputs):
    cfg = get_default_config().generator
    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "dagger_pi_turn_mixture_base"},
        shutdown_timeout=1,
    )
    raw = trial(dagger=True)
    trajectory = {key: raw[key] for key in ("schema_version", "trial_id", "policy_step", "turns", "planned_executors")}
    trajectory.update(critique_feedback=None, critique=None)
    # Validate through the *installed pinned Harbor package*, not a source-tree override.
    trajectory = PiCollatedTrajectory.model_validate(trajectory).model_dump(mode="json")
    graded = PiRolloutData(
        task_id="task-1",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(999,),
        loss_mask=(1,),
        logprobs=(-0.25,),
        grade=GradeResult(Outcome.GRADED, 0.0, passed=False),
        stop_reason="stop",
        pi_trajectory=trajectory,
        session_id="trial-1",
    )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows,
        trajectory_ids=(TrajectoryID("task-1", 0),),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    batch = await worker.training_batch(request, [graded])
    assert batch["rollout_logprobs"] is None
    assert len(batch["response_ids"]) == 2
    assert batch["trajectory_ids"] == [TrajectoryID("task-1", 0)] * 2
    assert [result.score for result in batch["verification_results"]] == [0.0, 0.0]
    second = replace(graded, session_id="trial-2", pi_trajectory={**trajectory, "trial_id": "trial-2"})
    grouped = replace(
        request,
        rows=request.rows * 2,
        trajectory_ids=(TrajectoryID("task-1", 0), TrajectoryID("task-2", 0)),
    )
    combined = await worker.training_batch(grouped, [graded, second])
    assert combined["rollout_logprobs"] is None
    assert len(combined["response_ids"]) == 4
    assert combined["trajectory_ids"] == [TrajectoryID("task-1", 0)] * 2 + [TrajectoryID("task-2", 0)] * 2
    assert all(source == TokenSource.TEACHER for row in combined["token_sources"] for source in row)
    with pytest.raises(ValueError, match="collated trial result"):
        await worker.training_batch(request, [replace(graded, failure=RolloutFailure("AgentTimeoutError"))])
    with pytest.raises(ValueError, match="leased identity"):
        await worker.training_batch(request, [replace(graded, pi_trajectory={**trajectory, "trial_id": "forged"})])
    with pytest.raises(ValueError, match="leased identity"):
        await worker.training_batch(request, [replace(graded, pi_trajectory={**trajectory, "policy_step": 99})])


@pytest.mark.asyncio
async def test_verifier_grpo_rejects_unleased_per_turn_rows(task_inputs):
    cfg = get_default_config().generator
    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "verifier_grpo_base"},
        shutdown_timeout=1,
    )
    raw = trial()
    trajectory = {key: raw[key] for key in ("schema_version", "trial_id", "policy_step", "turns", "planned_executors")}
    trajectory.update(critique_feedback=None, critique=None)
    graded = PiRolloutData(
        task_id="task-1",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(999,),
        loss_mask=(1,),
        logprobs=(-0.25,),
        grade=GradeResult(Outcome.GRADED, 0.0, passed=False),
        stop_reason="stop",
        pi_trajectory=PiCollatedTrajectory.model_validate(trajectory).model_dump(mode="json"),
        session_id="trial-1",
    )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows,
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    with pytest.raises(ValueError, match="physical trial IDs"):
        await worker.training_batch(request, [graded])


@pytest.mark.asyncio
async def test_worker_consumes_installed_harbor_critique_without_supervising_attempt(task_inputs):
    cfg = get_default_config().generator
    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        pi_scenario={"preset": "critique_revision_pi_base"},
        shutdown_timeout=1,
    )
    raw = trial()
    raw["turns"][1]["actor"] = "revision"
    add_critique(raw, "Private teacher feedback: fix file")
    trajectory = PiCollatedTrajectory.model_validate(
        {
            key: raw[key]
            for key in ("schema_version", "trial_id", "policy_step", "turns", "critique_feedback", "critique")
        }
        | {"planned_executors": None}
    ).model_dump(mode="json")
    graded = PiRolloutData(
        task_id="task-1",
        messages=(),
        prompt_token_ids=(1,),
        response_token_ids=(999,),
        loss_mask=(1,),
        logprobs=(-0.25,),
        grade=GradeResult(Outcome.GRADED, 1.0, passed=True),
        stop_reason="stop",
        pi_trajectory=trajectory,
        session_id="trial-1",
    )
    request = TaskRequestBatch(
        rows=task_inputs[1].rows,
        trajectory_ids=(TrajectoryID("task-1", 0),),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    batch = await worker.training_batch(request, [graded])
    assert batch["response_ids"] == [[999]]
    assert batch["loss_masks"] == [[1]]
    assert batch["token_sources"] == [[TokenSource.REVISION]]
    assert batch["rollout_logprobs"][0].tolist() == [-0.25]
    assert batch["rewards"] == [[1.0]]
    assert "Private principle: diagnose" not in str(batch["evidence_messages"])


@pytest.mark.asyncio
async def test_pi_eval_rejects_teacher_capability_before_sampling(task_inputs):
    cfg, request = task_inputs
    request = replace(request, batch_metadata=BatchMetadata(global_step=3, training_phase="eval"))

    class TeacherRuntime:
        student_only = False

    worker = TaskRolloutWorker(
        cfg,
        WholeTaskProjection(cfg, StudentTokenizer()),
        InferenceClient(),
        {},
        installed_agents=TeacherRuntime(),
        pi_scenario={"preset": "dagger_pi_turn_mixture_base"},
        shutdown_timeout=1,
    )
    with pytest.raises(ValueError, match="student-only literal agent before sampling"):
        await worker.generate(request)


def test_unknown_scenario_or_extra_behavior_controls_are_rejected():
    assert PiScenario.from_config({"preset": "dagger_pi_turn_mixture_base"}) is PiScenario.DAGGER_MIXTURE
    with pytest.raises(ValueError):
        PiScenario.from_config({"preset": "dagger_pi_turn_mixture_base", "plugin_path": "x.js"})
    with pytest.raises(ValueError):
        PiScenario.from_config({"preset": "unreviewed"})
