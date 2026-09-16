"""Exact student-selected evidence at the teacher-to-learner boundary."""

from dataclasses import replace

import pytest
import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    StudentSelectedTeacherEvidence,
    StudentTopKPolicySurrogateInput,
    TeacherScoreRequest,
    prepare_student_topk_policy_surrogate,
    validate_distillation_attachment,
    validate_teacher_evidence,
    validate_teacher_score_request,
)
from skyrl_train.distillation_adapters import (
    RoutedTeacherScoringPartition,
    RoutedTeacherScoringWork,
    ScoredDistillationBatch,
    TeacherEvidenceCoordinator,
    TeacherScoringWork,
    build_teacher_scoring_work,
)
from skyrl_train.inference_engines.vllm_teacher_oracle import teacher_evidence_from_prompt_logprobs
from skyrl_train.teacher_oracle import TeacherCapabilities, TeacherOracleOwner
from skyrl_train.teacher_routing import TeacherRoute
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.types import TrajectoryID


def _request(trajectory_id: str = "sample-0", teacher_id: str = "math") -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=(trajectory_id,),
        route_ids=(teacher_id,),
        teacher_id=teacher_id,
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routing-1",
        prompt_token_ids=torch.tensor([[11, 12]]),
        prompt_mask=torch.tensor([[True, True]]),
        response_token_ids=torch.tensor([[21, 22]]),
        response_mask=torch.tensor([[True, True]]),
        evidence=TeacherEvidenceKind.STUDENT_SELECTED_TOPK,
        top_k=2,
        student_topk_indices=torch.tensor([[[21, 23], [22, 24]]]),
        behavior_topk_logprobs=torch.log(torch.tensor([[[0.6, 0.3], [0.5, 0.4]]])),
        student_selected_mask=torch.tensor([[True, True]]),
    )


def _evidence(request: TeacherScoreRequest, scores: torch.Tensor) -> StudentSelectedTeacherEvidence:
    return StudentSelectedTeacherEvidence(
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        teacher_id=request.teacher_id,
        teacher_revision=f"{request.teacher_id}-revision",
        plan_version=request.plan_version,
        valid_mask=request.student_selected_mask,
        student_topk_indices=request.student_topk_indices,
        teacher_on_student_logprobs=scores,
    )


def test_student_selected_request_and_teacher_scores_require_exact_coordinates():
    request = _request()
    validate_teacher_score_request(request)
    evidence = _evidence(request, torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]]])))
    validate_teacher_evidence(request, evidence)

    with pytest.raises(ValueError, match="unique"):
        validate_teacher_score_request(replace(request, student_topk_indices=torch.tensor([[[21, 21], [22, 24]]])))
    with pytest.raises(ValueError, match="must match student-selected token coordinates"):
        validate_teacher_evidence(request, replace(evidence, teacher_on_student_logprobs=torch.zeros(1, 2, 3)))
    with pytest.raises(ValueError, match="exact student-selected token IDs"):
        validate_teacher_evidence(request, replace(evidence, student_topk_indices=torch.tensor([[[23, 21], [22, 24]]])))
    with pytest.raises(ValueError, match="student-selected behavior logprobs must match"):
        validate_teacher_score_request(replace(request, behavior_topk_logprobs=torch.zeros(1, 2, 1)))
    with pytest.raises(ValueError, match="selected-score request mask"):
        validate_teacher_evidence(request, replace(evidence, valid_mask=torch.tensor([[True, False]])))

    prepared = prepare_student_topk_policy_surrogate(request, evidence, coefficient=0.5, route_weights=torch.ones(1, 2))
    validate_distillation_attachment(
        evidence, prepared, trajectory_ids=request.trajectory_ids, response_mask=request.response_mask
    )
    with pytest.raises(ValueError, match="selected token IDs do not match teacher evidence"):
        validate_distillation_attachment(
            evidence,
            replace(prepared, student_topk_indices=torch.tensor([[[21, 23], [22, 25]]])),
            trajectory_ids=request.trajectory_ids,
            response_mask=request.response_mask,
        )


def test_teacher_topk_prompt_scores_cannot_substitute_for_selected_id_scores():
    request = _request()
    with pytest.raises(ValueError, match="cannot score arbitrary student-selected token IDs"):
        teacher_evidence_from_prompt_logprobs(
            request,
            teacher_revision="math-revision",
            prompt_lengths=(2,),
            prompt_logprobs=((None, {11: -1.0}, {21: -0.2}, {22: -0.3}),),
        )


@pytest.mark.asyncio
async def test_shared_coordinator_prepares_student_selected_input_from_oracle():
    request = _request()

    class SelectedTokenOracle:
        capabilities = TeacherCapabilities(
            teacher_id="math",
            teacher_revision="math-revision",
            tokenizer_fingerprint="sha256:shared-tokenizer",
            evidence_kinds=frozenset({TeacherEvidenceKind.STUDENT_SELECTED_TOPK}),
            max_sequence_length=8,
            supports_prompt_token_scoring=True,
            max_concurrency=2,
        )

        async def score(self, score_request: TeacherScoreRequest) -> StudentSelectedTeacherEvidence:
            return _evidence(score_request, torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]]])))

        async def close(self) -> None:
            return None

    async def start_oracle():
        return SelectedTokenOracle()

    owner = await TeacherOracleOwner.create({"math": start_oracle})
    coordinator = TeacherEvidenceCoordinator(owner)
    try:
        scored = await coordinator.score(TeacherScoringWork(request, 0.5, torch.tensor([[1.0, 0.25]])))
    finally:
        await coordinator.close()

    assert isinstance(scored.distillation, StudentTopKPolicySurrogateInput)
    torch.testing.assert_close(scored.distillation.student_topk_indices, request.student_topk_indices)
    torch.testing.assert_close(scored.distillation.loss_weights, torch.tensor([[0.5, 0.125]]))


def test_routed_student_selected_scores_restore_original_rows_and_weights():
    requests = (_request("sample-1", "code"), _request("sample-0", "math"))
    scores = (
        torch.log(torch.tensor([[[0.1, 0.7], [0.2, 0.6]]])),
        torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]]])),
    )
    partitions = []
    scored = []
    for original_index, request, teacher_scores in zip((1, 0), requests, scores, strict=True):
        evidence = _evidence(request, teacher_scores)
        work = TeacherScoringWork(request, 0.5, torch.tensor([[1.0, 0.25]]))
        partitions.append(RoutedTeacherScoringPartition((original_index,), work))
        scored.append(
            ScoredDistillationBatch(
                evidence,
                prepare_student_topk_policy_surrogate(
                    request, evidence, coefficient=work.coefficient, route_weights=work.route_weights
                ),
            )
        )
    routes = tuple(TeacherRoute(teacher_id, teacher_id, "opd", 1.0, "routing-1") for teacher_id in ("math", "code"))
    routed = RoutedTeacherScoringWork(
        trajectory_ids=("sample-0", "sample-1"),
        routes=routes,
        response_lengths=(2, 2),
        plan_version="routing-1",
        partitions=tuple(partitions),
    )

    assembled = TeacherEvidenceCoordinator.assemble_routed(routed, tuple(scored))

    assert isinstance(assembled.distillation, StudentTopKPolicySurrogateInput)
    torch.testing.assert_close(assembled.distillation.teacher_on_student_logprobs[0], scores[1][0])
    torch.testing.assert_close(assembled.distillation.teacher_on_student_logprobs[1], scores[0][0])
    torch.testing.assert_close(assembled.distillation.loss_weights, torch.tensor([[0.5, 0.125], [0.5, 0.125]]))
    assert assembled.teacher_revisions == ("math-revision", "code-revision")


def test_build_scoring_work_preserves_admitted_student_selected_tokens():
    batch = {
        "trajectory_ids": [TrajectoryID("first", 0), TrajectoryID("second", 0)],
        "prompt_token_ids": [[11, 12], [13]],
        "response_ids": [[21, 22], [31]],
        "loss_masks": [[1, 1], [1]],
        "student_topk_indices": [[[21, 23], [22, 24]], [[31, 32]]],
        "behavior_topk_logprobs": [[[-0.2, -2.0], [-0.3, -1.7]], [[-0.4, -1.4]]],
    }
    work = build_teacher_scoring_work(
        batch,
        route_ids=("math", "math"),
        teacher_id="math",
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routing-1",
        coefficient=0.5,
        route_weights=(1.0, 0.25),
        evidence=TeacherEvidenceKind.STUDENT_SELECTED_TOPK,
        top_k=2,
    )

    validate_teacher_score_request(work.request)
    assert work.request.student_topk_indices.tolist() == [[[21, 23], [22, 24]], [[31, 32], [-1, -1]]]
    assert torch.isnan(work.request.behavior_topk_logprobs[1, 1]).all()
    torch.testing.assert_close(work.route_weights, torch.tensor([[1.0, 1.0], [0.25, 0.0]]))


def test_build_scoring_work_masks_nontraining_response_tokens():
    batch = {
        "trajectory_ids": [TrajectoryID("tool-step", 0)],
        "prompt_token_ids": [[11]],
        "response_ids": [[21, 99, 22]],
        "loss_masks": [[1, 0, 1]],
        "student_topk_indices": [[[21, 23], [99, 98], [22, 24]]],
        "behavior_topk_logprobs": [[[-0.2, -2.0], [-0.1, -2.5], [-0.3, -1.7]]],
    }
    work = build_teacher_scoring_work(
        batch,
        route_ids=("math",),
        teacher_id="math",
        tokenizer_fingerprint="sha256:shared-tokenizer",
        plan_version="routing-1",
        coefficient=0.5,
        route_weights=(1.0,),
        evidence=TeacherEvidenceKind.STUDENT_SELECTED_TOPK,
        top_k=2,
    )

    validate_teacher_score_request(work.request)
    assert work.request.student_selected_mask.tolist() == [[True, False, True]]
    assert work.request.student_topk_indices.tolist() == [[[21, 23], [-1, -1], [22, 24]]]
    assert torch.isnan(work.request.behavior_topk_logprobs[0, 1]).all()
    evidence = _evidence(
        work.request,
        torch.tensor([[[-0.1, -2.0], [torch.nan, torch.nan], [-0.2, -1.7]]]),
    )
    validate_teacher_evidence(work.request, evidence)
    prepared = prepare_student_topk_policy_surrogate(
        work.request, evidence, coefficient=0.5, route_weights=work.route_weights
    )
    validate_distillation_attachment(
        evidence, prepared, trajectory_ids=work.request.trajectory_ids, response_mask=work.request.response_mask
    )


def test_student_selected_rollout_scores_survive_group_accumulation_without_sentinel_fallback():
    def group(token_id: int):
        return {
            "prompt_token_ids": [[11]],
            "response_ids": [[token_id]],
            "rewards": [1.0],
            "loss_masks": [[1]],
            "rollout_logprobs": None,
            "student_topk_indices": [[[token_id, token_id + 1]]],
            "behavior_topk_logprobs": [[[-0.2, -2.0]]],
        }

    first, second = group(21), group(31)
    merged = concatenate_trajectory_batches([first, second], tis_lcs_alert_threshold=0.005)
    assert merged["student_topk_indices"] == [[[21, 22]], [[31, 32]]]
    assert merged["behavior_topk_logprobs"] == [[[-0.2, -2.0]], [[-0.2, -2.0]]]

    second.pop("behavior_topk_logprobs")
    with pytest.raises(ValueError, match="missing rollout scores"):
        concatenate_trajectory_batches([first, second], tis_lcs_alert_threshold=0.005)
