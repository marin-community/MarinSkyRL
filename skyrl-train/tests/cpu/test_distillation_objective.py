from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf

from marinskyrl.distillation import TeacherEvidenceKind, DistillationObjectiveKind
from skyrl_train.config.objective_spec import LossReduction, TopKLossParams
from skyrl_train.distillation import (
    DISTILLATION_TOPK_METRIC,
    TeacherTopKInput,
    ChosenTokenTeacherInput,
    StudentTopKInput,
    TeacherScoreRequest,
    TopKTeacherEvidence,
    distillation_input_from_tensors,
    prepare_sampled_reverse_kl,
    prepare_sparse_forward_kl,
    student_topk_logprobs,
    validate_sampled_reverse_kl_attachment,
    validate_teacher_evidence,
)
from skyrl_train.distillation_adapters import build_teacher_scoring_work
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector
from skyrl_train.objective.losses import importance_sampling_policy_loss
from skyrl_train.objective.objective import TopKTeacherBatch, build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts, reduce_to_step
from skyrl_train.objective.teacher import teacher_advantages, topk_teacher_loss, mask_teacher_evidence


def _topk_params(distillation, config):
    kind = (
        DistillationObjectiveKind.SPARSE_FORWARD_KL
        if isinstance(distillation, TeacherTopKInput)
        else DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE
    )
    return TopKLossParams(kind, config.eps_clip_low, config.eps_clip_high, config.clip_ratio_c)


def _topk_teacher_row(student_selected_logprobs, distillation, loss_mask, config=None):
    config = _policy_config() if config is None else config
    evidence = mask_teacher_evidence(distillation, loss_mask)
    row = topk_teacher_loss(evidence, student_selected_logprobs, _topk_params(evidence, config))
    weight = evidence.valid_mask * loss_mask
    counts = step_counts(
        [weight], [weight], [weight], [torch.zeros_like(weight)], config.max_seq_len, lambda value: value
    )
    return reduce_to_step(
        row.values,
        weight,
        counts.teacher,
        LossReduction(config.loss_reduction),
        max_seq_len=counts.max_seq_len,
        nonzero_advantage_rows=0,
        numerator_weights=evidence.loss_weights,
    ), row.metrics


def _composed_objective(
    *,
    action_log_probs,
    old_action_log_probs,
    base_action_log_probs,
    advantages,
    loss_mask,
    rollout_logprobs,
    response_span_tags,
    token_entropy,
    config,
    policy_loss_fn,
    distillation=None,
    student_topk_logprobs=None,
):
    teacher = None
    if distillation is not None:
        if config.distillation.reward_mode == "replace":
            loss_mask = loss_mask * distillation.valid_mask
            advantages = torch.zeros_like(advantages)
        if isinstance(distillation, ChosenTokenTeacherInput):
            credit, _ = teacher_advantages(
                distillation.teacher_action_log_probs,
                old_action_log_probs,
                distillation.valid_mask & loss_mask.bool(),
                distillation.loss_weights,
                None,
            )
            advantages = advantages + credit
        else:
            teacher = TopKTeacherBatch(distillation, student_topk_logprobs, _topk_params(distillation, config))
    batch = build_objective_micro_batch(
        action_log_probs=action_log_probs,
        old_action_log_probs=old_action_log_probs,
        base_action_log_probs=base_action_log_probs,
        advantages=advantages,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
        response_span_tags=response_span_tags,
        token_entropy=token_entropy,
        think_token_weight=config.think_token_weight,
        teacher=teacher,
    )
    teacher_weights = [] if teacher is None else [loss_mask * teacher.evidence.valid_mask]
    counts = step_counts(
        [batch.policy_data_weights], [loss_mask], teacher_weights, [advantages], config.max_seq_len, lambda value: value
    )
    return compute_policy_objective(
        batch, loss=policy_loss_fn, counts=counts, config=config, loss_scale=1, report_scale=1
    )


def _request() -> TeacherScoreRequest:
    return TeacherScoreRequest(
        trajectory_ids=("math_0", "swe_0"),
        route_ids=("math", "swe"),
        teacher_id="teacher-a",
        tokenizer_fingerprint="sha256:student-tokenizer",
        plan_version="mopd-v1",
        prompt_token_ids=torch.tensor([[0, 11], [12, 13]]),
        prompt_mask=torch.tensor([[False, True], [True, True]]),
        response_token_ids=torch.tensor([[21, 22, 23], [31, 0, 0]]),
        response_mask=torch.tensor([[True, True, True], [True, False, False]]),
        evidence=TeacherEvidenceKind.CHOSEN_TOKEN,
    )


def _policy_config(loss_reduction: str = "token_mean", reward_mode: str = "add"):
    return OmegaConf.create(
        {
            "policy_loss_type": "importance_sampling",
            "loss_reduction": loss_reduction,
            "max_seq_len": 3,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_estimator_type": "k1",
            "use_tis": False,
            "tis_imp_ratio_cap": 2.0,
            "distillation": {"reward_mode": reward_mode},
        }
    )


def _objective(action_log_probs, teacher_logprobs):
    old_logprobs = torch.full_like(action_log_probs, -1.0)
    distillation = ChosenTokenTeacherInput(
        teacher_action_log_probs=teacher_logprobs,
        valid_mask=torch.isfinite(teacher_logprobs),
        loss_weights=torch.ones_like(action_log_probs),
    )
    return _composed_objective(
        action_log_probs=action_log_probs,
        old_action_log_probs=old_logprobs,
        base_action_log_probs=None,
        advantages=torch.zeros_like(action_log_probs),
        loss_mask=torch.ones_like(action_log_probs),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(action_log_probs),
        config=_policy_config(reward_mode="replace"),
        policy_loss_fn=importance_sampling_policy_loss,
        distillation=distillation,
    )


def test_chosen_token_evidence_preserves_identity_and_masks_invalid_positions(chosen_teacher_evidence):
    request = _request()
    evidence = chosen_teacher_evidence

    validate_teacher_evidence(request, evidence)
    prepared = prepare_sampled_reverse_kl(
        request,
        evidence,
        coefficient=0.5,
        route_weights=torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.6, 0.6]]),
    )

    torch.testing.assert_close(
        prepared.loss_weights,
        torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
    )
    validate_sampled_reverse_kl_attachment(
        evidence,
        prepared,
        trajectory_ids=request.trajectory_ids,
        response_mask=request.response_mask,
    )


def test_teacher_evidence_rejects_plausible_score_at_invalid_position(chosen_teacher_evidence):
    evidence = chosen_teacher_evidence
    evidence.chosen_logprobs[0, 2] = -0.7

    with pytest.raises(ValueError, match="invalid teacher chosen_logprobs must be NaN"):
        validate_teacher_evidence(_request(), evidence)


def test_teacher_evidence_rejects_wrong_trajectory_join(chosen_teacher_evidence):
    evidence = chosen_teacher_evidence
    request = _request()
    request = replace(request, trajectory_ids=("swe_0", "math_0"))

    with pytest.raises(ValueError, match="trajectory_ids do not match"):
        validate_teacher_evidence(request, evidence)


def test_topk_teacher_evidence_validates_as_a_distinct_transport_variant():
    request = replace(_request(), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2)
    valid_mask = torch.tensor([[True, True, False], [True, False, False]])
    evidence = TopKTeacherEvidence(
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        teacher_id=request.teacher_id,
        teacher_revision="teacher-revision",
        plan_version="mopd-v1",
        valid_mask=valid_mask,
        topk_indices=torch.tensor([[[1, 2], [3, 4], [-1, -1]], [[5, 6], [-1, -1], [-1, -1]]]),
        topk_logprobs=torch.log(
            torch.tensor(
                [
                    [[0.8, 0.15], [0.6, 0.2], [torch.nan, torch.nan]],
                    [[0.7, 0.2], [torch.nan, torch.nan], [torch.nan, torch.nan]],
                ]
            )
        ),
        retained_mass=torch.tensor([[0.95, 0.8, torch.nan], [0.9, torch.nan, torch.nan]]),
    )

    validate_teacher_evidence(request, evidence)


def _sparse_objective(student_logits: torch.Tensor, teacher_probs: torch.Tensor, topk: int):
    teacher_topk_probs, teacher_indices = teacher_probs.topk(topk, dim=-1)
    retained_mass = teacher_topk_probs.sum(dim=-1)
    valid_mask = torch.ones(teacher_probs.shape[:2], dtype=torch.bool)
    distillation = TeacherTopKInput(
        teacher_topk_indices=teacher_indices,
        teacher_topk_logprobs=teacher_topk_probs.log(),
        retained_mass=retained_mass,
        valid_mask=valid_mask,
        loss_weights=torch.ones_like(retained_mass),
    )
    selected_student_logprobs = student_topk_logprobs(student_logits, teacher_indices)
    return _composed_objective(
        action_log_probs=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        old_action_log_probs=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        base_action_log_probs=None,
        advantages=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        loss_mask=torch.ones(teacher_probs.shape[:2]),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        config=_policy_config(),
        policy_loss_fn=importance_sampling_policy_loss,
        distillation=distillation,
        student_topk_logprobs=selected_student_logprobs,
    )


def test_sparse_forward_kl_full_vocabulary_matches_dense_kl_and_gradient():
    teacher_logits = torch.tensor([[[2.0, 1.0, -0.5, -1.0]]], dtype=torch.float64)
    teacher_probs = teacher_logits.softmax(dim=-1)
    student_logits = torch.tensor([[[0.1, 0.4, -0.2, 0.8]]], dtype=torch.float64, requires_grad=True)

    objective = _sparse_objective(student_logits, teacher_probs, topk=4)
    dense_kl = torch.sum(teacher_probs * (teacher_probs.log() - student_logits.log_softmax(dim=-1)))
    objective.optimization_loss.backward()

    torch.testing.assert_close(objective.optimization_loss, dense_kl)
    torch.testing.assert_close(student_logits.grad, student_logits.softmax(dim=-1) - teacher_probs)
    assert objective.metrics["distillation_retained_mass_mean"] == pytest.approx(1.0)
    assert objective.metrics["distillation_topk"] == 4


def test_sparse_forward_kl_accepts_full_mass_float_roundoff():
    teacher_probs = torch.tensor([[[0.6, 0.4]]], dtype=torch.float64)
    teacher_indices = torch.tensor([[[0, 1]]])
    distillation = TeacherTopKInput(
        teacher_topk_indices=teacher_indices,
        teacher_topk_logprobs=teacher_probs.log(),
        retained_mass=torch.tensor([[1.0 + 1e-12]], dtype=torch.float64),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )

    loss, _ = _topk_teacher_row(
        student_topk_logprobs(torch.zeros_like(teacher_probs), teacher_indices),
        distillation,
        torch.ones((1, 1)),
    )

    assert torch.isfinite(loss)


def test_sparse_forward_kl_reports_top20_top256_and_full_retained_mass_and_learns():
    vocabulary_size = 300
    teacher_logits = -torch.arange(vocabulary_size, dtype=torch.float64).view(1, 1, -1) / 40
    teacher_probs = teacher_logits.softmax(dim=-1)
    retained_masses = []
    losses = []
    for topk in (20, 256, vocabulary_size):
        student_logits = torch.linspace(-0.5, 0.5, vocabulary_size, dtype=torch.float64).view(1, 1, -1)
        student_logits.requires_grad_()
        before = _sparse_objective(student_logits, teacher_probs, topk)
        before.optimization_loss.backward()
        with torch.no_grad():
            updated_logits = student_logits - 0.1 * student_logits.grad
        after = _sparse_objective(updated_logits, teacher_probs, topk)

        retained_masses.append(before.metrics["distillation_retained_mass_mean"])
        losses.append(before.metrics["distillation_loss"])
        assert after.optimization_loss < before.optimization_loss

    assert retained_masses[0] < retained_masses[1] < retained_masses[2]
    assert retained_masses[2] == pytest.approx(1.0)
    assert abs(losses[1] - losses[2]) < abs(losses[0] - losses[2])


def test_prepare_sparse_forward_kl_preserves_mass_and_route_weights():
    request = replace(_request(), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=2)
    evidence = TopKTeacherEvidence(
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        teacher_id=request.teacher_id,
        teacher_revision="teacher-revision",
        plan_version=request.plan_version,
        valid_mask=request.response_mask,
        topk_indices=torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [-1, -1], [-1, -1]]]),
        topk_logprobs=torch.log(
            torch.tensor(
                [[[0.7, 0.2], [0.6, 0.2], [0.5, 0.1]], [[0.8, 0.15], [torch.nan, torch.nan], [torch.nan, torch.nan]]]
            )
        ),
        retained_mass=torch.tensor([[0.9, 0.8, 0.6], [0.95, torch.nan, torch.nan]]),
    )

    prepared = prepare_sparse_forward_kl(
        request,
        evidence,
        coefficient=0.5,
        route_weights=torch.tensor([[0.4, 0.4, 0.4], [0.6, 0.0, 0.0]]),
    )

    torch.testing.assert_close(prepared.retained_mass, evidence.retained_mass, equal_nan=True)
    torch.testing.assert_close(prepared.loss_weights, torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.0, 0.0]]))


def test_topk_teacher_evidence_rejects_inconsistent_retained_mass():
    request = replace(_request(), evidence=TeacherEvidenceKind.TOPK_DISTRIBUTION, top_k=1)
    evidence = TopKTeacherEvidence(
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        teacher_id=request.teacher_id,
        teacher_revision="teacher-revision",
        plan_version=request.plan_version,
        valid_mask=request.response_mask,
        topk_indices=torch.tensor([[[1], [2], [3]], [[4], [-1], [-1]]]),
        topk_logprobs=torch.log(torch.tensor([[[0.9], [0.8], [0.7]], [[0.6], [torch.nan], [torch.nan]]])),
        retained_mass=torch.tensor([[0.8, 0.8, 0.7], [0.6, torch.nan, torch.nan]]),
    )

    with pytest.raises(ValueError, match="retained_mass must equal"):
        validate_teacher_evidence(request, evidence)


def test_sampled_reverse_kl_gradient_depends_on_teacher_distribution():
    first_actions = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    first = _objective(first_actions, torch.tensor([[-0.5, -2.0, torch.nan]], dtype=torch.float64))
    first.optimization_loss.backward()

    second_actions = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    second = _objective(second_actions, torch.tensor([[-2.0, -2.0, torch.nan]], dtype=torch.float64))
    second.optimization_loss.backward()

    torch.testing.assert_close(first_actions.grad, torch.tensor([[-0.25, 0.5, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(second_actions.grad, torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64))
    assert first.rows.policy.item() == pytest.approx(0.25)
    assert second.rows.policy.item() == pytest.approx(1.0)


def test_replace_mode_uses_teacher_credit_with_configured_kl_and_entropy():
    actions = torch.tensor([[-0.8, -1.2]], dtype=torch.float64, requires_grad=True)
    old_actions = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    teacher_actions = torch.tensor([[-0.5, -2.0]], dtype=torch.float64)
    distillation = ChosenTokenTeacherInput(
        teacher_action_log_probs=teacher_actions,
        valid_mask=torch.ones_like(actions, dtype=torch.bool),
        loss_weights=torch.ones_like(actions),
    )
    config = _policy_config(reward_mode="replace")
    config.use_entropy_loss = True
    config.entropy_loss_coef = 0.7
    config.use_kl_loss = True
    config.kl_loss_coef = 0.4

    objective = _composed_objective(
        action_log_probs=actions,
        old_action_log_probs=old_actions,
        base_action_log_probs=torch.tensor([[-1.4, -0.8]], dtype=torch.float64),
        advantages=torch.tensor([[3.0, -2.0]], dtype=torch.float64),
        loss_mask=torch.ones_like(actions),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.tensor([[0.6, 0.8]], dtype=torch.float64),
        config=config,
        policy_loss_fn=importance_sampling_policy_loss,
        distillation=distillation,
    )
    expected_policy = (torch.exp(actions - old_actions) * (old_actions - teacher_actions)).mean()
    torch.testing.assert_close(objective.optimization_loss, expected_policy + 0.4 * 0.1 - 0.7 * 0.7)
    torch.testing.assert_close(objective.rows.policy, expected_policy)
    assert objective.rows.entropy.item() == pytest.approx(0.7)
    assert objective.rows.kl.item() == pytest.approx(0.1)
    objective.optimization_loss.backward()
    expected_gradient = torch.exp(actions.detach() - old_actions) * (old_actions - teacher_actions) / actions.numel()
    torch.testing.assert_close(actions.grad, expected_gradient + 0.4 / 2)


def test_student_topk_surrogate_matches_selected_policy_gradient_at_behavior_policy():
    behavior_probs = torch.tensor([0.45, 0.35, 0.20], dtype=torch.float64)
    teacher_probs = torch.tensor([0.55, 0.40, 0.05], dtype=torch.float64)
    logits = behavior_probs.log().reshape(1, 1, 3).detach().requires_grad_()
    selected_ids = torch.tensor([[[0, 1]]])
    behavior_logprobs = behavior_probs.log()[selected_ids]
    teacher_logprobs = teacher_probs.log()[selected_ids]
    distillation = StudentTopKInput(
        student_topk_indices=selected_ids,
        behavior_topk_logprobs=behavior_logprobs,
        teacher_on_student_logprobs=teacher_logprobs,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    selected_current_logprobs = student_topk_logprobs(logits, selected_ids)
    objective = _composed_objective(
        action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
        old_action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
        base_action_log_probs=None,
        advantages=torch.zeros((1, 1), dtype=torch.float64),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros((1, 1), dtype=torch.float64),
        student_topk_logprobs=selected_current_logprobs,
        loss_mask=torch.ones((1, 1), dtype=torch.bool),
        config=_policy_config(reward_mode="replace"),
        policy_loss_fn=importance_sampling_policy_loss,
        distillation=distillation,
    )
    loss = objective.optimization_loss
    loss.backward()

    weighted_gaps = behavior_probs[:2] / behavior_probs[:2].sum() * (behavior_probs[:2].log() - teacher_probs[:2].log())
    expected_gradient = torch.zeros_like(behavior_probs)
    expected_gradient[:2] = weighted_gaps
    expected_gradient -= weighted_gaps.sum() * behavior_probs
    torch.testing.assert_close(loss, weighted_gaps.sum())
    torch.testing.assert_close(logits.grad.reshape(-1), expected_gradient)
    assert loss.item() < 0  # A selected-ID reward sum need not be a KL divergence.
    assert objective.rows.policy.item() == 0.0


def test_student_topk_surrogate_clips_improving_high_ratio_update():
    logits = torch.log(torch.tensor([[[0.9, 0.1]]], dtype=torch.float64)).requires_grad_()
    distillation = StudentTopKInput(
        student_topk_indices=torch.tensor([[[0]]]),
        behavior_topk_logprobs=torch.log(torch.tensor([[[0.6]]], dtype=torch.float64)),
        teacher_on_student_logprobs=torch.log(torch.tensor([[[0.95]]], dtype=torch.float64)),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    loss, metrics = _topk_teacher_row(
        distillation=distillation,
        student_selected_logprobs=student_topk_logprobs(logits, distillation.student_topk_indices),
        loss_mask=torch.ones((1, 1), dtype=torch.bool),
        config=_policy_config(),
    )
    loss.backward()

    expected_advantage = -(torch.log(torch.tensor(0.9 / 0.95, dtype=torch.float64)))
    torch.testing.assert_close(loss, -1.2 * expected_advantage)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    assert metrics["distillation_clip_fraction"] == 1.0


def test_student_topk_surrogate_applies_negative_advantage_dual_clip():
    logits = torch.log(torch.tensor([[[0.95, 0.02, 0.02, 0.01]]], dtype=torch.float64)).requires_grad_()
    distillation = StudentTopKInput(
        student_topk_indices=torch.tensor([[[0]]]),
        behavior_topk_logprobs=torch.log(torch.tensor([[[0.3]]], dtype=torch.float64)),
        teacher_on_student_logprobs=torch.log(torch.tensor([[[0.05]]], dtype=torch.float64)),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    loss, metrics = _topk_teacher_row(
        distillation=distillation,
        student_selected_logprobs=student_topk_logprobs(logits, distillation.student_topk_indices),
        loss_mask=torch.ones((1, 1), dtype=torch.bool),
        config=_policy_config(),
    )
    loss.backward()

    expected_advantage = -torch.log(torch.tensor(0.95 / 0.05, dtype=torch.float64))
    torch.testing.assert_close(loss, -3.0 * expected_advantage)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    assert metrics["distillation_dual_clip_fraction"] == 1.0


def test_best_of_n_teacher_scoring_requests_only_the_selected_trajectory():
    selection = BestOfNTrajectorySelector(2).select(
        {
            "prompt_token_ids": [[10], [10]],
            "response_ids": [[20, 21], [30, 31]],
            "rewards": [[0.1, 0.0], [0.0, 0.9]],
            "loss_masks": [[1, 1], [1, 1]],
            "trajectory_ids": [TrajectoryID("math", 0), TrajectoryID("math", 1)],
        },
        ["math", "math"],
    )
    work = build_teacher_scoring_work(
        selection.trajectory_batch,
        route_ids=("math",),
        teacher_id="teacher-a",
        tokenizer_fingerprint="sha256:student-tokenizer",
        plan_version="opd-v1",
        coefficient=0.5,
        route_weights=(1.0,),
    )

    assert work.request.trajectory_ids == ("math_1",)
    torch.testing.assert_close(work.request.response_token_ids, torch.tensor([[30, 31]]))


def test_training_batch_iterator_requires_driver_to_consume_chosen_teacher():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action_log_probs": torch.zeros(2, 2),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(2, 2),
            "advantages": torch.zeros(2, 2),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "loss_mask": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.long),
            "teacher_action_log_probs": torch.tensor([[-0.5, -0.7], [-0.2, -0.3]]),
            "teacher_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "distillation_loss_weights": torch.tensor([[0.4, 0.4], [0.6, 0.6]]),
        }
    )
    batch.metadata = {"response_length": 2}

    with pytest.raises(ValueError, match="chosen-token teacher tensors must be consumed on the driver"):
        list(TrainingBatchIterator(batch, sample_batch_size=1))


def test_student_topk_payload_rejects_mixed_or_partial_evidence():
    base = dict(
        teacher_action_log_probs=None,
        teacher_topk_indices=None,
        teacher_topk_logprobs=None,
        teacher_retained_mass=None,
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
        loss_weights=torch.ones(1, 1),
        student_topk_indices=torch.tensor([[[1, 2]]]),
        behavior_topk_logprobs=torch.tensor([[[-0.5, -0.8]]]),
        teacher_on_student_logprobs=torch.tensor([[[-0.4, -0.9]]]),
    )
    with pytest.raises(ValueError, match="cannot mix objective evidence variants"):
        distillation_input_from_tensors(**(base | {"teacher_topk_indices": torch.tensor([[[1]]])}))
    with pytest.raises(ValueError, match="selected IDs, behavior logprobs, and teacher scores together"):
        distillation_input_from_tensors(**(base | {"teacher_on_student_logprobs": None}))


def test_partial_distillation_payload_fails_closed():
    with pytest.raises(ValueError, match="evidence requires valid_mask and loss_weights"):
        distillation_input_from_tensors(
            teacher_action_log_probs=None,
            teacher_topk_indices=torch.tensor([[[1]]]),
            teacher_topk_logprobs=None,
            teacher_retained_mass=None,
            valid_mask=None,
            loss_weights=torch.ones(1, 1),
        )


def test_sampled_reverse_kl_all_masked_micro_batch_contributes_zero_with_gradient():
    """A trajectory masked by the agent loop is a whole micro-batch at micro batch size one."""
    actions = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64, requires_grad=True)
    objective = _objective(actions, torch.full((1, 3), torch.nan, dtype=torch.float64))
    objective.optimization_loss.backward()

    assert objective.optimization_loss.item() == 0.0
    assert objective.rows.policy.item() == 0.0
    torch.testing.assert_close(actions.grad, torch.zeros_like(actions))


def test_sparse_forward_kl_all_masked_micro_batch_contributes_zero_with_gradient():
    student_logits = torch.log(torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float64)).requires_grad_()
    teacher_probs = torch.tensor([[[0.6, 0.3, 0.1]]], dtype=torch.float64)
    teacher_topk_probs, teacher_indices = teacher_probs.topk(3, dim=-1)
    distillation = TeacherTopKInput(
        teacher_topk_indices=teacher_indices,
        teacher_topk_logprobs=teacher_topk_probs.log(),
        retained_mass=teacher_topk_probs.sum(dim=-1),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    selected = student_topk_logprobs(student_logits, teacher_indices)

    loss, metrics = _topk_teacher_row(selected, distillation, torch.zeros((1, 1)))
    loss.backward()

    assert loss.item() == 0.0
    assert metrics[DISTILLATION_TOPK_METRIC] == 3.0
    assert set(metrics) == {
        DISTILLATION_TOPK_METRIC,
        "distillation_retained_mass_mean",
        "distillation_retained_mass_min",
    }
    torch.testing.assert_close(student_logits.grad, torch.zeros_like(student_logits))


def test_student_topk_surrogate_all_masked_micro_batch_contributes_zero_with_gradient():
    probs = torch.tensor([0.45, 0.35, 0.20], dtype=torch.float64)
    logits = probs.log().reshape(1, 1, 3).detach().requires_grad_()
    selected_ids = torch.tensor([[[0, 1]]])
    distillation = StudentTopKInput(
        student_topk_indices=selected_ids,
        behavior_topk_logprobs=probs.log()[selected_ids],
        teacher_on_student_logprobs=probs.log()[selected_ids],
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    selected = student_topk_logprobs(logits, selected_ids)

    loss, metrics = _topk_teacher_row(
        selected, distillation, torch.zeros((1, 1), dtype=torch.bool), _policy_config(reward_mode="replace")
    )
    loss.backward()

    assert loss.item() == 0.0
    assert metrics[DISTILLATION_TOPK_METRIC] == 2.0
    assert set(metrics) == {
        DISTILLATION_TOPK_METRIC,
        "distillation_student_retained_mass_mean",
        "distillation_clip_fraction",
        "distillation_dual_clip_fraction",
    }
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize(
    ("evidence", "payload_type", "field", "expected_second_row"),
    [
        (
            {
                "teacher_topk_indices": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                "teacher_topk_logprobs": torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]], [[0.5, 0.2], [0.4, 0.2]]])),
                "teacher_retained_mass": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
            },
            TeacherTopKInput,
            "retained_mass",
            torch.tensor([[0.7, 0.6]]),
        ),
        (
            {
                "student_topk_indices": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                "behavior_topk_logprobs": torch.log(torch.tensor([[[0.4, 0.3], [0.5, 0.2]], [[0.6, 0.1], [0.3, 0.3]]])),
                "teacher_on_student_logprobs": torch.log(
                    torch.tensor([[[0.5, 0.2], [0.4, 0.3]], [[0.7, 0.1], [0.2, 0.2]]])
                ),
            },
            StudentTopKInput,
            "teacher_on_student_logprobs",
            torch.log(torch.tensor([[[0.7, 0.1], [0.2, 0.2]]])),
        ),
    ],
    ids=["sparse_forward_kl", "student_selected_topk"],
)
def test_training_batch_iterator_slices_distillation_payload_per_micro_batch(
    evidence, payload_type, field, expected_second_row
):
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "action_log_probs": torch.zeros(2, 2),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(2, 2),
            "advantages": torch.zeros(2, 2),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "loss_mask": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.long),
            "teacher_valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "distillation_loss_weights": torch.tensor([[0.4, 0.4], [0.6, 0.6]]),
            **evidence,
        }
    )
    batch.metadata = {"response_length": 2}

    experiences = list(TrainingBatchIterator(batch, sample_batch_size=1))

    assert len(experiences) == 2
    distillation = experiences[1].distillation
    assert isinstance(distillation, payload_type)
    torch.testing.assert_close(getattr(distillation, field), expected_second_row)
    torch.testing.assert_close(distillation.loss_weights, torch.tensor([[0.6, 0.6]]))
