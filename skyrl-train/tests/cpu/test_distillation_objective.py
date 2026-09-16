from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config, Qwen3ForCausalLM

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    ChosenTokenTeacherEvidence,
    SparseForwardKLInput,
    SampledReverseKLInput,
    StudentTopKPolicySurrogateInput,
    TeacherScoreRequest,
    TopKTeacherEvidence,
    distillation_input_from_tensors,
    prepare_sampled_reverse_kl,
    prepare_sparse_forward_kl,
    sparse_forward_kl_loss,
    student_topk_logprobs,
    validate_sampled_reverse_kl_attachment,
    validate_teacher_evidence,
)
from skyrl_train.distillation_adapters import build_teacher_scoring_work
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective, ppo_policy_loss, sft_policy_loss


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
            "policy_loss_type": "regular",
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
    distillation = SampledReverseKLInput(
        teacher_action_log_probs=teacher_logprobs,
        valid_mask=torch.isfinite(teacher_logprobs),
        loss_weights=torch.ones_like(action_log_probs),
    )
    return compute_policy_objective(
        action_log_probs=action_log_probs,
        old_action_log_probs=old_logprobs,
        base_action_log_probs=None,
        advantages=torch.zeros_like(action_log_probs),
        loss_mask=torch.ones_like(action_log_probs),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(action_log_probs),
        config=_policy_config(),
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
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
    distillation = SparseForwardKLInput(
        teacher_topk_indices=teacher_indices,
        teacher_topk_logprobs=teacher_topk_probs.log(),
        retained_mass=retained_mass,
        valid_mask=valid_mask,
        loss_weights=torch.ones_like(retained_mass),
    )
    selected_student_logprobs = student_topk_logprobs(student_logits, teacher_indices)
    return compute_policy_objective(
        action_log_probs=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        old_action_log_probs=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        base_action_log_probs=None,
        advantages=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        loss_mask=torch.ones(teacher_probs.shape[:2]),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros(teacher_probs.shape[:2], dtype=student_logits.dtype),
        config=_policy_config(),
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
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
    distillation = SparseForwardKLInput(
        teacher_topk_indices=teacher_indices,
        teacher_topk_logprobs=teacher_probs.log(),
        retained_mass=torch.tensor([[1.0 + 1e-12]], dtype=torch.float64),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )

    loss, _ = sparse_forward_kl_loss(
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
    assert first.metrics["distillation_loss"] == pytest.approx(0.25)
    assert second.metrics["distillation_loss"] == pytest.approx(1.0)


def test_replace_mode_optimizes_only_sampled_reverse_kl():
    actions = torch.tensor([[-0.8, -1.2]], dtype=torch.float64, requires_grad=True)
    old_actions = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    teacher_actions = torch.tensor([[-0.5, -2.0]], dtype=torch.float64)
    distillation = SampledReverseKLInput(
        teacher_action_log_probs=teacher_actions,
        valid_mask=torch.ones_like(actions, dtype=torch.bool),
        loss_weights=torch.ones_like(actions),
    )
    config = _policy_config(reward_mode="replace")
    config.use_entropy_loss = True
    config.entropy_loss_coef = 0.7
    config.use_kl_loss = True
    config.kl_loss_coef = 0.4

    objective = compute_policy_objective(
        action_log_probs=actions,
        old_action_log_probs=old_actions,
        base_action_log_probs=torch.tensor([[-1.4, -0.6]], dtype=torch.float64),
        advantages=torch.tensor([[3.0, -2.0]], dtype=torch.float64),
        loss_mask=torch.ones_like(actions),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.tensor([[0.6, 0.8]], dtype=torch.float64),
        config=config,
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
        distillation=distillation,
    )
    expected = (torch.exp(actions - old_actions) * (old_actions - teacher_actions)).mean()

    torch.testing.assert_close(objective.optimization_loss, expected)
    torch.testing.assert_close(objective.unscaled_loss, expected)
    assert objective.policy_loss.item() == 0.0
    assert objective.entropy.item() == 0.0
    assert objective.kl_loss.item() == 0.0
    objective.optimization_loss.backward()
    expected_gradient = torch.exp(actions.detach() - old_actions) * (old_actions - teacher_actions) / actions.numel()
    torch.testing.assert_close(actions.grad, expected_gradient)


def test_student_topk_surrogate_matches_selected_policy_gradient_at_behavior_policy():
    behavior_probs = torch.tensor([0.45, 0.35, 0.20], dtype=torch.float64)
    teacher_probs = torch.tensor([0.55, 0.40, 0.05], dtype=torch.float64)
    logits = behavior_probs.log().reshape(1, 1, 3).detach().requires_grad_()
    selected_ids = torch.tensor([[[0, 1]]])
    behavior_logprobs = behavior_probs.log()[selected_ids]
    teacher_logprobs = teacher_probs.log()[selected_ids]
    distillation = StudentTopKPolicySurrogateInput(
        student_topk_indices=selected_ids,
        behavior_topk_logprobs=behavior_logprobs,
        teacher_on_student_logprobs=teacher_logprobs,
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    selected_current_logprobs = student_topk_logprobs(logits, selected_ids)
    objective = compute_policy_objective(
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
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
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
    assert objective.policy_loss.item() == 0.0


def test_student_topk_surrogate_clips_improving_high_ratio_update():
    logits = torch.log(torch.tensor([[[0.9, 0.1]]], dtype=torch.float64)).requires_grad_()
    distillation = StudentTopKPolicySurrogateInput(
        student_topk_indices=torch.tensor([[[0]]]),
        behavior_topk_logprobs=torch.log(torch.tensor([[[0.6]]], dtype=torch.float64)),
        teacher_on_student_logprobs=torch.log(torch.tensor([[[0.95]]], dtype=torch.float64)),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    loss, metrics = distillation.objective_loss(
        action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
        old_action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
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
    distillation = StudentTopKPolicySurrogateInput(
        student_topk_indices=torch.tensor([[[0]]]),
        behavior_topk_logprobs=torch.log(torch.tensor([[[0.3]]], dtype=torch.float64)),
        teacher_on_student_logprobs=torch.log(torch.tensor([[[0.05]]], dtype=torch.float64)),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_weights=torch.ones((1, 1), dtype=torch.float64),
    )
    loss, metrics = distillation.objective_loss(
        action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
        old_action_log_probs=torch.zeros((1, 1), dtype=torch.float64),
        student_selected_logprobs=student_topk_logprobs(logits, distillation.student_topk_indices),
        loss_mask=torch.ones((1, 1), dtype=torch.bool),
        config=_policy_config(),
    )
    loss.backward()

    expected_advantage = -torch.log(torch.tensor(0.95 / 0.05, dtype=torch.float64))
    torch.testing.assert_close(loss, -3.0 * expected_advantage)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    assert metrics["distillation_dual_clip_fraction"] == 1.0


def test_replace_mode_rejects_batch_without_teacher_evidence():
    actions = torch.tensor([[-1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="replace requires distillation evidence"):
        compute_policy_objective(
            action_log_probs=actions,
            old_action_log_probs=actions,
            base_action_log_probs=None,
            advantages=torch.ones_like(actions),
            loss_mask=torch.ones_like(actions),
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(actions),
            config=_policy_config(reward_mode="replace"),
            policy_loss_fn=ppo_policy_loss,
            accumulation_steps=1,
            scaling=LossScaling.CALLER,
        )


def test_real_same_vocabulary_teacher_changes_student_gradient_and_update():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
    )
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    response_token_ids = token_ids[:, -2:]

    torch.manual_seed(0)
    initial_student = Qwen3ForCausalLM(config)
    initial_state = {name: value.detach().clone() for name, value in initial_student.state_dict().items()}

    def model(seed: int) -> Qwen3ForCausalLM:
        torch.manual_seed(seed)
        return Qwen3ForCausalLM(config)

    def chosen_logprobs(causal_lm: Qwen3ForCausalLM) -> torch.Tensor:
        response_logits = causal_lm(token_ids).logits[:, -3:-1]
        return response_logits.log_softmax(dim=-1).gather(-1, response_token_ids.unsqueeze(-1)).squeeze(-1)

    def optimize_once(teacher: Qwen3ForCausalLM) -> tuple[torch.Tensor, torch.Tensor]:
        student = model(0)
        student.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        with torch.no_grad():
            teacher_actions = chosen_logprobs(teacher)
        objective = _objective(chosen_logprobs(student), teacher_actions)
        optimizer.zero_grad()
        objective.optimization_loss.backward()
        gradient = torch.cat(
            [parameter.grad.flatten() for parameter in student.parameters() if parameter.grad is not None]
        )
        optimizer.step()
        updated_parameters = torch.cat([parameter.detach().flatten() for parameter in student.parameters()])
        return gradient, updated_parameters

    first_teacher = model(1)
    second_teacher = model(2)
    first_teacher_actions = chosen_logprobs(first_teacher)
    second_teacher_actions = chosen_logprobs(second_teacher)
    first_gradient, first_update = optimize_once(first_teacher)
    second_gradient, second_update = optimize_once(second_teacher)
    initial_parameters = torch.cat([value.flatten() for value in initial_state.values()])

    assert not torch.allclose(first_teacher_actions, second_teacher_actions)
    assert not torch.allclose(first_gradient, second_gradient)
    assert not torch.allclose(first_update, initial_parameters)
    assert not torch.allclose(second_update, initial_parameters)


def test_best_of_n_optional_teacher_changes_only_the_selected_update():
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

    evidence = ChosenTokenTeacherEvidence(
        trajectory_ids=work.request.trajectory_ids,
        route_ids=work.request.route_ids,
        teacher_id=work.request.teacher_id,
        teacher_revision="teacher-revision",
        plan_version=work.request.plan_version,
        valid_mask=work.request.response_mask,
        chosen_logprobs=torch.tensor([[-0.25, -2.0]], dtype=torch.float64),
    )
    distillation = prepare_sampled_reverse_kl(
        work.request,
        evidence,
        coefficient=work.coefficient,
        route_weights=work.route_weights,
    )

    def selected_update(attached_distillation):
        action_logprobs = torch.tensor([[-1.0, -1.0]], dtype=torch.float64, requires_grad=True)
        objective = compute_policy_objective(
            action_log_probs=action_logprobs,
            old_action_log_probs=torch.full_like(action_logprobs, -1.0),
            base_action_log_probs=None,
            advantages=torch.ones_like(action_logprobs),
            loss_mask=torch.ones_like(action_logprobs),
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(action_logprobs),
            config=_policy_config(),
            policy_loss_fn=sft_policy_loss,
            accumulation_steps=1,
            scaling=LossScaling.CALLER,
            distillation=attached_distillation,
        )
        objective.optimization_loss.backward()
        return action_logprobs.grad

    update_without_teacher = selected_update(None)
    update_with_teacher = selected_update(distillation)

    assert not torch.equal(update_without_teacher, update_with_teacher)


def test_distillation_absence_preserves_policy_objective_exactly():
    actions = torch.tensor([[-0.9, -1.1]], dtype=torch.float64, requires_grad=True)
    old = torch.full_like(actions, -1.0)
    advantages = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    config = _policy_config()

    expected_loss, expected_metrics = ppo_policy_loss(
        actions,
        old,
        advantages,
        config=config,
        loss_mask=torch.ones_like(actions),
    )
    result = compute_policy_objective(
        action_log_probs=actions,
        old_action_log_probs=old,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=torch.ones_like(actions),
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(actions),
        config=config,
        policy_loss_fn=ppo_policy_loss,
        accumulation_steps=1,
        scaling=LossScaling.CALLER,
    )

    torch.testing.assert_close(result.optimization_loss, expected_loss, rtol=0, atol=0)
    assert "distillation_loss" not in result.metrics
    assert (
        result.metrics
        == {
            "ppo_clip_ratio": 0.0,
            "ppo_clip_ratio_low": 0.0,
            "ppo_clip_ratio_high": 0.0,
            "ppo_clip_pressure_low": 0.0,
            "ppo_clip_pressure_high": 0.0,
            "ppo_ratio_exact_unit_fraction": 0.0,
        }
        | expected_metrics
    )


@pytest.mark.parametrize("loss_reduction", ["token_mean", "seq_mean_token_sum_norm_global"])
def test_distillation_gradient_matches_caller_and_megatron_scaling(loss_reduction):
    config = _policy_config(loss_reduction)
    accumulation_steps = 3
    common = {
        "old_action_log_probs": torch.full((1, 3), -1.0),
        "base_action_log_probs": None,
        "advantages": torch.zeros(1, 3),
        "loss_mask": torch.ones(1, 3),
        "rollout_logprobs": None,
        "response_span_tags": None,
        "token_entropy": torch.zeros(1, 3),
        "config": config,
        "policy_loss_fn": ppo_policy_loss,
        "accumulation_steps": accumulation_steps,
        "global_loss_denom": 3.0,
        "distillation": SampledReverseKLInput(
            teacher_action_log_probs=torch.tensor([[-0.5, -2.0, -0.25]]),
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
            loss_weights=torch.ones(1, 3),
        ),
    }
    caller_actions = torch.full((1, 3), -1.0, requires_grad=True)
    caller = compute_policy_objective(
        action_log_probs=caller_actions,
        scaling=LossScaling.CALLER,
        **common,
    )
    caller.optimization_loss.backward()

    megatron_actions = torch.full((1, 3), -1.0, requires_grad=True)
    megatron = compute_policy_objective(
        action_log_probs=megatron_actions,
        scaling=LossScaling.MEGATRON_PIPELINE,
        **common,
    )
    (megatron.optimization_loss / accumulation_steps).backward()

    torch.testing.assert_close(megatron_actions.grad, caller_actions.grad, rtol=0, atol=0)


def test_training_batch_iterator_preserves_minimal_distillation_payload():
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

    experiences = list(TrainingBatchIterator(batch, sample_batch_size=1))

    assert len(experiences) == 2
    assert experiences[1].distillation is not None
    torch.testing.assert_close(experiences[1].distillation.teacher_action_log_probs, torch.tensor([[-0.2, -0.3]]))
    torch.testing.assert_close(experiences[1].distillation.loss_weights, torch.tensor([[0.6, 0.6]]))


def test_training_batch_iterator_preserves_sparse_distillation_payload():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3]]),
            "action_log_probs": torch.zeros(1, 2),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(1, 2),
            "advantages": torch.zeros(1, 2),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "loss_mask": torch.ones(1, 2, dtype=torch.long),
            "response_mask": torch.ones(1, 2, dtype=torch.long),
            "teacher_topk_indices": torch.tensor([[[1, 2], [3, 4]]]),
            "teacher_topk_logprobs": torch.log(torch.tensor([[[0.7, 0.2], [0.6, 0.2]]])),
            "teacher_retained_mass": torch.tensor([[0.9, 0.8]]),
            "teacher_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "distillation_loss_weights": torch.tensor([[0.4, 0.4]]),
        }
    )
    batch.metadata = {"response_length": 2}

    [experience] = list(TrainingBatchIterator(batch, sample_batch_size=1))

    assert isinstance(experience.distillation, SparseForwardKLInput)
    torch.testing.assert_close(experience.distillation.retained_mass, torch.tensor([[0.9, 0.8]]))


def test_training_batch_iterator_preserves_student_selected_teacher_scores():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3]]),
            "action_log_probs": torch.zeros(1, 2),
            "base_action_log_probs": None,
            "values": None,
            "returns": torch.zeros(1, 2),
            "advantages": torch.zeros(1, 2),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "loss_mask": torch.ones(1, 2, dtype=torch.long),
            "response_mask": torch.ones(1, 2, dtype=torch.long),
            "student_topk_indices": torch.tensor([[[1, 2], [3, 4]]]),
            "behavior_topk_logprobs": torch.log(torch.tensor([[[0.4, 0.3], [0.5, 0.2]]])),
            "teacher_on_student_logprobs": torch.log(torch.tensor([[[0.5, 0.2], [0.4, 0.3]]])),
            "teacher_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "distillation_loss_weights": torch.tensor([[0.4, 0.6]]),
        }
    )
    batch.metadata = {"response_length": 2}

    [experience] = list(TrainingBatchIterator(batch, sample_batch_size=1))

    assert isinstance(experience.distillation, StudentTopKPolicySurrogateInput)
    torch.testing.assert_close(experience.distillation.student_token_ids(), torch.tensor([[[1, 2], [3, 4]]]))
    torch.testing.assert_close(
        experience.distillation.teacher_on_student_logprobs,
        torch.log(torch.tensor([[[0.5, 0.2], [0.4, 0.3]]])),
    )


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
        distillation_input_from_tensors(**(base | {"teacher_action_log_probs": torch.tensor([[-0.5]])}))
    with pytest.raises(ValueError, match="selected IDs, behavior logprobs, and teacher scores together"):
        distillation_input_from_tensors(**(base | {"teacher_on_student_logprobs": None}))


def test_student_topk_surrogate_rejects_plausible_invalid_teacher_scores():
    distillation = StudentTopKPolicySurrogateInput(
        student_topk_indices=torch.tensor([[[1, 2], [-1, -1]]]),
        behavior_topk_logprobs=torch.tensor([[[-0.5, -0.8], [float("nan"), float("nan")]]]),
        teacher_on_student_logprobs=torch.tensor([[[-0.4, -0.9], [-0.3, -0.7]]]),
        valid_mask=torch.tensor([[True, False]]),
        loss_weights=torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="invalid teacher_on_student_logprobs must be NaN"):
        distillation.objective_loss(
            action_log_probs=torch.zeros(1, 2),
            old_action_log_probs=torch.zeros(1, 2),
            student_selected_logprobs=torch.tensor([[[-0.5, -0.8], [float("nan"), float("nan")]]]),
            loss_mask=torch.tensor([[True, False]]),
            config=_policy_config(),
        )


def test_partial_distillation_payload_fails_closed():
    with pytest.raises(ValueError, match="evidence requires valid_mask and loss_weights"):
        distillation_input_from_tensors(
            teacher_action_log_probs=torch.tensor([[-0.5]]),
            teacher_topk_indices=None,
            teacher_topk_logprobs=None,
            teacher_retained_mass=None,
            valid_mask=None,
            loss_weights=torch.ones(1, 1),
        )


def test_sampled_reverse_kl_rejects_plausible_invalid_scores_at_learner_boundary():
    actions = torch.full((1, 2), -1.0)
    distillation = SampledReverseKLInput(
        teacher_action_log_probs=torch.tensor([[-0.5, -0.7]]),
        valid_mask=torch.tensor([[True, False]]),
        loss_weights=torch.ones(1, 2),
    )

    with pytest.raises(ValueError, match="invalid teacher action logprobs must be NaN"):
        compute_policy_objective(
            action_log_probs=actions,
            old_action_log_probs=actions,
            base_action_log_probs=None,
            advantages=torch.zeros_like(actions),
            loss_mask=torch.ones_like(actions),
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(actions),
            config=_policy_config(),
            policy_loss_fn=ppo_policy_loss,
            accumulation_steps=1,
            scaling=LossScaling.CALLER,
            distillation=distillation,
        )
