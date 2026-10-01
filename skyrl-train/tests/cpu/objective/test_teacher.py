import math

import torch

from marinskyrl.distillation import DistillationObjectiveKind
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.distillation import TeacherTopKInput, StudentTopKInput
from skyrl_train.objective.teacher import topk_teacher_loss


def test_sparse_forward_teacher_equality_preserves_conditional_teacher_formula():
    teacher = torch.tensor([[[0.42, 0.28], [torch.nan, torch.nan]]]).log()
    student = teacher.clone().requires_grad_()
    evidence = TeacherTopKInput(
        torch.tensor([[[0, 1], [-1, -1]]]),
        teacher,
        torch.tensor([[0.7, torch.nan]]),
        torch.tensor([[True, False]]),
        torch.tensor([[3.0, torch.nan]]),
    )
    result = topk_teacher_loss(
        evidence, student, TopKLossParams(DistillationObjectiveKind.SPARSE_FORWARD_KL, 0.2, 0.2, 3)
    )
    torch.testing.assert_close(result.values, torch.tensor([[-math.log(0.7), 0]]))
    result.values.sum().backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[-0.6, -0.4], [0, 0]]]))


def test_student_selected_surrogate_matches_independent_clipped_values_and_gradient():
    current = torch.tensor([[[0.2, 0.5], [torch.nan, torch.nan]]]).log().requires_grad_()
    behavior = torch.tensor([[[0.1, 0.1], [torch.nan, torch.nan]]]).log()
    teacher = torch.tensor([[[0.4, 0.25], [torch.nan, torch.nan]]]).log()
    evidence = StudentTopKInput(
        torch.tensor([[[0, 1], [-1, -1]]]),
        behavior,
        teacher,
        torch.tensor([[True, False]]),
        torch.tensor([[7.0, torch.nan]]),
    )
    result = topk_teacher_loss(
        evidence, current, TopKLossParams(DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE, 0.2, 0.2, 3)
    )
    # Positive credit selects the upper PPO bound; negative credit selects the dual bound.
    positive = math.log(2) * 2 / 7
    negative = -math.log(2) * 5 / 7
    torch.testing.assert_close(result.values, torch.tensor([[-positive * 1.2 - negative * 3, 0]]))
    result.values.sum().backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    assert result.metrics["distillation_clip_fraction"] == 0.5
    assert result.metrics["distillation_dual_clip_fraction"] == 0.5
