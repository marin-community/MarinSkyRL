import math

import pytest
import torch

from marinskyrl.distillation import DistillationObjectiveKind
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.distillation import TeacherTopKInput, StudentTopKInput, student_topk_logprobs
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
        evidence, student, TopKLossParams(DistillationObjectiveKind.SPARSE_FORWARD_KL, 0.2, 0.2, 3), vocabulary_size=3
    )
    torch.testing.assert_close(result.values, torch.tensor([[-math.log(0.7), 0]]))
    result.values.sum().backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[-0.6, -0.4], [0, 0]]]))


@pytest.mark.parametrize("width", [2, 6])
@pytest.mark.parametrize("equal_distributions", [False, True])
@pytest.mark.parametrize("objective", ["sparse_forward_kl", "sparse_reverse_kl", "sparse_jsd"])
def test_teacher_divergence_matches_dense_value_and_gradient(width, equal_distributions, objective):
    teacher = torch.tensor([0.32, 0.24, 0.16, 0.12, 0.10, 0.06])
    initial = teacher.log() if equal_distributions else torch.tensor([0.2, -0.5, 0.7, -0.1, 0.3, -0.2])
    logits = torch.stack((initial, torch.zeros_like(initial))).unsqueeze(0).requires_grad_()
    reference_logits = initial.double().requires_grad_()
    indices = torch.arange(width).expand(1, 2, -1).clone()
    indices[:, 1] = -1
    selected = student_topk_logprobs(logits, indices)
    scores = torch.stack((teacher[:width].log(), torch.full((width,), torch.nan))).unsqueeze(0)
    evidence = TeacherTopKInput(
        indices,
        scores,
        torch.tensor([[teacher[:width].sum(), torch.nan]]),
        torch.tensor([[True, False]]),
        torch.tensor([[1.0, torch.nan]]),
    )
    beta = 0.3
    params = TopKLossParams(DistillationObjectiveKind(objective), 0.2, 0.2, 3, jsd_beta=beta)
    result = topk_teacher_loss(evidence, selected, params, vocabulary_size=logits.shape[-1])

    student = reference_logits.softmax(-1)
    target = teacher.double()
    if objective == "sparse_forward_kl":
        conditional = target[:width] / target[:width].sum()
        expected = (conditional * (conditional.log() - student[:width].log())).sum()
    else:
        if width < len(teacher):
            student = torch.cat((student[:width], student[width:].sum().unsqueeze(0)))
            target = torch.cat((target[:width], target[width:].sum().unsqueeze(0)))
        if objective == "sparse_reverse_kl":
            expected = (student * (student.log() - target.log())).sum()
        else:
            mixture = beta * target + (1 - beta) * student
            expected = beta * (target * (target.log() - mixture.log())).sum()
            expected += (1 - beta) * (student * (student.log() - mixture.log())).sum()
    torch.testing.assert_close(result.values, torch.tensor([[expected.item(), 0]]), rtol=1e-5, atol=2e-7)
    result.values.sum().backward()
    expected.backward()
    torch.testing.assert_close(logits.grad[0, 0].double(), reference_logits.grad, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros_like(initial), rtol=0, atol=0)
    assert result.values[0, 0] >= -2e-7


def test_forward_teacher_entry_clip_caps_only_positive_contributions():
    teacher = torch.tensor([[[0.6, 0.4]]]).log()
    student = torch.tensor([[[0.1, 0.8]]]).log().requires_grad_()
    evidence = TeacherTopKInput(
        torch.tensor([[[0, 1]]]),
        teacher,
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1),
    )
    params = TopKLossParams(DistillationObjectiveKind.SPARSE_FORWARD_KL, 0.2, 0.2, 3, entry_clip=0.1)
    result = topk_teacher_loss(evidence, student, params, vocabulary_size=3)
    torch.testing.assert_close(result.values, torch.tensor([[0.1 + 0.4 * math.log(0.5)]]))
    result.values.sum().backward()
    torch.testing.assert_close(student.grad, torch.tensor([[[0.0, -0.4]]]))


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
        evidence,
        current,
        TopKLossParams(DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE, 0.2, 0.2, 3),
        vocabulary_size=3,
    )
    # Positive credit selects the upper PPO bound; negative credit selects the dual bound.
    positive = math.log(2) * 2 / 7
    negative = -math.log(2) * 5 / 7
    torch.testing.assert_close(result.values, torch.tensor([[-positive * 1.2 - negative * 3, 0]]))
    result.values.sum().backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    assert result.metrics["distillation_clip_fraction"] == 0.5
    assert result.metrics["distillation_dual_clip_fraction"] == 0.5
