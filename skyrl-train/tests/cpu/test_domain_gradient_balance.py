"""Released Open-MOPD domain-share behavior on row-aligned teacher evidence."""

import torch
from skyrl_train.distillation import StudentTopKPolicySurrogateInput
from skyrl_train.domain_gradient_balance import DomainGradientBalancer

from marinskyrl.distillation import DomainGradientBalanceSpec


def _evidence(math_gap: float, code_gap: float, if_gap: float) -> StudentTopKPolicySurrogateInput:
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    behavior = torch.full((3, 4, 1), -3.0)
    teacher = torch.tensor(
        [
            [[-3 + math_gap]] * 4,
            [[-3 + code_gap]] * 4,
            [[-3 + if_gap]] * 4,
        ],
        dtype=torch.float32,
    )
    return StudentTopKPolicySurrogateInput(
        student_topk_indices=torch.zeros((3, 4, 1), dtype=torch.long).masked_fill(~mask.unsqueeze(-1), -1),
        behavior_topk_logprobs=behavior.masked_fill(~mask.unsqueeze(-1), torch.nan),
        teacher_on_student_logprobs=teacher.masked_fill(~mask.unsqueeze(-1), torch.nan),
        valid_mask=mask,
        loss_weights=mask.float(),
    )


def test_balancer_equalizes_token_shares_then_follows_gap_drift():
    balancer = DomainGradientBalancer(
        DomainGradientBalanceSpec(target_shares=(("math", 1.0), ("code", 1.0), ("if", 1.0)), gap_scale_alpha=1.0)
    )
    routes = ("math", "code", "if")

    first, _ = balancer.apply(_evidence(1.0, 2.0, 2.5), routes)
    torch.testing.assert_close(first.loss_weights[:, 0], torch.tensor([7 / 12, 7 / 6, 7 / 3]))
    assert balancer.state_dict() == {"anchor": {"math": 1.0, "code": 2.0, "if": 2.5}}

    second, _ = balancer.apply(_evidence(2.0, 1.0, 2.5), routes)
    torch.testing.assert_close(second.loss_weights[:, 0], torch.tensor([1.0, 0.5, 2.0]))
    assert not second.loss_weights[1, 2:].any()
    assert not second.loss_weights[2, 1:].any()


def test_balancer_restored_anchor_keeps_continuous_weights():
    spec = DomainGradientBalanceSpec(target_shares=(("math", 1.0), ("code", 1.0), ("if", 1.0)), gap_scale_alpha=1.0)
    uninterrupted = DomainGradientBalancer(spec)
    uninterrupted.apply(_evidence(1.0, 2.0, 2.5), ("math", "code", "if"))

    resumed = DomainGradientBalancer(spec)
    resumed.load_state_dict(uninterrupted.state_dict())
    expected, _ = uninterrupted.apply(_evidence(2.0, 1.0, 2.5), ("math", "code", "if"))
    actual, _ = resumed.apply(_evidence(2.0, 1.0, 2.5), ("math", "code", "if"))

    torch.testing.assert_close(actual.loss_weights, expected.loss_weights)
