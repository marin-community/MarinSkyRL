import torch
from torch import nn

from skyrl_train.distributed.fsdp_utils import _refresh_ep_gradient_scaling
from skyrl_train.models.ep_gradient import ExpertGradientAveraging


class _Experts(nn.Module, ExpertGradientAveraging):
    def __init__(self, ep_size: int) -> None:
        super().__init__()
        self.ep_size = ep_size
        self.weight = nn.Parameter(torch.ones(3))


def _backward(experts: _Experts) -> torch.Tensor:
    experts.zero_grad()
    experts.weight.sum().backward()
    return experts.weight.grad


def test_ep_expert_gradient_hook_averages_replicated_batch_contributions() -> None:
    experts = _Experts(ep_size=4)

    modules, parameters = _refresh_ep_gradient_scaling(experts)

    assert (modules, parameters) == (1, 1)
    torch.testing.assert_close(_backward(experts), torch.full((3,), 0.25))


def test_ep_expert_gradient_hook_refresh_does_not_stack_scaling() -> None:
    experts = _Experts(ep_size=4)
    _refresh_ep_gradient_scaling(experts)

    _refresh_ep_gradient_scaling(experts)

    torch.testing.assert_close(_backward(experts), torch.full((3,), 0.25))


def test_ep_expert_gradient_hook_refreshes_after_parameter_assignment() -> None:
    experts = _Experts(ep_size=4)
    _refresh_ep_gradient_scaling(experts)
    experts.weight = nn.Parameter(torch.ones(3))

    _refresh_ep_gradient_scaling(experts)

    torch.testing.assert_close(_backward(experts), torch.full((3,), 0.25))


def test_untagged_parameter_gradient_is_not_scaled() -> None:
    experts = nn.Linear(3, 1, bias=False)

    modules, parameters = _refresh_ep_gradient_scaling(experts)

    assert (modules, parameters) == (0, 0)
    experts.weight.sum().backward()
    torch.testing.assert_close(experts.weight.grad, torch.ones(1, 3))
