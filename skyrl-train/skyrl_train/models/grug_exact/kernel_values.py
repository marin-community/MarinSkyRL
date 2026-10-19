"""Kernel values that carry the gradient of the trainer's own computation of the same values."""

from collections.abc import Callable

import torch


class _ValueWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Returned as is, autograd wraps the value in a view that carries this function's gradient: no copy.
        return value

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def kernel_value(value: torch.Tensor, reference: Callable[[], torch.Tensor]) -> torch.Tensor:
    """A kernel's ``value``, bits unchanged; with gradients enabled, differentiated as ``reference()``, which runs
    only then."""
    if not torch.is_grad_enabled():
        return value
    return _ValueWithGradient.apply(value, reference())
