# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Grug MuonH, AdamH, and Adam recipe on Megatron's FP32 master parameters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

Route = Literal["muonh", "adamh", "adam"]
ROUTE_KEY = "grug_route"
MUONH_ROUTE: Route = "muonh"
ADAMH_ROUTE: Route = "adamh"
ADAM_ROUTE: Route = "adam"
DEFAULT_MOMENTUM = 0.95
DEFAULT_NESTEROV = True
DEFAULT_NS_STEPS = 5
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPSILON = 1e-8
_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_MATRIX_RANKS = (2, 3)


def grug_muonh_route(name: str, parameter: Tensor) -> Route:
    """Classify the Megatron parameter using Marin's three optimizer routes."""
    lower_name = name.lower()
    if "gated_norm" in lower_name:
        return MUONH_ROUTE
    if (
        "embed" in lower_name
        or "router_bias" in lower_name
        or "attn_gate" in lower_name
        or ".router" in lower_name
        or lower_name.startswith("router.")
    ):
        return ADAM_ROUTE
    if "output_proj" in lower_name or "lm_head" in lower_name or "output_layer" in lower_name:
        return ADAMH_ROUTE
    if parameter.ndim in _MATRIX_RANKS:
        return MUONH_ROUTE
    return ADAM_ROUTE


def _newton_schulz_quintic(matrix: Tensor, *, steps: int, eps: float) -> Tensor:
    x = matrix.to(torch.bfloat16)
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for index in range(steps):
        a, b, c = _QUINTIC_COEFFICIENTS[index % len(_QUINTIC_COEFFICIENTS)]
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.mT
    return x.to(matrix.dtype)


def _hyperball_step_(parameter: Tensor, direction: Tensor, *, lr: float, clamp_final_norm: bool) -> None:
    parameter_norm = torch.linalg.vector_norm(parameter, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    direction_norm = torch.linalg.vector_norm(direction, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    direction.mul_(parameter_norm / direction_norm.clamp_min_(1e-10)).mul_(-lr).add_(parameter)
    candidate_norm = torch.linalg.vector_norm(direction, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    if clamp_final_norm:
        candidate_norm.clamp_min_(1e-10)
    parameter.copy_(direction.mul_(parameter_norm / candidate_norm))


class GrugMegatronMuonH(Optimizer):
    """One checkpoint and scheduler surface for the three Grug update rules."""

    def __init__(
        self,
        params: Iterable[dict],
        *,
        lr: float,
        momentum: float = DEFAULT_MOMENTUM,
        nesterov: bool = DEFAULT_NESTEROV,
        ns_steps: int = DEFAULT_NS_STEPS,
        betas: tuple[float, float] = DEFAULT_BETAS,
        eps: float = DEFAULT_EPSILON,
        muon_eps: float = DEFAULT_EPSILON,
        adam_lr: float | None = None,
    ) -> None:
        if ns_steps < 1:
            raise ValueError("MuonH backend_steps must be positive")
        if adam_lr is not None and lr <= 0:
            raise ValueError("MuonH master lr must be positive when adam_lr is set")
        defaults = {"lr": lr, "weight_decay": 0.0, ROUTE_KEY: MUONH_ROUTE}
        super().__init__(params, defaults)
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.betas = betas
        self.eps = eps
        self.muon_eps = muon_eps
        self.adam_lr_mult = (adam_lr / lr) if adam_lr is not None else 1.0
        for group in self.param_groups:
            if group[ROUTE_KEY] == ADAM_ROUTE:
                group["lr_mult"] = self.adam_lr_mult
                if adam_lr is not None:
                    group["lr"] = adam_lr
            elif group[ROUTE_KEY] not in (MUONH_ROUTE, ADAMH_ROUTE):
                raise ValueError(f"Unknown Grug optimizer route: {group[ROUTE_KEY]}")
            if group.get("weight_decay", 0.0) != 0.0:
                raise ValueError("MuonH requires weight_decay=0 for every parameter group")

    def _initialize_parameter_state(self, parameter: Tensor, route: Route, *, prototype: Tensor | None = None) -> dict:
        state = self.state[parameter]
        if not state:
            value = parameter if prototype is None else prototype
            if route == MUONH_ROUTE:
                state["momentum_buffer"] = torch.zeros_like(value)
            else:
                state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
                state["exp_avg"] = torch.zeros_like(value)
                state["exp_avg_sq"] = torch.zeros_like(value)
        return state

    def initialize_state(self) -> None:
        """Populate all states before a distributed checkpoint is loaded."""
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize_parameter_state(parameter, group[ROUTE_KEY])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            route = group[ROUTE_KEY]
            lr = group["lr"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MuonH does not support sparse gradients")
                if route != ADAM_ROUTE and parameter.ndim not in _MATRIX_RANKS:
                    raise RuntimeError(f"{route} received a rank-{parameter.ndim} parameter")
                state = self._initialize_parameter_state(parameter, route, prototype=gradient)
                if route == MUONH_ROUTE:
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(self.momentum).add_(gradient)
                    direction = gradient.add(momentum_buffer, alpha=self.momentum) if self.nesterov else momentum_buffer
                    direction = _newton_schulz_quintic(direction, steps=self.ns_steps, eps=self.muon_eps)
                    rows, columns = direction.shape[-2:]
                    direction.mul_(max(1.0, rows / columns) ** 0.5)
                    _hyperball_step_(parameter, direction, lr=lr, clamp_final_norm=True)
                    continue
                beta1, beta2 = self.betas
                state["step"].add_(1)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                step = int(state["step"].item())
                bias_corrected_mean = exp_avg / (1 - beta1**step)
                bias_corrected_variance = exp_avg_sq / (1 - beta2**step)
                direction = bias_corrected_mean / (bias_corrected_variance.sqrt() + self.eps)
                if route == ADAMH_ROUTE:
                    _hyperball_step_(parameter, direction, lr=lr, clamp_final_norm=False)
                else:
                    parameter.add_(direction, alpha=-lr)
        return loss
