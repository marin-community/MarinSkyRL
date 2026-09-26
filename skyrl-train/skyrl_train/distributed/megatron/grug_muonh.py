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
LAYOUT_KEY = "grug_layout"
QKV_LAYOUT = "qkv"
GATE_UP_LAYOUT = "gate_up"
MUONH_ROUTE: Route = "muonh"
ADAMH_ROUTE: Route = "adamh"
ADAM_ROUTE: Route = "adam"
DEFAULT_MOMENTUM = 0.95
DEFAULT_NESTEROV = True
DEFAULT_NS_STEPS = 5
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPSILON = 1e-8
_NORM_FLOOR = 1e-10
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
    if "gated_norm" in lower_name or (
        "embed_norm." in lower_name and lower_name.endswith(("down_proj.weight", "up_proj.weight"))
    ):
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
    direction.mul_(parameter_norm / direction_norm.clamp_min_(_NORM_FLOOR)).mul_(-lr).add_(parameter)
    candidate_norm = torch.linalg.vector_norm(direction, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    if clamp_final_norm:
        candidate_norm.clamp_min_(_NORM_FLOOR)
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
        min_lr: float = 0.0,
        qkv_num_query_groups: int | None = None,
        qkv_heads_per_group: int | None = None,
        qkv_head_dim: int | None = None,
        tensor_model_parallel_size: int = 1,
        expert_tensor_parallel_size: int = 1,
    ) -> None:
        if tensor_model_parallel_size != 1:
            raise ValueError("Grug MuonH requires tensor_model_parallel_size=1")
        if expert_tensor_parallel_size != 1:
            raise ValueError("Grug MuonH requires expert_tensor_parallel_size=1")
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
        self.qkv_num_query_groups = qkv_num_query_groups
        self.qkv_heads_per_group = qkv_heads_per_group
        self.qkv_head_dim = qkv_head_dim
        for group in self.param_groups:
            if group[ROUTE_KEY] == ADAM_ROUTE:
                group["lr_mult"] = self.adam_lr_mult
                if adam_lr is not None:
                    group["lr"] = adam_lr
                    group["max_lr"] = adam_lr
                    group["min_lr"] = min_lr * self.adam_lr_mult
            elif group[ROUTE_KEY] not in (MUONH_ROUTE, ADAMH_ROUTE):
                raise ValueError(f"Unknown Grug optimizer route: {group[ROUTE_KEY]}")
            layout = group.get(LAYOUT_KEY)
            if layout not in (None, QKV_LAYOUT, GATE_UP_LAYOUT):
                raise ValueError(f"Unknown Grug fused layout: {layout}")
            if layout == QKV_LAYOUT and not all(
                value is not None for value in (qkv_num_query_groups, qkv_heads_per_group, qkv_head_dim)
            ):
                raise ValueError("Fused QKV MuonH requires attention group geometry")
            if layout is not None and group[ROUTE_KEY] != MUONH_ROUTE:
                raise ValueError("Grug fused layouts require the MuonH route")
            if group.get("weight_decay", 0.0) != 0.0:
                raise ValueError("MuonH requires weight_decay=0 for every parameter group")

    def _muonh_matrix_step_(self, parameter: Tensor, gradient: Tensor, momentum_buffer: Tensor, lr: float) -> None:
        momentum_buffer.mul_(self.momentum).add_(gradient)
        direction = gradient.add(momentum_buffer, alpha=self.momentum) if self.nesterov else momentum_buffer
        direction = _newton_schulz_quintic(direction, steps=self.ns_steps, eps=self.muon_eps)
        rows, columns = direction.shape[-2:]
        direction.mul_(max(1.0, rows / columns) ** 0.5)
        _hyperball_step_(parameter, direction, lr=lr, clamp_final_norm=True)

    def _qkv_row_indices(self, parameter: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        num_groups = self.qkv_num_query_groups
        heads_per_group = self.qkv_heads_per_group
        head_dim = self.qkv_head_dim
        if num_groups is None or heads_per_group is None or head_dim is None:
            raise ValueError("Fused QKV MuonH requires attention group geometry")
        rows_per_group = (heads_per_group + 2) * head_dim
        if parameter.ndim != 2 or parameter.shape[0] != num_groups * rows_per_group:
            raise ValueError(f"Fused QKV parameter has incompatible shape {tuple(parameter.shape)}")
        offsets = torch.arange(num_groups, device=parameter.device)[:, None] * rows_per_group
        q = (offsets + torch.arange(heads_per_group * head_dim, device=parameter.device)).reshape(-1)
        k = (offsets + heads_per_group * head_dim + torch.arange(head_dim, device=parameter.device)).reshape(-1)
        v = (offsets + (heads_per_group + 1) * head_dim + torch.arange(head_dim, device=parameter.device)).reshape(-1)
        return q, k, v

    def _fused_muonh_step_(
        self, parameter: Tensor, gradient: Tensor, momentum_buffer: Tensor, lr: float, layout: str
    ) -> None:
        if layout == GATE_UP_LAYOUT:
            if parameter.shape[-2] % 2:
                raise ValueError(f"Fused gate/up parameter has incompatible shape {tuple(parameter.shape)}")
            for part, grad_part, momentum_part in zip(
                parameter.chunk(2, dim=-2), gradient.chunk(2, dim=-2), momentum_buffer.chunk(2, dim=-2)
            ):
                self._muonh_matrix_step_(part, grad_part, momentum_part, lr)
            return
        for indices in self._qkv_row_indices(parameter):
            part = parameter.index_select(0, indices)
            momentum_part = momentum_buffer.index_select(0, indices)
            self._muonh_matrix_step_(part, gradient.index_select(0, indices), momentum_part, lr)
            parameter.index_copy_(0, indices, part)
            momentum_buffer.index_copy_(0, indices, momentum_part)

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
                    layout = group.get(LAYOUT_KEY)
                    if layout is None:
                        self._muonh_matrix_step_(parameter, gradient, momentum_buffer, lr)
                    else:
                        self._fused_muonh_step_(parameter, gradient, momentum_buffer, lr, layout)
                    if "step" in state:
                        state["step"].add_(1)
                    continue
                beta1, beta2 = self.betas
                state["step"].add_(1)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                step = state["step"]
                bias_corrected_mean = exp_avg / (1 - torch.pow(beta1, step))
                bias_corrected_variance = exp_avg_sq / (1 - torch.pow(beta2, step))
                direction = bias_corrected_mean / (bias_corrected_variance.sqrt() + self.eps)
                if route == ADAMH_ROUTE:
                    _hyperball_step_(parameter, direction, lr=lr, clamp_final_norm=False)
                else:
                    parameter.add_(direction, alpha=-lr)
        return loss
