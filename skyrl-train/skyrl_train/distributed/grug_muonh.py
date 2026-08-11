# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production Grug MuonH / AdamH / Adam optimizer for FSDP2.

This is the PyTorch form of Marin's ``grug_moe_muonh_v1`` recipe:

* MuonH for hidden attention, MoE/shared matrices, and GatedNorm matrices.
* AdamH for the output head.
* ``torch.optim.Adam`` for embeddings, routers, attention gates, biases,
  vectors, and one-dimensional norm weights.

MuonH and AdamH share one learning-rate track. Plain Adam has its own track.
Every group has zero weight decay, independent of the surrounding trainer
configuration.

The Newton--Schulz routine and its quintic coefficients are adapted from
NVIDIA NeMo Emerging-Optimizers (Apache-2.0), pinned at commit
``6ef41445b246d2c64c2e6f82cc56fbc9c9c07937``:
https://github.com/NVIDIA-NeMo/Emerging-Optimizers
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from skyrl_train.distributed.muon_hybrid import _CompositeOptimizer


_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_HYPERBALL_EPS = 1e-10
_DEFAULT_ADAM_LR = 6e-4
_DEFAULT_BETAS = (0.9, 0.95)
_DEFAULT_EPS = 1e-8
_DEFAULT_MOMENTUM = 0.95
_DEFAULT_MUON_STEPS = 5

type MuonRoute = Literal["muonh", "adamh", "adam"]


class _OptimizerConfig(Protocol):
    lr: float

    def get(self, key: str, default: object = None) -> object: ...


def newton_schulz_quintic(
    matrix: Tensor,
    *,
    steps: int = _DEFAULT_MUON_STEPS,
    eps: float = _DEFAULT_EPS,
) -> Tensor:
    """Return Marin's BF16 quintic direction for a matrix or expert stack."""
    if matrix.ndim not in (2, 3):
        raise ValueError(f"Newton--Schulz requires rank 2 or 3, got shape {tuple(matrix.shape)}")
    if steps < 1:
        raise ValueError(f"Newton--Schulz steps must be positive, got {steps}")

    original_dtype = matrix.dtype
    x = matrix.to(torch.bfloat16)
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + eps)

    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT

    for index in range(steps):
        a, b, c = _QUINTIC_COEFFICIENTS[index % len(_QUINTIC_COEFFICIENTS)]
        gram = x @ x.mT
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x

    if transposed:
        x = x.mT
    return x.to(original_dtype)


def _muon_shape_scale(matrix: Tensor) -> float:
    """Return Marin's Muon scale for a PyTorch matrix layout."""
    # Marin stores matrices as (fan_in, fan_out). PyTorch stores Linear weights
    # transposed as (fan_out, fan_in), so the equivalent scale is rows / columns.
    rows, columns = matrix.shape[-2:]
    return max(1.0, rows / columns) ** 0.5


def _muon_direction(direction: Tensor, *, steps: int, eps: float) -> Tensor:
    """Orthogonalize a matrix or an expert stack and apply Marin's shape scale."""
    if direction.ndim not in (2, 3):
        raise ValueError(f"MuonH parameters must have rank 2 or 3, got shape {tuple(direction.shape)}")
    orthogonal = newton_schulz_quintic(direction, steps=steps, eps=eps)
    return orthogonal.mul_(_muon_shape_scale(orthogonal))


def _matrix_norm(
    value: Tensor,
    reference: DTensor | None,
) -> Tensor:
    """Return the global norm of each trailing matrix in ``value``."""
    axes = (-2, -1)
    squared_norm = torch.linalg.vector_norm(value, dim=axes, keepdim=True, dtype=torch.float32).square_()
    if reference is not None:
        reduced_axes = {reference.ndim - 2, reference.ndim - 1}
        for mesh_dim, placement in enumerate(reference.placements):
            if isinstance(placement, Shard) and placement.dim % reference.ndim in reduced_axes:
                dist.all_reduce(squared_norm, group=reference.device_mesh.get_group(mesh_dim))
    return squared_norm.sqrt_()


def _hyperball_step_(
    parameter: Tensor,
    direction: Tensor,
    *,
    reference: DTensor | None,
    lr: float,
    clamp_final_norm: bool,
) -> None:
    """Apply HyperBall in place, reusing ``direction`` for the candidate."""
    parameter_norm = _matrix_norm(parameter, reference)
    direction_norm = _matrix_norm(direction, reference).clamp_min_(_HYPERBALL_EPS)
    direction.mul_(parameter_norm / direction_norm).mul_(-lr).add_(parameter)
    candidate_norm = _matrix_norm(direction, reference)
    if clamp_final_norm:
        candidate_norm.clamp_min_(_HYPERBALL_EPS)
    direction.mul_(parameter_norm / candidate_norm)
    parameter.copy_(direction)


def _is_expert_batch_sharded(tensor: DTensor) -> bool:
    """Whether a rank-3 DTensor is sharded only across its expert axis."""
    if tensor.ndim != 3 or not any(isinstance(placement, Shard) for placement in tensor.placements):
        return False
    return all(
        isinstance(placement, Replicate) or (isinstance(placement, Shard) and placement.dim == 0)
        for placement in tensor.placements
    )


def _move_to_mesh_device(tensor: Tensor) -> Tensor:
    """Move an offloaded DTensor shard to its mesh device without gathering it."""
    if not isinstance(tensor, DTensor):
        return tensor
    # Native FSDP2 CPU offload leaves local shards on CPU while retaining a
    # CUDA/NCCL mesh. Move the shard to that mesh before any collective.
    if tensor.device_mesh.device_type == "cuda" and tensor.to_local().device.type == "cpu":
        return tensor.to(torch.device("cuda", torch.cuda.current_device()))
    return tensor


def _gathered_muon_direction_shard(direction: DTensor, *, steps: int, eps: float) -> Tensor:
    """Gather a BF16 Muon direction and return only this rank's result shard."""
    full_direction = direction.to(torch.bfloat16).full_tensor()
    orthogonal = newton_schulz_quintic(full_direction, steps=steps, eps=eps)
    sharded = distribute_tensor(
        orthogonal,
        direction.device_mesh,
        direction.placements,
        src_data_rank=None,
    )
    local_direction = sharded.to_local().to(direction.dtype, copy=True)
    return local_direction.mul_(_muon_shape_scale(direction))


def _matrix_step_(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    ns_steps: int | None = None,
    muon_eps: float = _DEFAULT_EPS,
    clamp_final_norm: bool,
) -> None:
    """Update ``parameter`` in place, consuming ``direction`` as scratch storage."""
    parameter_value = _move_to_mesh_device(parameter)
    direction_value = _move_to_mesh_device(direction)
    parameter_is_dtensor = isinstance(parameter_value, DTensor)
    if parameter_is_dtensor != isinstance(direction_value, DTensor):
        raise RuntimeError("parameter and optimizer direction have incompatible tensor layouts")
    needs_local_muon_direction = ns_steps is not None
    if parameter_is_dtensor:
        assert isinstance(parameter_value, DTensor)
        assert isinstance(direction_value, DTensor)
        if (
            parameter_value.device_mesh != direction_value.device_mesh
            or parameter_value.placements != direction_value.placements
        ):
            raise RuntimeError("parameter and optimizer direction have incompatible DTensor layouts")
        local_parameter = parameter_value.to_local()
        if ns_steps is not None and not _is_expert_batch_sharded(direction_value):
            local_direction = _gathered_muon_direction_shard(direction_value, steps=ns_steps, eps=muon_eps)
            needs_local_muon_direction = False
        else:
            local_direction = direction_value.to_local()
        reference = parameter_value
    else:
        local_parameter = parameter_value
        local_direction = direction_value
        reference = None
    if needs_local_muon_direction:
        assert ns_steps is not None
        local_direction = _muon_direction(local_direction, steps=ns_steps, eps=muon_eps)

    _hyperball_step_(
        local_parameter,
        local_direction,
        reference=reference,
        lr=lr,
        clamp_final_norm=clamp_final_norm,
    )
    if isinstance(parameter, DTensor):
        parameter.to_local().copy_(local_parameter.to(parameter.to_local().device))


class MuonH(Optimizer):
    """Marin Muon momentum, quintic Newton--Schulz, and HyperBall update."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = _DEFAULT_MOMENTUM,
        nesterov: bool = True,
        ns_steps: int = _DEFAULT_MUON_STEPS,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "eps": eps,
            "weight_decay": 0.0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MuonH does not support sparse gradients")
                if parameter.ndim not in (2, 3):
                    raise RuntimeError(f"MuonH received a rank-{parameter.ndim} parameter")

                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(group["momentum"]).add_(gradient)
                direction = (
                    gradient.add(momentum_buffer, alpha=group["momentum"]) if group["nesterov"] else momentum_buffer
                )
                _matrix_step_(
                    parameter,
                    direction,
                    lr=group["lr"],
                    ns_steps=group["ns_steps"],
                    muon_eps=group["eps"],
                    clamp_final_norm=True,
                )
        return loss


class AdamH(Optimizer):
    """Marin Adam moments and bias correction followed by a HyperBall update."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        betas: tuple[float, float] = _DEFAULT_BETAS,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamH does not support sparse gradients")
                if parameter.ndim not in (2, 3):
                    raise RuntimeError(f"AdamH received a rank-{parameter.ndim} parameter")

                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
                    state["exp_avg"] = torch.zeros_like(gradient)
                    state["exp_avg_sq"] = torch.zeros_like(gradient)

                state["step"].add_(1)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                step = int(state["step"].item())
                bias_corrected_mean = exp_avg / (1 - beta1**step)
                bias_corrected_variance = exp_avg_sq / (1 - beta2**step)
                direction = bias_corrected_mean / (bias_corrected_variance.sqrt() + group["eps"])
                _matrix_step_(
                    parameter,
                    direction,
                    lr=group["lr"],
                    clamp_final_norm=False,
                )
        return loss


class GrugMuonH(_CompositeOptimizer):
    """One scheduler/checkpoint surface over MuonH, AdamH, and plain Adam."""

    def __init__(
        self,
        muonh_params: Iterable[Tensor],
        adamh_params: Iterable[Tensor],
        adam_params: Iterable[Tensor],
        *,
        lr: float,
        adam_lr: float = _DEFAULT_ADAM_LR,
        momentum: float = _DEFAULT_MOMENTUM,
        nesterov: bool = True,
        ns_steps: int = _DEFAULT_MUON_STEPS,
        betas: tuple[float, float] = _DEFAULT_BETAS,
        eps: float = _DEFAULT_EPS,
        muon_eps: float = _DEFAULT_EPS,
    ) -> None:
        muonh_params = list(muonh_params)
        adamh_params = list(adamh_params)
        adam_params = list(adam_params)
        if not muonh_params:
            raise ValueError("MuonH routing found no hidden rank-2 or rank-3 matrices")
        if not adamh_params:
            raise ValueError("MuonH routing found no output-head parameter for AdamH")

        self.muonh = MuonH(
            muonh_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=muon_eps,
        )
        self.adamh = AdamH(adamh_params, lr=lr, betas=betas, eps=eps)
        # This must remain the shipped optimizer. It is intentionally Adam, not
        # AdamW, and weight decay is therefore absent rather than inherited.
        self.adam = Adam(adam_params, lr=adam_lr, betas=betas, eps=eps) if adam_params else None

        children = [child for child in (self.muonh, self.adamh, self.adam) if child is not None]
        super().__init__(children, defaults={"lr": lr, "weight_decay": 0.0})

    def state_dict(self) -> dict[str, Any]:
        return {
            "muonh": self.muonh.state_dict(),
            "adamh": self.adamh.state_dict(),
            "adam": self.adam.state_dict() if self.adam is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muonh.load_state_dict(state_dict["muonh"])
        self.adamh.load_state_dict(state_dict["adamh"])
        adam_state = state_dict.get("adam")
        if (self.adam is None) != (adam_state is None):
            raise ValueError("checkpoint Adam state does not match the current MuonH parameter route")
        if self.adam is not None:
            assert adam_state is not None
            self.adam.load_state_dict(adam_state)
        self._refresh_composite_views()


def grug_muonh_route(name: str, parameter: Tensor) -> MuonRoute:
    """Return the pinned Marin route adapted to PyTorch parameter names."""
    lower_name = name.lower()
    # Check GatedNorm before the broader embedding and attention-gate markers
    # so ``embed_gated_norm`` and ``attn_gated_norm`` stay on Marin's route.
    if "gated_norm" in lower_name:
        return "muonh"
    if (
        "embed" in lower_name
        or "router_bias" in lower_name
        or "attn_gate" in lower_name
        or ".router" in lower_name
        or lower_name.startswith("router.")
    ):
        return "adam"
    if "output_proj" in lower_name or "lm_head" in lower_name:
        return "adamh"
    if parameter.ndim in (2, 3):
        return "muonh"
    return "adam"


def build_grug_muonh(
    named_parameters: Iterable[tuple[str, Tensor]],
    optim_config: _OptimizerConfig,
) -> GrugMuonH:
    """Classify named parameters and construct the exact three-group recipe."""
    parameters: dict[MuonRoute, list[Tensor]] = {"muonh": [], "adamh": [], "adam": []}
    names: dict[MuonRoute, list[str]] = {"muonh": [], "adamh": [], "adam": []}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        route = grug_muonh_route(name, parameter)
        parameters[route].append(parameter)
        names[route].append(name)

    raw_extra = optim_config.get("optimizer_kwargs", {})
    if not isinstance(raw_extra, Mapping):
        raise TypeError("MuonH optimizer_kwargs must be a mapping")
    extra = dict(raw_extra)
    known = {
        "adam_lr",
        "momentum",
        "nesterov",
        "backend_steps",
        "epsilon",
        "muon_epsilon",
    }
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ValueError(f"Unknown MuonH optimizer_kwargs: {unknown}")

    weight_decay = float(optim_config.get("weight_decay", 0.0))
    if weight_decay != 0.0:
        raise ValueError(f"MuonH requires weight_decay=0, got {weight_decay}")
    raw_betas = tuple(float(value) for value in optim_config.get("adam_betas", _DEFAULT_BETAS))
    if len(raw_betas) != 2:
        raise ValueError(f"MuonH adam_betas must contain two values, got {raw_betas}")
    betas = (raw_betas[0], raw_betas[1])

    optimizer = GrugMuonH(
        parameters["muonh"],
        parameters["adamh"],
        parameters["adam"],
        lr=float(optim_config.lr),
        adam_lr=float(extra.get("adam_lr", _DEFAULT_ADAM_LR)),
        momentum=float(extra.get("momentum", _DEFAULT_MOMENTUM)),
        nesterov=bool(extra.get("nesterov", True)),
        ns_steps=int(extra.get("backend_steps", _DEFAULT_MUON_STEPS)),
        betas=betas,
        eps=float(extra.get("epsilon", _DEFAULT_EPS)),
        muon_eps=float(extra.get("muon_epsilon", _DEFAULT_EPS)),
    )
    optimizer._muonh_param_names = names["muonh"]  # type: ignore[attr-defined]
    optimizer._adamh_param_names = names["adamh"]  # type: ignore[attr-defined]
    optimizer._adam_param_names = names["adam"]  # type: ignore[attr-defined]
    return optimizer


__all__ = [
    "AdamH",
    "GrugMuonH",
    "MuonH",
    "build_grug_muonh",
    "grug_muonh_route",
    "newton_schulz_quintic",
]
