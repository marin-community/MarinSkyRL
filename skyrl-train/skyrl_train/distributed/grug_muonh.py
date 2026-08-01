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

from itertools import chain
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from skyrl_train.distributed.muon_hybrid import _MergedState


_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_HYPERBALL_EPS = 1e-10


def newton_schulz_quintic(matrix: Tensor, *, steps: int = 5, eps: float = 1e-8) -> Tensor:
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


def _muon_direction(direction: Tensor, *, steps: int, eps: float) -> Tensor:
    """Orthogonalize a matrix or an expert stack and apply Marin's shape scale."""
    if direction.ndim not in (2, 3):
        raise ValueError(f"MuonH parameters must have rank 2 or 3, got shape {tuple(direction.shape)}")
    orthogonal = newton_schulz_quintic(direction, steps=steps, eps=eps)

    # Marin stores matrices as (fan_in, fan_out). PyTorch stores Linear weights
    # transposed as (fan_out, fan_in), so the equivalent scale is rows / columns.
    rows, columns = orthogonal.shape[-2:]
    return orthogonal * max(1.0, rows / columns) ** 0.5


def _hyperball_delta(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    clamp_final_norm: bool,
) -> Tensor:
    """Return the norm-preserving HyperBall delta over trailing matrix axes."""
    axes = (-2, -1)
    parameter_norm = torch.linalg.vector_norm(parameter, dim=axes, keepdim=True)
    direction_norm = torch.linalg.vector_norm(direction, dim=axes, keepdim=True)
    candidate = parameter - lr * direction * parameter_norm / direction_norm.clamp_min(_HYPERBALL_EPS)
    candidate_norm = torch.linalg.vector_norm(candidate, dim=axes, keepdim=True)
    if clamp_final_norm:
        candidate_norm = candidate_norm.clamp_min(_HYPERBALL_EPS)
    return candidate / candidate_norm * parameter_norm - parameter


def _is_expert_batch_sharded(tensor: DTensor) -> bool:
    """Whether a rank-3 DTensor is sharded only across its expert axis."""
    if tensor.ndim != 3 or not any(isinstance(placement, Shard) for placement in tensor.placements):
        return False
    return all(
        isinstance(placement, Replicate) or (isinstance(placement, Shard) and placement.dim == 0)
        for placement in tensor.placements
    )


def _materialize_for_matrix_math(tensor: Tensor) -> tuple[Tensor, bool]:
    """Return a regular tensor and whether it is a local expert-stack shard."""
    if not isinstance(tensor, DTensor):
        return tensor, False
    # Native FSDP2 CPU offload leaves local shards on CPU while retaining a
    # CUDA/NCCL mesh. Move the shard to that mesh before any collective.
    if tensor.device_mesh.device_type == "cuda" and tensor.to_local().device.type == "cpu":
        tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))
    if _is_expert_batch_sharded(tensor):
        return tensor.to_local(), True
    return tensor.full_tensor(), False


def _redistribute_like(value: Tensor, reference: Tensor, *, local_expert_shard: bool) -> Tensor:
    """Restore ``value`` to the DTensor layout of ``reference`` when needed."""
    if not isinstance(reference, DTensor):
        return value
    if local_expert_shard:
        restored = DTensor.from_local(
            value,
            reference.device_mesh,
            reference.placements,
            run_check=False,
            shape=reference.shape,
            stride=reference.stride(),
        )
    else:
        # Every rank already reconstructed the same full value. ``src_data_rank=None``
        # avoids a redundant broadcast and selects each rank's local shard.
        restored = distribute_tensor(
            value,
            reference.device_mesh,
            reference.placements,
            src_data_rank=None,
        )
    return restored.to(reference.to_local().device)


def _matrix_delta(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    muon_steps: int | None = None,
    muon_eps: float = 1e-8,
    clamp_final_norm: bool,
) -> Tensor:
    """Compute a HyperBall delta with global dense or local expert semantics."""
    parameter_value, parameter_is_local_experts = _materialize_for_matrix_math(parameter)
    direction_value, direction_is_local_experts = _materialize_for_matrix_math(direction)
    if parameter_is_local_experts != direction_is_local_experts:
        raise RuntimeError("parameter and optimizer direction have incompatible DTensor layouts")
    if muon_steps is not None:
        direction_value = _muon_direction(direction_value, steps=muon_steps, eps=muon_eps)
    delta = _hyperball_delta(
        parameter_value,
        direction_value,
        lr=lr,
        clamp_final_norm=clamp_final_norm,
    )
    return _redistribute_like(delta, parameter, local_expert_shard=parameter_is_local_experts)


class MuonH(Optimizer):
    """Marin Muon momentum, quintic Newton--Schulz, and HyperBall update."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
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
                delta = _matrix_delta(
                    parameter,
                    direction,
                    lr=group["lr"],
                    muon_steps=group["ns_steps"],
                    muon_eps=group["eps"],
                    clamp_final_norm=True,
                )
                parameter.add_(delta)
        return loss


class AdamH(Optimizer):
    """Marin Adam moments and bias correction followed by a HyperBall update."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
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
                delta = _matrix_delta(
                    parameter,
                    direction,
                    lr=group["lr"],
                    clamp_final_norm=False,
                )
                parameter.add_(delta)
        return loss


class GrugMuonH(Optimizer):
    """One scheduler/checkpoint surface over MuonH, AdamH, and plain Adam."""

    def __init__(
        self,
        muonh_params: Iterable[Tensor],
        adamh_params: Iterable[Tensor],
        adam_params: Iterable[Tensor],
        *,
        lr: float,
        adam_lr: float = 6e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        muon_eps: float = 1e-8,
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
        self.adamh = AdamH(adamh_params, lr=lr, betas=(beta1, beta2), eps=eps)
        # This must remain the shipped optimizer. It is intentionally Adam, not
        # AdamW, and weight decay is therefore absent rather than inherited.
        self.adam = Adam(adam_params, lr=adam_lr, betas=(beta1, beta2), eps=eps) if adam_params else None

        all_params = list(chain(muonh_params, adamh_params, adam_params))
        super().__init__(all_params, defaults={"lr": lr, "weight_decay": 0.0})
        self.param_groups = list(chain.from_iterable(child.param_groups for child in self._children))
        self.state = _MergedState(self._children)

    @property
    def _children(self) -> list[Optimizer]:
        return [child for child in (self.muonh, self.adamh, self.adam) if child is not None]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for child in self._children:
            child.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for child in self._children:
            child.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muonh": self.muonh.state_dict(),
            "adamh": self.adamh.state_dict(),
            "adam": self.adam.state_dict() if self.adam is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muonh.load_state_dict(state_dict["muonh"])
        self.adamh.load_state_dict(state_dict["adamh"])
        if self.adam is not None and state_dict.get("adam") is not None:
            self.adam.load_state_dict(state_dict["adam"])
        # Optimizer.load_state_dict replaces each child's param-group mapping. Refresh
        # the composite view so a restored scheduler mutates the live child groups.
        self.param_groups = list(chain.from_iterable(child.param_groups for child in self._children))
        self.state = _MergedState(self._children)


def grug_muonh_route(name: str, parameter: Tensor) -> str:
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


def build_grug_muonh(named_parameters, optim_config) -> GrugMuonH:
    """Classify named parameters and construct the exact three-group recipe."""
    parameters: dict[str, list[Tensor]] = {"muonh": [], "adamh": [], "adam": []}
    names: dict[str, list[str]] = {"muonh": [], "adamh": [], "adam": []}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        route = grug_muonh_route(name, parameter)
        parameters[route].append(parameter)
        names[route].append(name)

    extra = dict(optim_config.get("optimizer_kwargs", {}) or {})
    known = {
        "adam_lr",
        "momentum",
        "nesterov",
        "backend_steps",
        "beta1",
        "beta2",
        "epsilon",
        "muon_epsilon",
        # Weight decay is recognized only so a global-style mapping cannot leak
        # into any child. The value is deliberately ignored.
        "weight_decay",
        "adam_weight_decay",
        "muon_weight_decay",
    }
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ValueError(f"Unknown MuonH optimizer_kwargs: {unknown}")

    optimizer = GrugMuonH(
        parameters["muonh"],
        parameters["adamh"],
        parameters["adam"],
        lr=float(optim_config.lr),
        adam_lr=float(extra.get("adam_lr", 6e-4)),
        momentum=float(extra.get("momentum", 0.95)),
        nesterov=bool(extra.get("nesterov", True)),
        ns_steps=int(extra.get("backend_steps", 5)),
        beta1=float(extra.get("beta1", 0.9)),
        beta2=float(extra.get("beta2", 0.95)),
        eps=float(extra.get("epsilon", 1e-8)),
        muon_eps=float(extra.get("muon_epsilon", 1e-8)),
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
