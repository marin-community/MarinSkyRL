"""AdamW parameter writes that retain sub-ulp updates in BF16 storage."""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.optim import AdamW, Optimizer


class BFloat16UpdateMode(StrEnum):
    """How AdamW writes an FP32 candidate value into parameter storage."""

    NEAREST = "nearest"
    STOCHASTIC = "stochastic"
    KAHAN = "kahan"
    FP32_MASTER = "fp32_master"


def parse_bf16_update_mode(value: str | BFloat16UpdateMode | None) -> BFloat16UpdateMode:
    """Return a validated update mode, defaulting to stochastic rounding."""

    if value is None:
        return BFloat16UpdateMode.STOCHASTIC
    try:
        return BFloat16UpdateMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in BFloat16UpdateMode)
        raise ValueError(f"bf16_update_mode must be one of {choices}, got {value!r}") from error


def _local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _parameter_shard_coordinate(parameter: Tensor) -> tuple[int, ...]:
    if isinstance(parameter, DTensor):
        coordinate = parameter.device_mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("the current rank is not part of the parameter DTensor mesh")
        return tuple(value for value, placement in zip(coordinate, parameter.placements) if not isinstance(placement, Replicate))
    if dist.is_available() and dist.is_initialized():
        return (dist.get_rank(),)
    return (0,)


def _mix_rounding_seed(base_seed: int, step: int, parameter_index: int, coordinate: tuple[int, ...]) -> int:
    """Mix stable optimizer and shard identities into a torch generator seed."""

    mask = (1 << 64) - 1
    mixed = base_seed & mask
    for value in (step, parameter_index, len(coordinate), *coordinate):
        mixed ^= (int(value) + 0x9E3779B97F4A7C15 + ((mixed << 6) & mask) + (mixed >> 2)) & mask
    return mixed


@torch.no_grad()
def copy_stochastic_bfloat16_(target: Tensor, source: Tensor, *, generator: torch.Generator) -> None:
    """Copy an FP32 tensor to BF16 with unbiased bitwise stochastic rounding.

    BF16 is the upper 16 bits of FP32. Adding a uniform 16-bit integer to the
    discarded bits before masking them implements stochastic rounding between
    the two adjacent BF16 values.
    """

    if target.dtype is not torch.bfloat16 or source.dtype is not torch.float32:
        raise TypeError(f"stochastic copy requires BF16 target and FP32 source, got {target.dtype}, {source.dtype}")
    if target.shape != source.shape or target.device != source.device:
        raise ValueError("stochastic copy requires matching target and source shapes and devices")

    source = source.contiguous()
    rounded_bits = torch.randint(
        0,
        1 << 16,
        source.shape,
        dtype=torch.int32,
        device=source.device,
        generator=generator,
    )
    rounded_bits.add_(source.view(torch.int32)).bitwise_and_(-65536)
    target.copy_(rounded_bits.view(torch.float32))


class BFloat16AdamW(Optimizer):
    """AdamW with stochastic or Kahan-compensated BF16 parameter writes.

    Moments retain PyTorch AdamW's existing parameter dtype. Only the candidate
    parameter value is formed in FP32. This follows the memory profile and
    update boundary of the AdamW-SR reference implementation.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        update_mode: BFloat16UpdateMode,
        seed: int,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if update_mode not in (BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN):
            raise ValueError(f"BFloat16AdamW does not implement {update_mode.value!r}")
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        unsupported = {
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": fused,
        }
        enabled = [name for name, value in unsupported.items() if value not in (None, False)]
        if enabled:
            raise ValueError(f"BFloat16AdamW does not support {', '.join(enabled)}")

        self._configured_mode = update_mode
        self._rounding_seed = int(seed)
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "bf16_update_mode": update_mode.value,
            "bf16_update_step": 0,
            "rounding_seed": self._rounding_seed,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _state_step(state: dict[str, Any]) -> int:
        value = state.get("step", 0)
        if isinstance(value, Tensor):
            return int(value.item())
        return int(value)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        parameter_index = 0
        for group in self.param_groups:
            group_step = int(group.get("bf16_update_step", 0)) + 1
            group["bf16_update_step"] = group_step
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                current_parameter_index = parameter_index
                parameter_index += 1
                if gradient is None:
                    continue

                parameter_local = _local_tensor(parameter)
                gradient_local = _local_tensor(gradient)
                if parameter_local.dtype is not torch.bfloat16:
                    raise TypeError(f"BFloat16AdamW requires BF16 parameters, got {parameter_local.dtype}")
                if gradient_local.is_sparse:
                    raise RuntimeError("BFloat16AdamW does not support sparse gradients")
                if gradient_local.dtype is not torch.bfloat16:
                    gradient_local = gradient_local.to(torch.bfloat16)
                if group["maximize"]:
                    gradient_local = -gradient_local

                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter_local, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter_local, memory_format=torch.preserve_format)
                if group["amsgrad"] and "max_exp_avg_sq" not in state:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        parameter_local, memory_format=torch.preserve_format
                    )
                if self._configured_mode is BFloat16UpdateMode.KAHAN and "rounding_residual" not in state:
                    state["rounding_residual"] = torch.zeros_like(
                        parameter_local, memory_format=torch.preserve_format
                    )

                step = self._state_step(state) + 1
                state["step"] = step
                exp_avg = _local_tensor(state["exp_avg"])
                exp_avg_sq = _local_tensor(state["exp_avg_sq"])
                exp_avg.mul_(beta1).add_(gradient_local, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient_local, gradient_local, value=1 - beta2)

                variance = exp_avg_sq
                if group["amsgrad"]:
                    max_exp_avg_sq = _local_tensor(state["max_exp_avg_sq"])
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    variance = max_exp_avg_sq

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                candidate = variance.float().sqrt_().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                candidate.reciprocal_().mul_(exp_avg).mul_(-group["lr"] / bias_correction1)
                candidate.add_(parameter_local, alpha=1 - group["lr"] * group["weight_decay"])

                if self._configured_mode is BFloat16UpdateMode.KAHAN:
                    residual = _local_tensor(state["rounding_residual"])
                    candidate.add_(residual.float())
                    rounded = candidate.to(torch.bfloat16)
                    residual.copy_((candidate - rounded.float()).to(torch.bfloat16))
                    parameter_local.copy_(rounded)
                    continue

                coordinate = _parameter_shard_coordinate(parameter)
                generator = torch.Generator(device=parameter_local.device)
                generator.manual_seed(
                    _mix_rounding_seed(
                        int(group.get("rounding_seed", self._rounding_seed)),
                        group_step,
                        current_parameter_index,
                        coordinate,
                    )
                )
                copy_stochastic_bfloat16_(parameter_local, candidate, generator=generator)

        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for group in state_dict.get("param_groups", ()):
            saved_mode = group.get("bf16_update_mode")
            if saved_mode is not None and parse_bf16_update_mode(saved_mode) is not self._configured_mode:
                raise ValueError(
                    f"checkpoint bf16_update_mode={saved_mode!r} does not match "
                    f"configured mode={self._configured_mode.value!r}"
                )

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group["bf16_update_mode"] = self._configured_mode.value
            group.setdefault("rounding_seed", self._rounding_seed)
            if "bf16_update_step" not in group:
                group["bf16_update_step"] = max(
                    (self._state_step(self.state[parameter]) for parameter in group["params"]),
                    default=0,
                )


def build_adamw(
    params: Iterable[Tensor],
    *,
    update_mode: str | BFloat16UpdateMode | None,
    seed: int,
    **kwargs: Any,
) -> Optimizer:
    """Build AdamW with the selected parameter-write behavior."""

    parameters = list(params)
    mode = parse_bf16_update_mode(update_mode)
    dtypes = {parameter.dtype for parameter in parameters}
    if mode in (BFloat16UpdateMode.STOCHASTIC, BFloat16UpdateMode.KAHAN) and torch.bfloat16 in dtypes:
        if dtypes != {torch.bfloat16}:
            raise ValueError(f"low-precision AdamW requires uniform BF16 parameters, got {sorted(map(str, dtypes))}")
        return BFloat16AdamW(parameters, update_mode=mode, seed=seed, **kwargs)
    if mode is BFloat16UpdateMode.FP32_MASTER and torch.bfloat16 in dtypes:
        raise ValueError("fp32_master requires FSDP parameter storage dtype float32")
    return AdamW(parameters, **kwargs)


__all__ = [
    "BFloat16AdamW",
    "BFloat16UpdateMode",
    "build_adamw",
    "copy_stochastic_bfloat16_",
    "parse_bf16_update_mode",
]
