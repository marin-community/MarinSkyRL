"""Consecutive gradient directions over explicitly disjoint optimizer shards."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
import torch.distributed as dist


CHUNK_ELEMENTS = 1 << 20

# What a worker reports before, or without, a gradient observation.
NO_GRADIENT_METRICS: Mapping[str, float] = MappingProxyType({})

GradientStore = Literal["gpu_fp32", "cpu_bf16"]
DEFAULT_GRADIENT_STORE: GradientStore = "gpu_fp32"


@dataclass(frozen=True)
class GradCosineSettings:
    """Resolved trainer.algorithm.grad_cosine knobs."""

    enabled: bool
    store: GradientStore


def grad_cosine_settings(algorithm_cfg) -> GradCosineSettings:
    """Read the grad-cosine section; an absent section leaves the measurement off."""
    section = algorithm_cfg.get("grad_cosine") or {}
    return GradCosineSettings(
        enabled=bool(section.get("enabled", False)),
        store=section.get("store", DEFAULT_GRADIENT_STORE),
    )


class GradientDirectionTracker:
    """Retain one local gradient shard and reduce its moments across policy ranks.

    The caller owns parameter coverage: a shared or replicated gradient must occur
    once across the reduction group. CPU bf16 storage approximates the previous
    direction; the current norm uses the original gradient in fp32. An invalid
    update clears the history on every rank.

    Args:
        store: Previous-gradient storage.
        device: Device for current gradients and the collective, including empty shards.
        world_group: Distributed group spanning the disjoint policy gradient shards.
        reduce_fn: Optional SUM collective returning the reduced moment vector.
    """

    def __init__(
        self,
        store: GradientStore,
        device: torch.device,
        world_group: dist.ProcessGroup | None = None,
        reduce_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        if store not in {"gpu_fp32", "cpu_bf16"}:
            raise ValueError(f"Unknown gradient storage: {store}")
        self.store = store
        self.device = device
        self.world_group = world_group
        self.reduce_fn = reduce_fn
        self.previous: list[torch.Tensor] | None = None

    def reset(self) -> None:
        """Clear history after a skipped step or a change in parameter ownership."""
        self.previous = None

    @torch.no_grad()
    def observe(self, grads: Iterable[torch.Tensor], *, successful: bool = True) -> dict[str, float]:
        """Measure one optimizer update before its gradients are cleared.

        All ranks must call this method, including unsuccessful updates and ranks
        with empty shards. A first, skipped, nonfinite or zero-norm comparison reports
        no cosine. A shape change starts a new history; a change of parameter
        identity or order needs an explicit reset.
        """
        gradients = list(grads)
        if any(grad.device != self.device or not grad.is_contiguous() for grad in gradients):
            raise ValueError("Gradient shards must be contiguous and on the collective device")
        history = self.previous
        same_layout = history is not None and [g.shape for g in gradients] == [p.shape for p in history]
        if not same_layout:
            previous_device = torch.device("cpu") if self.store == "cpu_bf16" else self.device
            previous_dtype = torch.bfloat16 if self.store == "cpu_bf16" else torch.float32
            self.previous = [
                torch.empty(
                    grad.shape,
                    device=previous_device,
                    dtype=previous_dtype,
                    pin_memory=self.store == "cpu_bf16" and self.device.type == "cuda",
                )
                for grad in gradients
            ]
        assert self.previous is not None
        # SUM carries invalidity flags alongside moments: a skipped/nonfinite rank
        # must not let its peers retain a different consecutive-update history.
        moments = torch.zeros(5, dtype=torch.float64, device=self.device)
        moments[3] = float(not successful)
        moments[4] = float(not same_layout)
        for grad, previous in zip(gradients, self.previous, strict=True):
            current_flat, previous_flat = grad.detach().view(-1), previous.view(-1)
            for offset in range(0, grad.numel(), CHUNK_ELEMENTS):
                current = current_flat[offset : offset + CHUNK_ELEMENTS].float()
                saved = previous_flat[offset : offset + CHUNK_ELEMENTS]
                moments[1] += torch.dot(current, current).double()
                if same_layout:
                    prior = saved.to(device=self.device, dtype=torch.float32)
                    moments[0] += torch.dot(current, prior).double()
                    moments[2] += torch.dot(prior, prior).double()
                # Synchronous copy makes the CPU snapshot ready for the next step
                # without retaining a full additional GPU staging allocation.
                saved.copy_(current)
        if self.reduce_fn is not None:
            moments = self.reduce_fn(moments)
        elif dist.is_initialized():
            dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=self.world_group)
        dot, current_squared, previous_squared, failed, missing_history = moments.tolist()
        finite = bool(torch.isfinite(moments[:3]).all().item())
        if failed or not finite:
            self.reset()
            return {
                "grad_cosine": 0.0,
                "grad_cosine_valid": 0.0,
                "grad_norm_reduced": 0.0,
                "grad_norm_valid": 0.0,
                "grad_dot": 0.0,
            }
        valid = not missing_history and current_squared > 0 and previous_squared > 0
        cosine = dot / (current_squared * previous_squared) ** 0.5 if valid else 0.0
        return {
            "grad_cosine": max(-1.0, min(1.0, cosine)),
            "grad_cosine_valid": float(valid),
            "grad_norm_reduced": current_squared**0.5,
            "grad_norm_valid": 1.0,
            "grad_dot": dot if valid else 0.0,
        }


def gradient_direction_summary(updates: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Range of the valid consecutive comparisons in one policy update call.

    A tracker's first comparison has no predecessor and is excluded, so a call with
    a single update reports nothing.
    """
    measured = [row["grad_cosine"] for row in updates if row.get("grad_cosine_valid") == 1.0]
    if not measured:
        return {}
    return {"grad_cosine_min": min(measured), "grad_cosine_max": max(measured)}
