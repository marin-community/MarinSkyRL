"""Explicit ownership of optimizer gradients used by direction monitoring."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def fsdp_gradient_shards(parameters) -> list[torch.Tensor]:
    """Return each full-world FSDP2 gradient element on exactly one rank.

    Replicated mesh dimensions contribute only at coordinate zero. Submeshes,
    partial placements and distributed plain gradients need a separate coverage
    qualification; fail instead of silently counting those gradients incorrectly.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    gradients = []
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if not isinstance(grad, DTensor):
            if world_size != 1:
                raise ValueError("Gradient direction requires full-world FSDP2 DTensors for distributed FSDP")
            gradients.append(grad)
            continue
        mesh = grad.device_mesh
        coordinate = mesh.get_coordinate()
        if mesh.size() != world_size or coordinate is None:
            raise ValueError("Gradient direction has not qualified FSDP submesh ownership")
        if not all(isinstance(placement, (Shard, Replicate)) for placement in grad.placements):
            raise ValueError("Gradient direction requires fully reduced FSDP gradient placements")
        if any(
            isinstance(placement, Replicate) and coordinate[index] != 0
            for index, placement in enumerate(grad.placements)
        ):
            continue
        gradients.append(grad.to_local())
    return gradients
