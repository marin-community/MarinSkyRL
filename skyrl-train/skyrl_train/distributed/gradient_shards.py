"""Assign each optimizer gradient element to one rank for gradient-direction monitoring."""

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def fsdp_gradient_shards(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Return each full-world FSDP2 gradient element on exactly one rank.

    Replicated mesh dimensions contribute only at coordinate zero. Submeshes,
    partial placements and plain distributed gradients raise, because this
    function cannot prove each of their elements is counted once.
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
            raise ValueError("Gradient direction does not support FSDP submeshes")
        if not all(isinstance(placement, (Shard, Replicate)) for placement in grad.placements):
            raise ValueError("Gradient direction requires fully reduced FSDP gradient placements")
        if any(
            isinstance(placement, Replicate) and coordinate[index] != 0
            for index, placement in enumerate(grad.placements)
        ):
            continue
        gradients.append(grad.to_local())
    return gradients
