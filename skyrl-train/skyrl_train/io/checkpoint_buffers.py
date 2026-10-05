"""Reuse checkpoint destinations when loaded shards are adjacent views."""

from collections.abc import Callable

import torch


def merge_adjacent_checkpoint_shards(
    shards: list[torch.Tensor], fallback: Callable[[list[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Join contiguous views without allocating another model or optimizer buffer.

    DCP reads directly into the gate/up views created by Megatron's SwiGLU
    factory. Concatenating them would keep both the destination and a full copy
    alive until optimizer restore finishes. Other shard layouts use the
    factory's original merge.
    """
    if not shards or any(type(shard) is not torch.Tensor or shard.numel() == 0 for shard in shards):
        return fallback(shards)
    first = shards[0]
    if first.ndim == 0 or not first.is_contiguous():
        return fallback(shards)
    storage_pointer = first.untyped_storage().data_ptr()
    offset = first.storage_offset()
    rows = 0
    for shard in shards:
        if (
            shard.device != first.device
            or shard.dtype != first.dtype
            or not shard.is_contiguous()
            or shard.shape[1:] != first.shape[1:]
            or shard.stride() != first.stride()
            or shard.untyped_storage().data_ptr() != storage_pointer
            or shard.storage_offset() != offset
        ):
            return fallback(shards)
        offset += shard.numel()
        rows += shard.shape[0]
    return first.as_strided((rows, *first.shape[1:]), first.stride())
