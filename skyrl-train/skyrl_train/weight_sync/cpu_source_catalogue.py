"""Gather replay-only metadata without NCCL object-serialization CUDA buffers."""

from datetime import timedelta

import torch


def gather_source_catalogue(local: dict) -> list[dict]:
    group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        rows = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(rows, local, group=group)
        return rows
    finally:
        torch.distributed.destroy_process_group(group)
