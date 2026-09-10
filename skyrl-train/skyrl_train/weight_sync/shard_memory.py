"""Synchronized rank-local allocator evidence for untimed weight sync proofs."""

import torch


def device_memory(device):
    if device.type != "cuda":
        return {"cuda_measured": False}
    torch.cuda.synchronize(device)
    return {
        "cuda_measured": True,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_bytes": torch.cuda.mem_get_info(device)[0],
    }
