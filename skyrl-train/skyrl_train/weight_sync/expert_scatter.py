"""Direct copies for qualified TP1, unquantized TRITON Grug receiver layouts."""

from typing import Sequence

import torch

from skyrl_train.weight_sync.manifest import ManifestEntry


def scatter_grug_experts(
    entry: ManifestEntry,
    source: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_map: Sequence[int],
    backend: str,
) -> int:
    """Install a whole-expert chunk without renumbering its global expert IDs.

    The caller must qualify the native backend and TP1 layout before selecting
    this path. All metadata is checked before any destination write. Returns the
    number of local experts copied; parameter storage and unrelated slots stay
    unchanged. This helper does not allocate transfer buffers or synchronize GPUs.
    """
    if backend != "TRITON":
        raise ValueError("Direct expert copies require the TRITON backend")
    projection = entry.hf_name.rsplit(".experts.", 1)[-1]
    if projection not in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        raise ValueError("Unsupported Grug expert projection")
    if len(entry.full_shape) != 3 or len(entry.shape) != 3 or entry.expert_start is None:
        raise ValueError("Expected a manifest entry with explicit expert offsets")
    first = entry.expert_start
    count = entry.shape[0]
    if (
        first < 0
        or count <= 0
        or first + count > entry.full_shape[0]
        or entry.shape[1:] != entry.full_shape[1:]
        or entry.tensor_offset != first * entry.shape[1] * entry.shape[2]
        or entry.numel != source.numel()
        or tuple(source.shape) != entry.shape
    ):
        raise ValueError("Inconsistent expert slice metadata")
    if len(expert_map) != entry.full_shape[0] or any(type(x) is not int or x < -1 for x in expert_map):
        raise ValueError("Invalid global-to-local expert map")
    local_ids = sorted(x for x in expert_map if x >= 0)
    if local_ids != list(range(len(local_ids))):
        raise ValueError("Local expert IDs must be unique and cover the receiver")
    intermediate, hidden = entry.shape[1:]
    if projection == "down_proj.weight":
        hidden, intermediate = entry.shape[1:]
    if tuple(w13.shape) != (len(local_ids), 2 * intermediate, hidden) or tuple(w2.shape) != (
        len(local_ids),
        hidden,
        intermediate,
    ):
        raise ValueError("Receiver shapes do not match the TP1 gate/up/down layout")
    if entry.wire_dtype not in ("bfloat16", "float32"):
        raise ValueError("Unsupported expert wire dtype")
    if any(
        tensor.dtype != getattr(torch, entry.wire_dtype) or tensor.device != source.device or not tensor.is_contiguous()
        for tensor in (source, w13, w2)
    ):
        raise ValueError("Expert dtype, device and contiguous layout must match")
    copied = 0
    with torch.no_grad():
        for chunk_index in range(count):
            local = expert_map[first + chunk_index]
            if local < 0:
                continue
            if projection == "down_proj.weight":
                destination = w2[local]
            elif projection == "gate_proj.weight":
                destination = w13[local, :intermediate]
            else:
                destination = w13[local, intermediate:]
            destination.copy_(source[chunk_index])
            copied += 1
    return copied
