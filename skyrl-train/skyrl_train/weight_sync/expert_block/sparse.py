"""Exact indexed patches for routed experts in the expert-block transport."""

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.schedule import ExpertBroadcast

BUCKET_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SparseStats:
    matrices: int = 0
    logical_bytes: int = 0
    encoded_bytes: int = 0
    total_values: int = 0
    changed_values: int = 0
    buckets: int = 0
    collectives: int = 0
    detect_seconds: float = 0.0
    pack_seconds: float = 0.0
    transfer_seconds: float = 0.0
    apply_seconds: float = 0.0


def expert_buckets(items: tuple[ExpertBroadcast, ...]) -> list[tuple[ExpertBroadcast, ...]]:
    """Group adjacent experts by NCCL group and 128 MiB dense size."""
    buckets = []
    current = []
    current_bytes = 0
    for item in items:
        if current and (item.group != current[0].group or current_bytes + item.entry.nbytes > BUCKET_BYTES):
            buckets.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item.entry.nbytes
    if current:
        buckets.append(tuple(current))
    return buckets


def run_sparse_experts(stream, baseline: dict[str, torch.Tensor] | None) -> SparseStats:
    """Broadcast raw BF16 replacement values and flat int32 positions into bound views.

    Component times measure CPU submission. The enclosing stream measures the full expert
    phase with its existing CUDA synchronization, without a wait after every patch.
    """
    matrices = logical_bytes = encoded_bytes = total_values = changed_values = buckets = collectives = 0
    detect_seconds = pack_seconds = transfer_seconds = apply_seconds = 0.0
    for bucket in stream.expert_buckets:
        first = bucket[0]
        sender = first.root == stream.participant
        receiver = stream.participant in first.destinations
        if not sender and not receiver:
            continue
        local = [(item, None if sender else stream.expert_landing(item)) for item in bucket]
        sizes = [item.entry.nbytes // 2 for item, _ in local]
        matrices += len(local)
        logical_bytes += sum(item.entry.nbytes for item, _ in local)
        total_values += sum(sizes)
        group = stream.groups[first.group]

        if sender:
            if baseline is None:
                raise RuntimeError("Sparse expert sender has no acknowledged baseline")
            started = time.perf_counter()
            patches = []
            for item, _ in local:
                current = stream.source_view(item).view(-1)
                previous = baseline[item.entry.name]
                if (
                    current.dtype != torch.bfloat16
                    or current.shape != previous.shape
                    or current.device != previous.device
                ):
                    raise ValueError(f"Sparse baseline differs from {item.entry.name}")
                mask = current.view(torch.int16).ne(previous.view(torch.int16))
                positions = mask.nonzero(as_tuple=False).view(-1).to(torch.int32)
                patches.append((current, positions))
            detect_seconds += time.perf_counter() - started
            started = time.perf_counter()
            counts = [positions.numel() for _, positions in patches]
            metadata = torch.tensor(counts, dtype=torch.int64, device=stream.device)
            if any(counts):
                position_parts = [positions for _, positions in patches if positions.numel()]
                value_parts = [
                    current.index_select(0, positions.to(torch.int64))
                    for current, positions in patches
                    if positions.numel()
                ]
                positions = torch.cat(position_parts) if len(position_parts) > 1 else position_parts[0]
                values = torch.cat(value_parts) if len(value_parts) > 1 else value_parts[0]
            pack_seconds += time.perf_counter() - started
        else:
            metadata = torch.empty(len(local), dtype=torch.int64, device=stream.device)

        started = time.perf_counter()
        dist.broadcast(metadata, src=0, group=group)
        collectives += 1
        if not sender:
            counts = metadata.tolist()
        count = sum(counts)
        changed_values += count
        encoded_bytes += metadata.numel() * metadata.element_size()
        if count:
            if not sender:
                positions = torch.empty(count, dtype=torch.int32, device=stream.device)
                values = torch.empty(count, dtype=torch.bfloat16, device=stream.device)
            dist.broadcast(positions, src=0, group=group)
            dist.broadcast(values, src=0, group=group)
            collectives += 2
            encoded_bytes += count * (torch.int32.itemsize + torch.bfloat16.itemsize)
        transfer_seconds += time.perf_counter() - started

        if receiver and count:
            started = time.perf_counter()
            offset = 0
            for (_, landing), item_count in zip(local, counts, strict=True):
                if item_count:
                    landing.installed.view(-1).index_copy_(
                        0, positions.narrow(0, offset, item_count).to(torch.int64), values.narrow(0, offset, item_count)
                    )
                    offset += item_count
            apply_seconds += time.perf_counter() - started
        buckets += 1

    return SparseStats(
        matrices,
        logical_bytes,
        encoded_bytes,
        total_values,
        changed_values,
        buckets,
        collectives,
        detect_seconds,
        pack_seconds,
        transfer_seconds,
        apply_seconds,
    )
