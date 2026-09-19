"""Disposable per-transfer sparse transport over the existing routed schedule.

This keeps #689's groups, source views and destination views. Each sparse transfer sends a
GPU count, then positions and replacement values through the same group as its dense
broadcast. Dense slices retain the node-local fanout. The many count and payload calls
are intentional: this is the per-tensor reference for a later bucket comparison.
"""

import os
import resource
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.schedule import ExpertBroadcast
from skyrl_train.weight_sync.expert_block.source_views import dense_source_view, expert_source_view
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import (
    Patch,
    changed_mask,
    pack_bitmap,
)
from skyrl_train.weight_sync.expert_block.sparse_experiment_codec import (
    apply as apply_patch,
)
from skyrl_train.weight_sync.expert_block.stream import Stream


@dataclass(frozen=True)
class SparseReport:
    participant: int
    version: int
    encoding: str
    transfers: int
    changed_values: int
    total_values: int
    logical_bytes: int
    metadata_bytes: int
    collectives: int
    detect_seconds: float
    construct_seconds: float
    pack_allocation_seconds: float
    transfer_seconds: float
    apply_seconds: float
    seconds: float
    gpu_allocated_start: int
    gpu_peak_allocated: int
    gpu_free_start: int
    gpu_free_end: int
    host_rss_start: int
    host_rss_end: int
    host_peak_rss: int


@dataclass(frozen=True)
class DistributionRow:
    participant: int
    version: int
    name: str
    family: str
    expert: int | None
    source_key: str
    dtype: str
    values: int
    changed: int
    adjacent_changed_pairs: int
    occupied_256_value_blocks: int
    total_256_value_blocks: int


def _family(item) -> str:
    if isinstance(item, ExpertBroadcast):
        return f"expert_{item.entry.projection}"
    name = item.source.hf_name
    if ".self_attn." in name:
        return "attention"
    if ".router." in name:
        return "router"
    if ".shared_expert." in name:
        return "shared_expert"
    if "norm" in name:
        return "norm"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    return "other_dense"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rss() -> int:
    with open("/proc/self/statm", encoding="ascii") as source:
        return int(source.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _free(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0


def _allocated(device: torch.device) -> int:
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else 0


def memory_sample(stream: Stream, reset_peak: bool = False) -> dict:
    """Record a participant's memory outside the timed publication interval."""
    device = stream.device
    _sync(device)
    if reset_peak and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    host_rss = _rss()
    return {
        "participant": stream.participant,
        "gpu_allocated_bytes": _allocated(device),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "gpu_free_bytes": _free(device),
        "host_rss_bytes": host_rss,
        "host_peak_rss_bytes": max(host_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _baseline_view(stream: Stream, item, baseline: dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(item, ExpertBroadcast):
        return expert_source_view(stream.expert_sources[item.entry.name], baseline)
    return dense_source_view(item.source, baseline)


@torch.no_grad()
def measure_distribution(stream: Stream, version: int, baseline: dict[str, torch.Tensor]) -> list[dict]:
    """Measure every routed source view before the baseline advances."""
    if not stream.trainer:
        raise ValueError("Only sender ranks own the sparse baseline")
    rows = []
    for item, landing in stream.transfers():
        if landing is not None:
            continue
        current = stream.source_view(item)
        mask = changed_mask(current, _baseline_view(stream, item, baseline))
        n = mask.numel()
        full_blocks, remainder = divmod(n, 256)
        blocks = mask[: full_blocks * 256].view(full_blocks, 256).any(dim=1).sum()
        if remainder:
            blocks += mask[full_blocks * 256 :].any()
        source_key = (
            stream.expert_sources[item.entry.name].source_key
            if isinstance(item, ExpertBroadcast)
            else item.source.source_key
        )
        rows.append(
            asdict(
                DistributionRow(
                    participant=stream.participant,
                    version=version,
                    name=item.entry.name if isinstance(item, ExpertBroadcast) else item.source.hf_name,
                    family=_family(item),
                    expert=item.entry.expert if isinstance(item, ExpertBroadcast) else None,
                    source_key=source_key,
                    dtype=str(current.dtype).removeprefix("torch."),
                    values=n,
                    changed=int(mask.sum().item()),
                    adjacent_changed_pairs=int((mask[:-1] & mask[1:]).sum().item()),
                    occupied_256_value_blocks=int(blocks.item()),
                    total_256_value_blocks=full_blocks + int(bool(remainder)),
                )
            )
        )
    return rows


def _relay(stream: Stream, item, tensor: torch.Tensor) -> int:
    """One expert-group call, or the dense root call plus receiver-local fanout."""
    if isinstance(item, ExpertBroadcast):
        dist.broadcast(tensor, src=0, group=stream.groups[item.group])
        return 1
    count = 0
    if stream.trainer or stream.participant in item.landings:
        dist.broadcast(tensor, src=0, group=stream.groups[item.group])
        count += 1
    if not stream.trainer:
        local_name, members = stream.local
        origin = next(rank for rank in item.landings if rank in members)
        dist.broadcast(tensor, src=members.index(origin), group=stream.groups[local_name])
        count += 1
    return count


def _encode_timed(
    current: torch.Tensor, previous: torch.Tensor, encoding: str, device: torch.device, *, profile: bool = True
):
    started = time.perf_counter()
    mask = changed_mask(current, previous)
    if profile:
        _sync(device)
    detect = time.perf_counter() - started

    started = time.perf_counter()
    if encoding == "indices":
        if current.numel() >= 2**31:
            raise ValueError("A single indexed transfer must fit signed int32 positions")
        positions = mask.nonzero(as_tuple=False).view(-1).to(torch.int32)
    elif encoding == "bitmap":
        positions = pack_bitmap(mask)
    else:
        raise ValueError(f"Unknown exact sparse encoding {encoding}")
    if profile:
        _sync(device)
    construct = time.perf_counter() - started

    started = time.perf_counter()
    flat = current.detach().view(-1)
    values = flat.index_select(0, positions.to(torch.int64)) if encoding == "indices" else flat.masked_select(mask)
    if profile:
        _sync(device)
    pack_allocation = time.perf_counter() - started
    return Patch(encoding, flat.numel(), positions, values), detect, construct, pack_allocation


@torch.no_grad()
def run_sparse(
    stream: Stream,
    version: int,
    encoding: str,
    baseline: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Run one candidate into the bound views and return participant-level components."""
    if encoding in {"dense_expert_bucket_fast", "dense_expert_bucket_profiled"}:
        return run_dense_expert_bucket(stream, version, profile=encoding.endswith("profiled"))
    if encoding == "indices_expert_bucket_profiled":
        return run_sparse_bucket(stream, version, "indices", baseline, dense_fallback=True)
    if encoding in {"indices_bucket_fast", "indices_expert_bucket_fast"}:
        return run_sparse_bucket(
            stream,
            version,
            "indices",
            baseline,
            profile=False,
            dense_fallback=encoding == "indices_expert_bucket_fast",
        )
    if encoding.endswith("_bucket"):
        return run_sparse_bucket(stream, version, encoding.removesuffix("_bucket"), baseline)
    if stream.trainer and baseline is None:
        raise ValueError("A sparse sender needs its preceding acknowledged baseline")
    device = stream.device
    _sync(device)
    host_rss_start = _rss()
    gpu_allocated_start = _allocated(device)
    gpu_free_start = _free(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    transfers = changed_values = total_values = logical_bytes = metadata_bytes = collectives = 0
    detect_seconds = construct_seconds = pack_allocation_seconds = transfer_seconds = apply_seconds = 0.0

    for item, landing in stream.transfers():
        transfers += 1
        current = stream.source_view(item) if landing is None else None
        numel = current.numel() if current is not None else landing.installed.numel()
        total_values += numel
        if current is not None:
            previous = _baseline_view(stream, item, baseline)
            patch, detect, construct, pack_allocation = _encode_timed(current, previous, encoding, device)
            detect_seconds += detect
            construct_seconds += construct
            pack_allocation_seconds += pack_allocation
            count = patch.changed
            changed_values += count
        else:
            patch = None
            count = 0

        stage = time.perf_counter()
        metadata = torch.tensor([count], dtype=torch.int64, device=device)
        collectives += _relay(stream, item, metadata)
        _sync(device)
        count = int(metadata.item())
        changed_values += count if patch is None else 0
        metadata_bytes += metadata.numel() * metadata.element_size()
        if count:
            if patch is None:
                pos_dtype = torch.int32 if encoding == "indices" else torch.uint8
                pos_numel = count if encoding == "indices" else (numel + 7) // 8
                positions = torch.empty(pos_numel, dtype=pos_dtype, device=device)
                values = torch.empty(count, dtype=landing.wire_dtype, device=device)
                patch = Patch(encoding, numel, positions, values)
            collectives += _relay(stream, item, patch.positions)
            collectives += _relay(stream, item, patch.values)
            _sync(device)
            logical_bytes += patch.payload_bytes
        transfer_seconds += time.perf_counter() - stage

        if landing is not None and count:
            stage = time.perf_counter()
            apply_patch(landing.installed, patch)
            _sync(device)
            apply_seconds += time.perf_counter() - stage

    _sync(device)
    report = SparseReport(
        participant=stream.participant,
        version=version,
        encoding=encoding,
        transfers=transfers,
        changed_values=changed_values,
        total_values=total_values,
        logical_bytes=logical_bytes + metadata_bytes,
        metadata_bytes=metadata_bytes,
        collectives=collectives,
        detect_seconds=detect_seconds,
        construct_seconds=construct_seconds,
        pack_allocation_seconds=pack_allocation_seconds,
        transfer_seconds=transfer_seconds,
        apply_seconds=apply_seconds,
        seconds=time.perf_counter() - started,
        gpu_allocated_start=gpu_allocated_start,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        gpu_free_start=gpu_free_start,
        gpu_free_end=_free(device),
        host_rss_start=host_rss_start,
        host_rss_end=_rss(),
        host_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    return asdict(report)


def _global_buckets(stream: Stream) -> list[tuple]:
    """Use one schedule-wide bucket order so overlapping NCCL groups cannot diverge."""
    maximum_bytes = 128 * 1024 * 1024
    buckets: list[tuple] = []
    current = []
    current_bytes = 0
    for item in stream.schedule.experts:
        if current and (item.group != current[0].group or current_bytes + item.entry.nbytes > maximum_bytes):
            buckets.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item.entry.nbytes
    if current:
        buckets.append(tuple(current))
    buckets.extend((item,) for item in stream.schedule.dense)
    return buckets


def _local_bucket(stream: Stream, bucket: tuple) -> list[tuple]:
    first = bucket[0]
    if isinstance(first, ExpertBroadcast):
        if first.root == stream.participant:
            return [(item, None) for item in bucket]
        if stream.participant in first.destinations:
            return [(item, stream.expert_landing(item)) for item in bucket]
        return []
    if first.root == stream.participant:
        return [(first, None)]
    if stream.lands(first):
        return [(first, stream.dense_landing(first))]
    return []


@torch.no_grad()
def run_dense_expert_bucket(stream: Stream, version: int, *, profile: bool = False) -> dict:
    """Coalesce compatible dense expert tensors, retaining #689's dense nonexpert route."""
    device = stream.device
    _sync(device)
    host_rss_start = _rss()
    gpu_allocated_start = _allocated(device)
    gpu_free_start = _free(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    transfers = total_values = logical_bytes = collectives = 0
    pack_allocation_seconds = transfer_seconds = apply_seconds = 0.0

    for bucket in _global_buckets(stream):
        local = _local_bucket(stream, bucket)
        if not local:
            continue
        transfers += len(local)
        if not isinstance(bucket[0], ExpertBroadcast):
            item, landing = local[0]
            total_values += item.source.numel
            stage = time.perf_counter()
            if landing is None:
                logical_bytes += stream.send(item)
                collectives += 1
            else:
                wire = stream.wire_tensor(landing)
                logical_bytes += stream.receive(item, wire)
                collectives += 2
                if not landing.direct:
                    landing.installed.copy_(wire)
            if profile:
                _sync(device)
                transfer_seconds += time.perf_counter() - stage
            continue

        sizes = [
            stream.source_view(item).numel() if landing is None else landing.installed.numel()
            for item, landing in local
        ]
        total_values += sum(sizes)
        sender = local[0][1] is None
        stage = time.perf_counter()
        if sender:
            sources = [stream.source_view(item).contiguous().view(-1) for item, _ in local]
            wire = torch.cat(sources) if len(sources) > 1 else sources[0]
        elif len(local) == 1:
            wire = local[0][1].installed.view(-1)
        else:
            wire = torch.empty(sum(sizes), dtype=torch.bfloat16, device=device)
        if profile:
            _sync(device)
            pack_allocation_seconds += time.perf_counter() - stage

        stage = time.perf_counter()
        dist.broadcast(wire, src=0, group=stream.groups[local[0][0].group])
        collectives += 1
        logical_bytes += wire.numel() * wire.element_size()
        if profile:
            _sync(device)
            transfer_seconds += time.perf_counter() - stage

        if not sender and len(local) > 1:
            stage = time.perf_counter()
            offset = 0
            for (_, landing), size in zip(local, sizes):
                landing.installed.copy_(wire.narrow(0, offset, size).view_as(landing.installed))
                offset += size
            if profile:
                _sync(device)
                apply_seconds += time.perf_counter() - stage

    _sync(device)
    report = SparseReport(
        participant=stream.participant,
        version=version,
        encoding="dense_expert_bucket_profiled" if profile else "dense_expert_bucket_fast",
        transfers=transfers,
        changed_values=0,
        total_values=total_values,
        logical_bytes=logical_bytes,
        metadata_bytes=0,
        collectives=collectives,
        detect_seconds=0.0,
        construct_seconds=0.0,
        pack_allocation_seconds=pack_allocation_seconds,
        transfer_seconds=transfer_seconds,
        apply_seconds=apply_seconds,
        seconds=time.perf_counter() - started,
        gpu_allocated_start=gpu_allocated_start,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        gpu_free_start=gpu_free_start,
        gpu_free_end=_free(device),
        host_rss_start=host_rss_start,
        host_rss_end=_rss(),
        host_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    row = asdict(report)
    row["timing_mode"] = "profiled" if profile else "end_to_end_only"
    return row


@torch.no_grad()
def run_sparse_bucket(
    stream: Stream,
    version: int,
    encoding: str,
    baseline: dict[str, torch.Tensor] | None = None,
    *,
    profile: bool = True,
    dense_fallback: bool = False,
) -> dict:
    """Coalesce compatible expert patches into 128 MiB buckets.

    ``profile=False`` removes per-component CUDA waits and reports only total wall time.
    ``dense_fallback`` sends non-expert slices through the unchanged dense stream.
    """
    if encoding not in {"indices", "bitmap"}:
        raise ValueError(f"Unknown bucket encoding {encoding}")
    if stream.trainer and baseline is None:
        raise ValueError("A sparse sender needs its preceding acknowledged baseline")
    device = stream.device
    _sync(device)
    host_rss_start = _rss()
    gpu_allocated_start = _allocated(device)
    gpu_free_start = _free(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    transfers = changed_values = total_values = logical_bytes = metadata_bytes = collectives = 0
    detect_seconds = construct_seconds = pack_allocation_seconds = transfer_seconds = apply_seconds = 0.0

    for bucket in _global_buckets(stream):
        local = _local_bucket(stream, bucket)
        if not local:
            continue
        transfers += len(local)
        if dense_fallback and not isinstance(bucket[0], ExpertBroadcast):
            item, landing = local[0]
            total_values += item.source.numel
            stage = time.perf_counter()
            if landing is None:
                logical_bytes += stream.send(item)
                collectives += 1
            else:
                wire = stream.wire_tensor(landing)
                logical_bytes += stream.receive(item, wire)
                collectives += 2
                if not landing.direct:
                    landing.installed.copy_(wire)
            if profile:
                _sync(device)
                transfer_seconds += time.perf_counter() - stage
            continue
        sizes = [
            stream.source_view(item).numel() if landing is None else landing.installed.numel()
            for item, landing in local
        ]
        total_values += sum(sizes)
        sender = local[0][1] is None
        patches = []
        counts = []
        if sender:
            for item, _ in local:
                patch, detect, construct, allocation = _encode_timed(
                    stream.source_view(item),
                    _baseline_view(stream, item, baseline),
                    encoding,
                    device,
                    profile=profile,
                )
                patches.append(patch)
                counts.append(patch.changed)
                if profile:
                    detect_seconds += detect
                    construct_seconds += construct
                    pack_allocation_seconds += allocation
            changed_values += sum(counts)

        stage = time.perf_counter()
        metadata = (
            torch.tensor(counts, dtype=torch.int64, device=device)
            if sender
            else torch.empty(len(local), dtype=torch.int64, device=device)
        )
        collectives += _relay(stream, local[0][0], metadata)
        if profile:
            _sync(device)
        if not sender:
            counts = metadata.tolist()
            changed_values += sum(counts)
        metadata_bytes += metadata.numel() * metadata.element_size()
        if any(counts):
            if sender:
                pack_started = time.perf_counter()
                positions_parts = [patch.positions for patch in patches if patch.changed]
                values_parts = [patch.values for patch in patches if patch.changed]
                positions = torch.cat(positions_parts) if len(positions_parts) > 1 else positions_parts[0]
                values = torch.cat(values_parts) if len(values_parts) > 1 else values_parts[0]
                if profile:
                    _sync(device)
                    pack_allocation_seconds += time.perf_counter() - pack_started
            else:
                position_count = sum(
                    count if encoding == "indices" else (size + 7) // 8 for size, count in zip(sizes, counts) if count
                )
                position_dtype = torch.int32 if encoding == "indices" else torch.uint8
                positions = torch.empty(position_count, dtype=position_dtype, device=device)
                values = torch.empty(sum(counts), dtype=local[0][1].wire_dtype, device=device)
            collectives += _relay(stream, local[0][0], positions)
            collectives += _relay(stream, local[0][0], values)
            if profile:
                _sync(device)
            logical_bytes += positions.numel() * positions.element_size() + values.numel() * values.element_size()
        if profile:
            transfer_seconds += time.perf_counter() - stage

        if not sender and any(counts):
            stage = time.perf_counter()
            position_offset = value_offset = 0
            for (_, landing), size, count in zip(local, sizes, counts):
                if not count:
                    continue
                position_count = count if encoding == "indices" else (size + 7) // 8
                patch = Patch(
                    encoding,
                    size,
                    positions.narrow(0, position_offset, position_count),
                    values.narrow(0, value_offset, count),
                )
                apply_patch(landing.installed, patch)
                position_offset += position_count
                value_offset += count
            if profile:
                _sync(device)
                apply_seconds += time.perf_counter() - stage

    _sync(device)
    report = SparseReport(
        participant=stream.participant,
        version=version,
        encoding=f"{encoding}_bucket",
        transfers=transfers,
        changed_values=changed_values,
        total_values=total_values,
        logical_bytes=logical_bytes + metadata_bytes,
        metadata_bytes=metadata_bytes,
        collectives=collectives,
        detect_seconds=detect_seconds,
        construct_seconds=construct_seconds,
        pack_allocation_seconds=pack_allocation_seconds,
        transfer_seconds=transfer_seconds,
        apply_seconds=apply_seconds,
        seconds=time.perf_counter() - started,
        gpu_allocated_start=gpu_allocated_start,
        gpu_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        gpu_free_start=gpu_free_start,
        gpu_free_end=_free(device),
        host_rss_start=host_rss_start,
        host_rss_end=_rss(),
        host_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    row = asdict(report)
    if dense_fallback:
        row["encoding"] = "indices_expert_bucket_profiled" if profile else "indices_expert_bucket_fast"
    elif not profile:
        row["encoding"] = "indices_bucket_fast"
    row["timing_mode"] = "profiled" if profile else "end_to_end_only"
    return row
