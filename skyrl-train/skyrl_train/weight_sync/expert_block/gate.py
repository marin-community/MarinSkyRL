"""The opt-in byte gate for expert-block sync: proves a sync landed every byte, for qualification runs.

Off by default and never part of a production sync. With ``verify: true`` the
driver runs it after each sync:

* every root re-sends exactly what it sent, every receiver lands it in scratch
  and counts bytes that differ from what it installed, and tallies the bytes it
  compared against every byte of every parameter it holds — so a wrong slot, a
  skipped tensor or a parameter the schedule never covers all show up;
* every trainer rank checks that its data-parallel peers hold byte-identical
  parameters, since roots rotate across those peers.

The replay costs about one extra sync on the wire; the compare is one elementwise
``ne`` per landed tensor.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.schedule import BF16
from skyrl_train.weight_sync.expert_block.source_views import (
    dense_destination_view,
    dense_source_view,
    expert_destination_view,
    expert_source_view,
)
from skyrl_train.weight_sync.expert_block.stream import Stream


@dataclass(frozen=True)
class ReplayReport:
    participant: int
    version: int
    compared_bytes: int
    parameter_bytes: int
    mismatched_bytes: int


def replay(stream: Stream, version: int) -> ReplayReport:
    """Re-run the sync's collectives; receivers compare each landed transfer against what they installed."""
    with torch.no_grad():
        return _replay(stream, version)


def _replay(stream: Stream, version: int) -> ReplayReport:
    schedule, me = stream.schedule, stream.participant
    largest = max(
        [item.entry.nbytes for item in schedule.experts if me in item.destinations]
        + [item.source.nbytes for item in schedule.dense if stream.lands(item)]
        + [0]
    )
    scratch = torch.empty(largest, dtype=torch.uint8, device=stream.device)
    mismatched = torch.zeros((), dtype=torch.int64, device=stream.device)
    compared = 0

    def compare(landed: torch.Tensor, installed: torch.Tensor) -> None:
        nonlocal compared
        if installed.dtype != landed.dtype:
            # The FP32 router weight was widened from BF16 on install; compare at wire precision.
            installed = installed.to(landed.dtype)
        mismatched.add_(landed.view(torch.uint8).ne(installed.contiguous().view(torch.uint8)).sum())
        compared += landed.numel() * landed.element_size()

    for item in schedule.experts:
        if item.root == me:
            dist.broadcast(expert_source_view(stream.expert_sources[item.entry.name], stream.sources), src=0, group=stream.groups[item.group])
        elif me in item.destinations:
            landed = scratch.narrow(0, 0, item.entry.nbytes).view(torch.bfloat16)
            dist.broadcast(landed, src=0, group=stream.groups[item.group])
            compare(landed, expert_destination_view(item.entry, stream.parameters, stream.expert_maps))
    for item in schedule.dense:
        if item.root == me:
            dist.broadcast(dense_source_view(item.source, stream.sources), src=0, group=stream.groups[item.group])
        elif stream.lands(item):
            landed = scratch.narrow(0, 0, item.source.nbytes).view(getattr(torch, item.source.wire_dtype))
            if me in item.landings:
                dist.broadcast(landed, src=0, group=stream.groups[item.group])
            local_name, members = stream.local
            origin = next(rank for rank in item.landings if rank in members)
            dist.broadcast(landed, src=members.index(origin), group=stream.groups[local_name])
            compare(landed, dense_destination_view(item.source, stream.parameters))
    parameter_bytes = 0
    if not stream.trainer:
        parameter_bytes = sum(value.numel() * value.element_size() for value in stream.parameters.values())
        # A widened parameter is compared at wire width; count it that way for coverage.
        for item in schedule.dense:
            if stream.lands(item):
                installed = dense_destination_view(item.source, stream.parameters)
                if installed.dtype != torch.bfloat16 and item.source.wire_dtype == BF16:
                    parameter_bytes -= installed.numel() * (installed.element_size() - 2)
    if stream.device.type == "cuda":
        torch.cuda.synchronize(stream.device)
    return ReplayReport(me, version, compared, parameter_bytes, int(mismatched.item()))


@dataclass(frozen=True)
class ReplicaReport:
    participant: int
    version: int
    compared_bytes: int
    mismatched_bytes: int


def compare_replicas(sources: dict[str, torch.Tensor], groups: dict[str, dist.ProcessGroup], participant: int, version: int) -> ReplicaReport:
    """Count bytes on which this rank's parameters differ from the peers that hold the same ones.

    ``groups`` maps each parameter name to the process group of its replicas. Byte
    patterns are reduced as integers, so the comparison is exact.
    """
    mismatched = 0
    compared = 0
    with torch.no_grad():
        for name, source in sources.items():
            words = source.detach().contiguous().view(-1).view(torch.uint8)
            low = words.to(torch.int32)
            high = low.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=groups[name])
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=groups[name])
            mismatched += int(low.ne(high).sum().item())
            compared += words.numel()
    return ReplicaReport(participant, version, compared, mismatched)
