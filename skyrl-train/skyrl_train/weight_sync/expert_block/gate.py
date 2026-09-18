"""The opt-in byte gate for expert-block sync: proves a sync landed every byte, for qualification runs.

Off by default and never part of a production sync. With ``verify: true`` the
driver runs it after each sync:

* every root re-sends exactly what it sent, every receiver lands it in scratch
  and counts bytes that differ from what it installed, and tallies the bytes it
  compared against every byte of every parameter it holds (as the receiver
  installs them: a padded vocabulary tensor counts only its HF rows) — so a
  wrong slot, a skipped tensor or a parameter the schedule never covers all
  show up;
* every trainer rank checks that its data-parallel peers hold byte-identical
  parameters, since roots rotate across those peers.

The replay costs about one extra sync on the wire; the compare is one elementwise
``ne`` per landed tensor.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.source_views import dense_source_view, expert_source_view
from skyrl_train.weight_sync.expert_block.stream import Landing, Stream


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
    landings = {}
    for item in schedule.experts:
        if me in item.destinations and item.root != me:
            landings[item.entry.name] = stream.expert_landing(item)
    for item in schedule.dense:
        if stream.lands(item):
            landings[item.source.identity()] = stream.dense_landing(item)
    scratch = torch.empty(
        max([landing.nbytes for landing in landings.values()] + [0]), dtype=torch.uint8, device=stream.device
    )
    mismatched = torch.zeros((), dtype=torch.int64, device=stream.device)
    compared = 0

    def landed(landing: Landing) -> torch.Tensor:
        return scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)

    def compare(landing: Landing, wire: torch.Tensor) -> None:
        nonlocal compared
        # A widened parameter is compared at wire precision.
        installed = landing.installed.to(landing.wire_dtype).contiguous()
        mismatched.add_(wire.contiguous().view(torch.uint8).ne(installed.view(torch.uint8)).sum())
        compared += landing.nbytes

    for item in schedule.experts:
        if item.root == me:
            dist.broadcast(
                expert_source_view(stream.expert_sources[item.entry.name], stream.sources),
                src=0,
                group=stream.groups[item.group],
            )
        elif me in item.destinations:
            landing = landings[item.entry.name]
            wire = landed(landing)
            dist.broadcast(wire, src=0, group=stream.groups[item.group])
            compare(landing, wire)
    for item in schedule.dense:
        if item.root == me:
            dist.broadcast(dense_source_view(item.source, stream.sources), src=0, group=stream.groups[item.group])
        elif stream.lands(item):
            landing = landings[item.source.identity()]
            wire = landed(landing)
            if me in item.landings:
                dist.broadcast(wire, src=0, group=stream.groups[item.group])
            local_name, members = stream.local
            origin = next(rank for rank in item.landings if rank in members)
            dist.broadcast(wire, src=members.index(origin), group=stream.groups[local_name])
            compare(landing, wire)
    parameter_bytes = 0
    if not stream.trainer:
        # Coverage counts every parameter byte at wire width, so a widened parameter counts as BF16.
        parameter_bytes = sum(value.numel() * value.element_size() for value in stream.parameters.values())
        for landing in landings.values():
            parameter_bytes -= landing.installed.numel() * (
                landing.installed.element_size() - landing.nbytes // landing.installed.numel()
            )
    if stream.device.type == "cuda":
        torch.cuda.synchronize(stream.device)
    return ReplayReport(me, version, compared, parameter_bytes, int(mismatched.item()))


@dataclass(frozen=True)
class ReplicaReport:
    participant: int
    version: int
    compared_bytes: int
    mismatched_bytes: int


# Bytes compared per collective. Each chunk is widened to int32 twice (the group minimum and
# maximum), so the working set is eight times this on a GPU the trainer already fills.
REPLICA_COMPARE_CHUNK_BYTES = 16 << 20


def compare_replicas(
    sources: dict[str, torch.Tensor],
    groups: dict[str, dist.ProcessGroup],
    participant: int,
    version: int,
    *,
    chunk_bytes: int = REPLICA_COMPARE_CHUNK_BYTES,
) -> ReplicaReport:
    """Count bytes on which this rank's parameters differ from the peers that hold the same ones.

    ``groups`` maps each parameter name to the process group of its replicas. Byte
    patterns are reduced as integers, so the comparison is exact.
    """
    mismatched = 0
    compared = 0
    with torch.no_grad():
        for name, source in sources.items():
            words = source.detach().contiguous().view(-1).view(torch.uint8)
            for start in range(0, words.numel(), chunk_bytes):
                low = words.narrow(0, start, min(chunk_bytes, words.numel() - start)).to(torch.int32)
                high = low.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN, group=groups[name])
                dist.all_reduce(high, op=dist.ReduceOp.MAX, group=groups[name])
                mismatched += int(low.ne(high).sum().item())
            compared += words.numel()
    return ReplicaReport(participant, version, compared, mismatched)
