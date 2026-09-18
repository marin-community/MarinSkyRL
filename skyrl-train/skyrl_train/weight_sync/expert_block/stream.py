"""Run one sync's collectives on one participant, in the schedule's order.

Every participant walks the same schedule and issues only the collectives of
groups it belongs to, so the relative order of operations on every communicator
is identical everywhere. Expert matrices go straight from the owner's parameter
to the receiver's slot. Dense slices go from the owner to one receiver per
replica and are then re-broadcast on the replica's local group; a slice whose
installed dtype is wider than the wire dtype lands in scratch and is copied.
"""

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.schedule import BF16, Schedule
from skyrl_train.weight_sync.expert_block.source_views import (
    ExpertSource,
    dense_destination_view,
    dense_source_view,
    expert_destination_view,
    expert_source_view,
)


@dataclass(frozen=True)
class InstallReport:
    participant: int
    version: int
    expert_matrices: int
    wire_bytes: int
    seconds: float


class Stream:
    """A prepared participant: its groups, its live tensors and its place in the schedule."""

    def __init__(
        self,
        participant: int,
        schedule: Schedule,
        groups: dict[str, dist.ProcessGroup],
        *,
        sources: dict[str, torch.Tensor] | None = None,
        expert_sources: dict[str, ExpertSource] | None = None,
        parameters: dict[str, torch.Tensor] | None = None,
        expert_maps: dict[str, tuple[int, ...]] | None = None,
        device: torch.device,
    ):
        self.participant = participant
        self.schedule = schedule
        self.groups = groups
        self.trainer = participant < schedule.trainer_count
        self.sources = sources if self.trainer else None
        self.expert_sources = expert_sources if self.trainer else None
        self.parameters = None if self.trainer else parameters
        self.expert_maps = None if self.trainer else expert_maps
        self.device = device
        scratch_bytes = 0
        if not self.trainer:
            for item in schedule.dense:
                view = dense_destination_view(item.source, self.parameters)
                if view.dtype != torch.bfloat16 and item.source.wire_dtype == BF16:
                    scratch_bytes = max(scratch_bytes, item.source.nbytes)
        self.scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        # Resolve every view once so a bad layout fails at preparation, not mid-sync.
        for broadcast in schedule.experts:
            if broadcast.root == participant:
                expert_source_view(self.expert_sources[broadcast.entry.name], self.sources)
            elif participant in broadcast.destinations:
                expert_destination_view(broadcast.entry, self.parameters, self.expert_maps)
        for item in schedule.dense:
            if item.root == participant:
                dense_source_view(item.source, self.sources)
            elif not self.trainer:
                dense_destination_view(item.source, self.parameters)

    def run(self, version: int) -> InstallReport:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        matrices = wire_bytes = 0
        for broadcast in self.schedule.experts:
            if broadcast.root == self.participant:
                tensor = expert_source_view(self.expert_sources[broadcast.entry.name], self.sources)
            elif self.participant in broadcast.destinations:
                tensor = expert_destination_view(broadcast.entry, self.parameters, self.expert_maps)
            else:
                continue
            dist.broadcast(tensor, src=0, group=self.groups[broadcast.group])
            matrices += 1
            wire_bytes += broadcast.entry.nbytes
        local = self.local_group()
        for item in self.schedule.dense:
            if item.root == self.participant:
                dist.broadcast(dense_source_view(item.source, self.sources), src=0, group=self.groups[item.group])
                wire_bytes += item.source.nbytes
            elif not self.trainer:
                destination = dense_destination_view(item.source, self.parameters)
                widened = destination.dtype != torch.bfloat16 and item.source.wire_dtype == BF16
                landing = self.scratch.narrow(0, 0, item.source.nbytes).view(torch.bfloat16) if widened else destination
                if self.participant in item.landings:
                    dist.broadcast(landing, src=0, group=self.groups[item.group])
                local_name, members = local
                origin = next(rank for rank in item.landings if rank in members)
                dist.broadcast(landing, src=members.index(origin), group=self.groups[local_name])
                if widened:
                    destination.copy_(landing)
                wire_bytes += item.source.nbytes
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return InstallReport(self.participant, version, matrices, wire_bytes, time.perf_counter() - started)

    def local_group(self) -> tuple[str, tuple[int, ...]] | None:
        for group in self.schedule.groups:
            if group.name.startswith("local-") and self.participant in group.members:
                return group.name, group.members
        return None
