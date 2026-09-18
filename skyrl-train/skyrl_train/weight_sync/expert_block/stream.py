"""Run one sync's collectives on one participant, in the schedule's order.

Every participant walks the same schedule and issues only the collectives of
groups it belongs to, so the relative order of operations on every communicator
is identical everywhere. A transfer whose destination is one contiguous run and
the wire dtype lands straight in the live parameter; a column block (a
tensor-parallel row shard) or a widened tensor (the FP32 router weight, BF16 on
the wire) lands in scratch and is copied into its region. Dense slices go from
the owner to one receiver per replica and are then re-broadcast on the replica's
local group.
"""

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, create_groups, destroy_groups, warm_groups
from skyrl_train.weight_sync.expert_block.schedule import LOCAL_GROUP_PREFIX, DenseBroadcast, ExpertBroadcast, Schedule
from skyrl_train.weight_sync.expert_block.source_views import (
    ExpertSource,
    dense_flat_view,
    dense_source_view,
    expert_slot_region,
    expert_slot_view,
    expert_source_view,
    region_view,
)


def storage_identity(tensors: dict[str, torch.Tensor]) -> dict[str, tuple]:
    """Where each live parameter's storage is; a sync must see the storage its plan was built on."""
    return {
        name: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
        for name, value in tensors.items()
    }


@dataclass(frozen=True)
class InstallReport:
    """What one participant moved in a sync, with the expert and dense phases timed separately."""

    participant: int
    version: int
    expert_matrices: int
    wire_bytes: int
    seconds: float
    expert_seconds: float = 0.0
    dense_seconds: float = 0.0


@dataclass(frozen=True)
class Landing:
    """Where a receiver puts a transfer: the installed region, and the tensor the collective writes."""

    installed: torch.Tensor
    wire_dtype: torch.dtype

    @property
    def direct(self) -> bool:
        return self.installed.is_contiguous() and self.installed.dtype == self.wire_dtype

    @property
    def nbytes(self) -> int:
        return self.installed.numel() * torch.empty(0, dtype=self.wire_dtype).element_size()


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
        hidden_size: int | None = None,
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
        self.hidden_size = hidden_size
        self.device = device
        self.local = self.local_group()
        # Resolve every view once so a bad layout fails at preparation, not mid-sync.
        scratch_bytes = 0
        for broadcast in schedule.experts:
            if broadcast.root == participant:
                expert_source_view(self.expert_sources[broadcast.entry.name], self.sources)
            elif participant in broadcast.destinations:
                landing = self.expert_landing(broadcast)
                if not landing.direct:
                    scratch_bytes = max(scratch_bytes, landing.nbytes)
        for item in schedule.dense:
            if item.root == participant:
                dense_source_view(item.source, self.sources)
            elif self.lands(item):
                landing = self.dense_landing(item)
                if not landing.direct:
                    scratch_bytes = max(scratch_bytes, landing.nbytes)
        self.scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)

    def lands(self, item: DenseBroadcast) -> bool:
        """Whether this receiver installs a dense slice: only the receivers of a stage holding the tensor do."""
        return not self.trainer and self.local is not None and self.local[0] in item.local_groups

    def expert_landing(self, broadcast: ExpertBroadcast) -> Landing:
        slot = expert_slot_view(broadcast.entry, self.parameters, self.expert_maps)
        return Landing(region_view(slot, expert_slot_region(broadcast.entry, slot, self.hidden_size)), torch.bfloat16)

    def dense_landing(self, item: DenseBroadcast) -> Landing:
        flat = dense_flat_view(item.source, self.parameters)
        return Landing(region_view(flat, item.source.region), getattr(torch, item.source.wire_dtype))

    def wire_tensor(self, landing: Landing) -> torch.Tensor:
        """The tensor a collective writes: the installed region itself, or scratch shaped like it."""
        if landing.direct:
            return landing.installed
        return self.scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)

    def run(self, version: int) -> InstallReport:
        with torch.no_grad():
            return self._run(version)

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def transfers(self):
        """This participant's part of a sync, in collective order.

        Yields ``(item, landing)``: ``landing`` is None when this participant is the item's
        root; items it neither sends nor lands are skipped. The gate replays exactly this walk,
        so the two never drift apart.
        """
        for broadcast in self.schedule.experts:
            if broadcast.root == self.participant:
                yield broadcast, None
            elif self.participant in broadcast.destinations:
                yield broadcast, self.expert_landing(broadcast)
        for item in self.schedule.dense:
            if item.root == self.participant:
                yield item, None
            elif self.lands(item):
                yield item, self.dense_landing(item)

    def send(self, item) -> int:
        """Broadcast one item this participant roots; returns the bytes put on the wire."""
        if isinstance(item, ExpertBroadcast):
            tensor = expert_source_view(self.expert_sources[item.entry.name], self.sources)
            nbytes = item.entry.nbytes
        else:
            tensor = dense_source_view(item.source, self.sources)
            nbytes = item.source.nbytes
        dist.broadcast(tensor, src=0, group=self.groups[item.group])
        return nbytes

    def receive(self, item, tensor: torch.Tensor) -> int:
        """Run the collectives that land one item into ``tensor``; returns the bytes received."""
        if isinstance(item, ExpertBroadcast):
            dist.broadcast(tensor, src=0, group=self.groups[item.group])
            return item.entry.nbytes
        # A dense slice reaches one landing worker per replica from the root, then fans out
        # within the replica from that worker.
        if self.participant in item.landings:
            dist.broadcast(tensor, src=0, group=self.groups[item.group])
        local_name, members = self.local
        origin = next(rank for rank in item.landings if rank in members)
        dist.broadcast(tensor, src=members.index(origin), group=self.groups[local_name])
        return item.source.nbytes

    def _run(self, version: int) -> InstallReport:
        self._sync_device()
        started = time.perf_counter()
        matrices = wire_bytes = 0
        experts_done = None
        for item, landing in self.transfers():
            if experts_done is None and isinstance(item, DenseBroadcast):
                self._sync_device()
                experts_done = time.perf_counter()
            if landing is None:
                wire_bytes += self.send(item)
            else:
                tensor = self.wire_tensor(landing)
                wire_bytes += self.receive(item, tensor)
                if not landing.direct:
                    landing.installed.copy_(tensor)
            if isinstance(item, ExpertBroadcast):
                matrices += 1
        self._sync_device()
        finished = time.perf_counter()
        if experts_done is None:
            experts_done = finished
        return InstallReport(
            self.participant,
            version,
            matrices,
            wire_bytes,
            finished - started,
            expert_seconds=experts_done - started,
            dense_seconds=finished - experts_done,
        )

    def local_group(self) -> tuple[str, tuple[int, ...]] | None:
        for group in self.schedule.groups:
            if group.name.startswith(LOCAL_GROUP_PREFIX) and self.participant in group.members:
                return group.name, group.members
        return None


def bind(participant: int, plan: Schedule, rendezvous: Rendezvous, device: torch.device, **tensors) -> tuple:
    """Create and warm this participant's groups and resolve its views; returns ``(groups, stream, warm-up seconds)``.

    ``tensors`` are the side's ``Stream`` keyword arguments. Groups already created are destroyed
    if warming or view resolution fails, so a failed bind leaves no communicator behind.
    """
    groups = create_groups(participant, plan.groups, rendezvous)
    try:
        warm = warm_groups(participant, plan.groups, groups, device)
        stream = Stream(participant, plan, groups, device=device, **tensors)
    except BaseException:
        destroy_groups(groups)
        raise
    return groups, stream, warm
