"""Ordered expert and replicated dense collectives for prepared K10 ranks.

Group creation, scheduler-idle acknowledgement and the policy weight lease are
owned by the enclosing native session. This runner performs blocking broadcasts
and synchronizes CUDA before returning; gate replay remains a separate phase.
"""

from dataclasses import replace

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.frozen_source_views import source_view
from skyrl_train.weight_sync.shard_source_inventory import expert_destination_view, expert_source_view


class ShardStreamRank:
    def __init__(
        self,
        rank,
        schedule,
        expert_views,
        dense_plan,
        sources,
        parameters,
        expert_maps,
        scratch,
        groups,
        local_group,
        *,
        dense_chunk_bytes=None,
    ):
        self.rank = rank
        self.schedule = schedule
        self.expert_views = {item.entry.name: item for item in expert_views}
        self.dense_plan = dense_plan
        self.dense_chunk_bytes = scratch.numel() if dense_chunk_bytes is None else dense_chunk_bytes
        if type(self.dense_chunk_bytes) is not int or not 4 <= self.dense_chunk_bytes <= scratch.numel():
            raise ValueError("Dense chunk capacity must be explicit, at least one FP32 element and within workspace")
        self.sources = sources
        self.parameters = parameters
        self.expert_maps = expert_maps
        self.scratch = scratch
        self.groups = groups
        self.local_group = local_group
        self.trainers = {global_rank: native for native, global_rank in schedule.trainer_global_ranks}
        self.receivers = {global_rank: native for native, global_rank in schedule.receiver_global_ranks}
        self.native_to_global = {native: global_rank for native, global_rank in schedule.trainer_global_ranks}
        self.receiver_to_global = {native: global_rank for native, global_rank in schedule.receiver_global_ranks}
        self.publication_id = None
        self.manifest_id = None
        self.completed = False
        if rank not in self.trainers and rank not in self.receivers:
            raise ValueError("Unknown stream participant")
        if len(self.expert_views) != len(expert_views) or set(self.expert_views) != {
            item.name for item in schedule.broadcasts
        }:
            raise ValueError("Expert view catalogue differs from the collective schedule")
        for item in schedule.broadcasts:
            if self.expert_views[item.name].entry.nbytes != item.nbytes:
                raise ValueError("Expert view bytes differ from scheduled collective")
        if scratch.dtype != torch.uint8 or scratch.ndim != 1 or not scratch.is_contiguous():
            raise ValueError("Stream scratch must be contiguous uint8 storage")
        largest = max(
            [item.nbytes for item in schedule.broadcasts]
            + [min(self.dense_chunk_bytes, self._dense_bytes(item)) for item in dense_plan]
        )
        if scratch.numel() < largest:
            raise ValueError("Existing stream workspace cannot hold the largest scheduled transfer")
        left, right = scratch.data_ptr(), scratch.data_ptr() + scratch.numel()
        for tensor in (*sources.values(), *parameters.values()):
            if tensor.device != scratch.device:
                raise ValueError("Stream inputs and workspace must share the rank device")
            if max(left, tensor.data_ptr()) < min(right, tensor.data_ptr() + tensor.numel() * tensor.element_size()):
                raise ValueError("Stream scratch must not overlap live source or installed storage")

        for group in schedule.groups:
            if rank in group.members:
                self._validate_group(groups[group.ep], group.members)
        for item in schedule.broadcasts:
            if rank == item.root:
                expert_source_view(self.expert_views[item.name], sources)
            elif rank in item.receiver_destinations:
                expert_destination_view(self.expert_views[item.name], parameters, expert_maps, backend="TRITON")
        for item in dense_plan:
            if rank == self.native_to_global[item.root_native_rank]:
                source_view(item.source, sources)
            if rank in self.receivers:
                native = self.receivers[rank]
                replica = next(i for i, ranks in enumerate(item.replica_fanout) if native in ranks)
                local_members = tuple(self.receiver_to_global[r] for r in item.replica_fanout[replica])
                self._validate_group(local_group, local_members)
                installed = parameters[item.source.hf_name]
                width = item.source.hf_offset + item.source.numel
                dtype = getattr(torch, item.source.wire_dtype)
                widening = (
                    item.source.hf_name.endswith(".mlp.router.weight")
                    and dtype == torch.bfloat16
                    and installed.dtype == torch.float32
                )
                if (
                    not installed.is_contiguous()
                    or width > installed.numel()
                    or (installed.dtype != dtype and not widening)
                ):
                    raise ValueError("Dense installed storage differs from the qualified wire range/conversion")

    def _validate_group(self, group, members):
        # Standalone custom groups span independent default worlds. dist.get_rank
        # translates the caller's default rank, so inspect the actual PG instead.
        if group is None or group.size() != len(members) or group.rank() != members.index(self.rank):
            raise ValueError("Actual communicator identity differs from prepared membership")

    @staticmethod
    def _dense_bytes(item):
        if item.source.wire_dtype not in ("bfloat16", "float32"):
            raise ValueError("Unqualified dense wire dtype")
        return item.source.numel * (2 if item.source.wire_dtype == "bfloat16" else 4)

    def begin(self, *, manifest_id, publication_id):
        if self.manifest_id is not None or not isinstance(manifest_id, str) or not manifest_id:
            raise ValueError("Stream requires a fresh named session")
        if type(publication_id) is not int or publication_id < 0:
            raise ValueError("Stream requires a typed publication version")
        self.manifest_id, self.publication_id = manifest_id, publication_id

    def reset(self, *, manifest_id, publication_id):
        """Reuse installed metadata only after the owning session completes its configured checks."""
        if not self.completed or manifest_id != self.manifest_id or publication_id != self.publication_id:
            raise ValueError("Cannot reset an incomplete or different shard publication")
        self.manifest_id = None
        self.publication_id = None
        self.completed = False

    def _workspace(self, nbytes, dtype):
        return self.scratch.narrow(0, 0, nbytes).view(dtype)

    def _broadcast(self, tensor, *, members, root, group):
        if self.rank not in members or root not in members:
            raise ValueError("Collective participant or root differs from prepared membership")
        dist.broadcast(tensor, src=members.index(root), group=group)

    def run(self, *, manifest_id, publication_id):
        if (
            self.completed
            or self.manifest_id is None
            or manifest_id != self.manifest_id
            or type(publication_id) is not int
            or publication_id != self.publication_id
        ):
            raise ValueError("Stream call does not match a fresh prepared publication")
        # Claim the call before the first collective: failures cannot replay a
        # partially installed publication through the same session.
        self.completed = True
        rows = []
        if self.scratch.is_cuda:
            torch.cuda.synchronize(self.scratch.device)
        for index, item in enumerate(self.schedule.broadcasts):
            members = self.schedule.groups[item.group_ep].members
            if self.rank not in members:
                continue
            view = self.expert_views[item.name]
            if self.rank == item.root:
                tensor = expert_source_view(view, self.sources).view(-1)
                role = "source"
            elif self.rank in self.receivers:
                tensor = expert_destination_view(view, self.parameters, self.expert_maps, backend="TRITON").view(-1)
                role = "installed"
            else:
                tensor = self._workspace(item.nbytes, torch.bfloat16)
                role = "scratch"
            self._broadcast(tensor, members=members, root=item.root, group=self.groups[item.group_ep])
            rows.append({"phase": "expert", "index": index, "role": role, "bytes": item.nbytes})
        for index, descriptor in enumerate(self.dense_plan):
            for item in dense_chunks(descriptor, self.dense_chunk_bytes):
                group = self.schedule.groups[item.group_ep]
                root = self.native_to_global[item.root_native_rank]
                nbytes = self._dense_bytes(item)
                dtype = getattr(torch, item.source.wire_dtype)
                if self.rank == root:
                    tensor = source_view(item.source, self.sources)
                else:
                    tensor = self._workspace(nbytes, dtype)
                if self.rank in group.members:
                    self._broadcast(tensor, members=group.members, root=root, group=self.groups[item.group_ep])
                if self.rank in self.receivers:
                    native_rank = self.receivers[self.rank]
                    replica = next(i for i, local_ranks in enumerate(item.replica_fanout) if native_rank in local_ranks)
                    local_members = tuple(self.receiver_to_global[r] for r in item.replica_fanout[replica])
                    landing = self.receiver_to_global[item.landing_native_ranks[replica]]
                    self._broadcast(tensor, members=local_members, root=landing, group=self.local_group)
                    installed = (
                        self.parameters[item.source.hf_name]
                        .view(-1)
                        .narrow(0, item.source.hf_offset, item.source.numel)
                    )
                    widening = (
                        item.source.hf_name.endswith(".mlp.router.weight")
                        and dtype == torch.bfloat16
                        and installed.dtype == torch.float32
                    )
                    if installed.dtype != dtype and not widening:
                        raise ValueError("Dense installed dtype differs from the qualified wire conversion")
                    with torch.no_grad():
                        installed.copy_(tensor)
                    rows.append(
                        {
                            "phase": "dense",
                            "index": index,
                            "hf_offset": item.source.hf_offset,
                            "numel": item.source.numel,
                            "wire_dtype": item.source.wire_dtype,
                            "wire_bytes": nbytes,
                            "role": "installed",
                            "bytes": installed.numel() * installed.element_size(),
                        }
                    )
                elif self.rank in group.members:
                    rows.append(
                        {
                            "phase": "dense",
                            "index": index,
                            "hf_offset": item.source.hf_offset,
                            "numel": item.source.numel,
                            "wire_dtype": item.source.wire_dtype,
                            "wire_bytes": nbytes,
                            "role": "source" if self.rank == root else "scratch",
                            "bytes": nbytes,
                        }
                    )
        if self.scratch.is_cuda:
            torch.cuda.synchronize(self.scratch.device)
        return {"rank": self.rank, "manifest_id": manifest_id, "publication_id": publication_id, "rows": rows}


def dense_chunks(item, capacity_bytes):
    """Preserve exact source/destination ranges while bounding dense landing storage."""
    if type(capacity_bytes) is not int or capacity_bytes < 4:
        raise ValueError("Dense chunks require at least one FP32 element")
    width = {"bfloat16": 2, "float32": 4}.get(item.source.wire_dtype)
    if width is None or item.source.numel <= 0:
        raise ValueError("Dense chunk descriptor has unqualified dtype or empty range")
    elements = capacity_bytes // width
    for offset in range(0, item.source.numel, elements):
        source = replace(
            item.source,
            source_offset=item.source.source_offset + offset,
            hf_offset=item.source.hf_offset + offset,
            numel=min(elements, item.source.numel - offset),
        )
        yield replace(item, source=source)
