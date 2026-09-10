"""Complete source-copy comparison before rotating K10 broadcast roots.

Proof groups are separate from installation groups. Expert owners compare every
original source byte across expert-DP replicas; dense owners additionally compare
across EP copies on the same PP stage. Logical proof traffic is not NIC egress.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.byte_replay import compare_installed_views
from skyrl_train.weight_sync.shard_session import SourceReplicaProof, storage_versions


@dataclass(frozen=True)
class ReplicaTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    expert: bool

    @property
    def nbytes(self):
        count = 1
        for dim in self.shape:
            count *= dim
        return count * {"bfloat16": 2, "float32": 4}[self.dtype]


@dataclass(frozen=True)
class ReplicaCatalogue:
    rank: int
    tensors: tuple[ReplicaTensor, ...]


@dataclass(frozen=True)
class ReplicaGroup:
    name: str
    members: tuple[int, ...]
    tensors: tuple[ReplicaTensor, ...]
    # Primary groups cover each local source once. Dense groups independently
    # prove EP copies but do not inflate unique source coverage in the receipt.
    primary: bool


@dataclass(frozen=True)
class ReplicaPlan:
    groups: tuple[ReplicaGroup, ...]
    catalogue: tuple[ReplicaCatalogue, ...]

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_replica_catalogue(rank, slices, sources):
    kinds = {}
    for descriptor in slices:
        kinds.setdefault(descriptor.source_key, set()).add(descriptor.expert)
    if set(kinds) != set(sources) or any(len(values) != 1 for values in kinds.values()):
        raise ValueError("Every original source requires one unambiguous expert/dense classification")
    rows = []
    for name, source in sorted(sources.items()):
        if not source.is_contiguous() or not source.numel() or source.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Replica comparison requires contiguous BF16/FP32 source storage")
        rows.append(
            ReplicaTensor(name, tuple(source.shape), str(source.dtype).removeprefix("torch."), kinds[name].pop())
        )
    return ReplicaCatalogue(rank, tuple(rows))


def build_replica_plan(trainers, schedule, catalogue):
    """Bind actual source metadata to the already-qualified complete topology."""
    global_ranks = dict(schedule.trainer_global_ranks)
    if set(global_ranks) != {row.rank for row in trainers} or len(trainers) != len(global_ranks):
        raise ValueError("Replica topology differs from the installation schedule")
    rows = {row.rank: row for row in catalogue}
    if len(rows) != len(catalogue) or set(rows) != set(global_ranks.values()):
        raise ValueError("Replica catalogue misses or duplicates a policy rank")
    for row in catalogue:
        if type(row.rank) is not int or not row.tensors or len({item.name for item in row.tensors}) != len(row.tensors):
            raise ValueError("Replica catalogue requires unique typed rank/tensor identities")
        if any(
            not item.name
            or item.dtype not in ("bfloat16", "float32")
            or type(item.expert) is not bool
            or any(type(dim) is not int or dim <= 0 for dim in item.shape)
            for item in row.tensors
        ):
            raise ValueError("Replica catalogue contains unsupported tensor metadata")
    groups = []
    for pp, ep in sorted({(row.pp, row.ep) for row in trainers}):
        members = tuple(
            global_ranks[row.rank] for row in sorted(trainers, key=lambda row: row.dp) if (row.pp, row.ep) == (pp, ep)
        )
        tensors = tuple(sorted(rows[members[0]].tensors, key=lambda item: item.name))
        if any(tuple(sorted(rows[rank].tensors, key=lambda item: item.name)) != tensors for rank in members):
            raise ValueError("Expert-DP copies have different complete source inventories")
        groups.append(ReplicaGroup(f"replica-pp{pp}-ep{ep}", members, tensors, True))
    for pp in sorted({row.pp for row in trainers}):
        members = tuple(
            global_ranks[row.rank] for row in sorted(trainers, key=lambda row: (row.dp, row.ep)) if row.pp == pp
        )
        tensors = tuple(
            sorted((item for item in rows[members[0]].tensors if not item.expert), key=lambda item: item.name)
        )
        if not tensors or any(
            tuple(sorted((item for item in rows[rank].tensors if not item.expert), key=lambda item: item.name))
            != tensors
            for rank in members
        ):
            raise ValueError("Dense PP copies have different source inventories across DP/EP")
        groups.append(ReplicaGroup(f"dense-pp{pp}", members, tensors, False))
    return ReplicaPlan(tuple(groups), tuple(sorted(catalogue, key=lambda row: row.rank)))


class FullReplicaComparator:
    """Use prepared groups and bounded independent storage; never overwrite weights."""

    def __init__(self, rank, plan, groups, transfer_workspace, comparison_workspace, *, broadcast_source_ranks=None):
        self.rank, self.plan, self.groups = rank, plan, groups
        self.broadcast_source_ranks = (
            {item.name: 0 for item in plan.groups if rank in item.members}
            if broadcast_source_ranks is None
            else dict(broadcast_source_ranks)
        )
        if set(self.broadcast_source_ranks) != {item.name for item in plan.groups if rank in item.members} or any(
            type(value) is not int or value < 0 for value in self.broadcast_source_ranks.values()
        ):
            raise ValueError("Replica collective roots require complete typed native identities")
        self.transfer, self.comparison = transfer_workspace, comparison_workspace
        self.last_receipt = None
        if (
            transfer_workspace.dtype != torch.uint8
            or transfer_workspace.ndim != 1
            or not transfer_workspace.is_contiguous()
            or not transfer_workspace.numel()
            or comparison_workspace.dtype != torch.bool
            or comparison_workspace.ndim != 1
            or not comparison_workspace.is_contiguous()
            or not 0 < comparison_workspace.numel() <= 64 * 1024
            or transfer_workspace.device != comparison_workspace.device
        ):
            raise ValueError("Replica proof requires independent byte workspace and at most 64 KiB bool workspace")
        matching = [row for row in plan.catalogue if row.rank == rank]
        if len(matching) != 1:
            raise ValueError("Replica comparator rank is absent from the complete plan")
        self.tensors = matching[0].tensors
        for item in plan.groups:
            if rank in item.members:
                group = groups[item.name]
                if group.size() != len(item.members) or group.rank() != item.members.index(rank):
                    raise ValueError("Actual replica communicator differs from typed source membership")

    def __call__(self, sources, manifest_id, publication_id, rank):
        if rank != self.rank or type(publication_id) is not int or publication_id < 0:
            raise ValueError("Replica proof caller identity differs from preparation")
        if set(sources) != {item.name for item in self.tensors}:
            raise ValueError("Live source inventory differs from prepared replica catalogue")
        buffers = [self.transfer, self.comparison]
        for item in self.tensors:
            value = sources[item.name]
            if (
                tuple(value.shape) != item.shape
                or str(value.dtype).removeprefix("torch.") != item.dtype
                or not value.is_contiguous()
                or value.device != self.transfer.device
            ):
                raise ValueError("Live source layout differs from prepared replica catalogue")
            buffers.append(value)
        for index, left in enumerate(buffers):
            for right in buffers[index + 1 :]:
                if max(left.data_ptr(), right.data_ptr()) < min(
                    left.data_ptr() + left.numel() * left.element_size(),
                    right.data_ptr() + right.numel() * right.element_size(),
                ):
                    raise ValueError("Replica proof workspace/source storage overlaps")
        before = storage_versions(sources)
        if self.transfer.is_cuda:
            torch.cuda.synchronize(self.transfer.device)
        rows, unique_bytes, total_mismatches = [], 0, 0
        # Network chunks use the existing transfer workspace. The local byte
        # comparator independently splits each received chunk by bool scratch.
        count = self.transfer.numel()
        for item in self.plan.groups:
            if rank not in item.members:
                continue
            compared = mismatches = 0
            group = self.groups[item.name]
            for tensor in item.tensors:
                local = sources[tensor.name].detach().view(-1).view(torch.uint8)
                for offset in range(0, tensor.nbytes, count):
                    size = min(count, tensor.nbytes - offset)
                    source = local.narrow(0, offset, size)
                    wire = self.transfer.narrow(0, 0, size)
                    if rank == item.members[0]:
                        wire.copy_(source)
                    dist.broadcast(wire, src=self.broadcast_source_ranks[item.name], group=group)
                    result = compare_installed_views(((wire, source),), self.comparison, expected_bytes=size)
                    compared += result.compared_bytes
                    mismatches += result.mismatches
            # Complete every ordered collective even when one copy differs.
            # All participants receive the group's mismatch disposition.
            mismatch_tensor = torch.tensor(mismatches, dtype=torch.int64, device=self.transfer.device)
            dist.all_reduce(mismatch_tensor, group=group)
            mismatch_sum = int(mismatch_tensor.item())
            del mismatch_tensor
            rows.append(
                {
                    "group": item.name,
                    "members": item.members,
                    "primary": item.primary,
                    "compared_bytes": compared,
                    "mismatches": mismatch_sum,
                }
            )
            if item.primary:
                unique_bytes += compared
            total_mismatches += mismatch_sum
        if self.transfer.is_cuda:
            torch.cuda.synchronize(self.transfer.device)
        expected = sum(item.nbytes for item in self.tensors)
        if unique_bytes != expected or storage_versions(sources) != before:
            raise ValueError("Replica comparison lost complete stable source coverage")
        self.last_receipt = {
            "plan_id": self.plan.identity,
            "rank": rank,
            "manifest_id": manifest_id,
            "publication_id": publication_id,
            "groups": rows,
            "unique_source_bytes": unique_bytes,
            "mismatches": total_mismatches,
        }
        return SourceReplicaProof(
            manifest_id, publication_id, rank, unique_bytes, expected, total_mismatches, before, "full-byte-comparison"
        )


def borrowed_policy_groups(parallel_state, rank, schedule, plan):
    """Reuse initialized Megatron groups only after exact native membership joins."""
    native_for_global = {global_rank: native for native, global_rank in schedule.trainer_global_ranks}
    groups, sources, receipts = {}, {}, []
    for item in plan.groups:
        if rank not in item.members:
            continue
        group = (
            parallel_state.get_expert_data_parallel_group()
            if item.primary
            else parallel_state.get_data_parallel_group()
        )
        expected = tuple(native_for_global[member] for member in item.members)
        actual = tuple(dist.get_process_group_ranks(group))
        if actual != expected or group.rank() != item.members.index(rank) or group.size() != len(expected):
            raise ValueError("Existing Megatron group differs from the exact source-replica membership")
        groups[item.name] = group
        # torch.distributed.broadcast src uses default-world/global identity;
        # standalone custom groups instead register identity ranks starting at 0.
        sources[item.name] = actual[0]
        receipts.append(
            {
                "group": item.name,
                "global_members": item.members,
                "native_members": actual,
                "local_group_rank": group.rank(),
                "borrowed": True,
            }
        )
    return groups, sources, tuple(receipts)
