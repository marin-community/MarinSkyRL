"""Verified payloads shared by distributed fault-injection workers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh


VERIFICATION_DTYPE = torch.int64
REDUCTION_DTYPE = torch.float32
RANK_VALUE_STRIDE = 100


def numel_for_mib(payload_mib: int, dtype: torch.dtype) -> int:
    return payload_mib * 1024 * 1024 // torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True)
class CollectiveGroup:
    process_group: dist.ProcessGroup
    ranks: tuple[int, ...]
    rank: int
    device: torch.device

    @classmethod
    def from_process_group(
        cls,
        process_group: dist.ProcessGroup,
        rank: int,
        device: torch.device,
    ) -> CollectiveGroup:
        return cls(process_group, tuple(dist.get_process_group_ranks(process_group)), rank, device)


@dataclass(frozen=True)
class MeshCollectives:
    ep: CollectiveGroup
    fsdp: CollectiveGroup

    @classmethod
    def from_mesh(cls, mesh: DeviceMesh, rank: int, device: torch.device) -> MeshCollectives:
        ep_process_group = mesh["ep"].get_group()
        fsdp_process_group = mesh["fsdp"].get_group()
        return cls(
            ep=CollectiveGroup.from_process_group(ep_process_group, rank, device),
            fsdp=CollectiveGroup.from_process_group(fsdp_process_group, rank, device),
        )


def run_verified_all_gather(
    collective: CollectiveGroup,
    input_values_per_rank: int,
) -> None:
    """Gather rank-tagged values and verify every source segment."""

    input_values = (
        torch.arange(input_values_per_rank, device=collective.device, dtype=VERIFICATION_DTYPE)
        + collective.rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty(
        input_values_per_rank * len(collective.ranks),
        dtype=input_values.dtype,
        device=collective.device,
    )
    dist.all_gather_into_tensor(output_values, input_values, group=collective.process_group)
    offsets = torch.arange(input_values_per_rank, device=collective.device, dtype=VERIFICATION_DTYPE)
    expected_values = torch.cat([offsets + source_rank * RANK_VALUE_STRIDE for source_rank in collective.ranks])
    torch.testing.assert_close(output_values, expected_values)


def run_verified_all_to_all(
    collective: CollectiveGroup,
    input_values_per_rank: int,
) -> None:
    """Exchange evenly split rank-tagged values and verify every received segment."""

    if input_values_per_rank % len(collective.ranks) != 0:
        raise ValueError(
            f"input_values_per_rank={input_values_per_rank} must divide evenly over {len(collective.ranks)} ranks"
        )
    group_rank = collective.ranks.index(collective.rank)
    values_per_peer = input_values_per_rank // len(collective.ranks)
    input_values = (
        torch.arange(input_values_per_rank, device=collective.device, dtype=VERIFICATION_DTYPE)
        + collective.rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty_like(input_values)
    dist.all_to_all_single(output_values, input_values, group=collective.process_group)
    offsets = torch.arange(values_per_peer, device=collective.device, dtype=VERIFICATION_DTYPE)
    expected_values = torch.cat(
        [offsets + source_rank * RANK_VALUE_STRIDE + group_rank * values_per_peer for source_rank in collective.ranks]
    )
    torch.testing.assert_close(output_values, expected_values)


def warm_ep_and_fsdp_communicators(
    collectives: MeshCollectives,
    *,
    rounds: int,
    ep_values_per_rank: int,
    fsdp_values_per_rank: int,
) -> None:
    """Run verified EP and FSDP rounds, then synchronize the world group."""

    for _ in range(rounds):
        run_verified_all_to_all(collectives.ep, ep_values_per_rank)
        run_verified_all_gather(collectives.fsdp, fsdp_values_per_rank)
    dist.barrier()
