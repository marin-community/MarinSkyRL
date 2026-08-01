"""Verified payloads shared by distributed fault-injection workers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


RANK_VALUE_STRIDE = 100


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


def run_verified_all_gather(
    collective: CollectiveGroup,
    total_values: int,
) -> None:
    """Gather rank-tagged values and verify every source segment."""

    input_values = (
        torch.arange(total_values, device=collective.device, dtype=torch.int64) + collective.rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty(
        total_values * len(collective.ranks),
        dtype=input_values.dtype,
        device=collective.device,
    )
    dist.all_gather_into_tensor(output_values, input_values, group=collective.process_group)
    offsets = torch.arange(total_values, device=collective.device, dtype=torch.int64)
    expected_values = torch.cat([offsets + source_rank * RANK_VALUE_STRIDE for source_rank in collective.ranks])
    torch.testing.assert_close(output_values, expected_values)


def run_verified_all_to_all(
    collective: CollectiveGroup,
    total_values: int,
) -> None:
    """Exchange evenly split rank-tagged values and verify every received segment."""

    if total_values % len(collective.ranks) != 0:
        raise ValueError(f"total_values={total_values} must divide evenly over {len(collective.ranks)} ranks")
    group_rank = collective.ranks.index(collective.rank)
    values_per_peer = total_values // len(collective.ranks)
    input_values = (
        torch.arange(total_values, device=collective.device, dtype=torch.int64) + collective.rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty_like(input_values)
    dist.all_to_all_single(output_values, input_values, group=collective.process_group)
    offsets = torch.arange(values_per_peer, device=collective.device, dtype=torch.int64)
    expected_values = torch.cat(
        [offsets + source_rank * RANK_VALUE_STRIDE + group_rank * values_per_peer for source_rank in collective.ranks]
    )
    torch.testing.assert_close(output_values, expected_values)
