"""Verified payloads shared by distributed fault-injection workers."""

from __future__ import annotations

import torch
import torch.distributed as dist


RANK_VALUE_STRIDE = 100


def run_verified_all_to_all(
    group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
    total_values: int,
) -> None:
    """Exchange evenly split rank-tagged values and verify every received segment."""

    group_ranks = tuple(dist.get_process_group_ranks(group))
    if total_values % len(group_ranks) != 0:
        raise ValueError(f"total_values={total_values} must divide evenly over {len(group_ranks)} ranks")
    group_rank = group_ranks.index(rank)
    values_per_peer = total_values // len(group_ranks)
    input_values = (
        torch.arange(total_values, device=device, dtype=torch.int64) + rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty_like(input_values)
    dist.all_to_all_single(output_values, input_values, group=group)
    offsets = torch.arange(values_per_peer, device=device, dtype=torch.int64)
    expected_values = torch.cat(
        [
            offsets + source_rank * RANK_VALUE_STRIDE + group_rank * values_per_peer
            for source_rank in group_ranks
        ]
    )
    torch.testing.assert_close(output_values, expected_values)
