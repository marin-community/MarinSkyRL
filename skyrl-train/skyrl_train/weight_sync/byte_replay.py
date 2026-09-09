"""Exact comparisons for an untimed frozen-sender replay.

The caller owns the transferred source views, installed receiver views and
scratch storage. This module neither gathers weights nor allocates replay
buffers. A native transport must separately prove the source stayed frozen and
that these views cover every installed parameter byte.
"""

from dataclasses import dataclass
from typing import Iterable

import torch


MAX_SCRATCH_BYTES = 2**20


@dataclass(frozen=True)
class ByteComparison:
    compared_bytes: int
    mismatches: int


def compare_installed_views(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    scratch: torch.Tensor,
    *,
    expected_bytes: int,
) -> ByteComparison:
    """Compare every byte, including signed zero and NaN payload bits.

Only the caller's boolean scratch and one scalar reduction are written. All
views must remain stable until comparison returns. Expected coverage is supplied
independently by the complete local manifest, rather than inferred from this
iterator's contents. An empty receiver slice is valid only with zero coverage.
    """
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError("Expected coverage must be a nonnegative byte count")
    if (
        scratch.dtype != torch.bool
        or scratch.ndim != 1
        or not scratch.is_contiguous()
        or not 0 < scratch.numel() <= MAX_SCRATCH_BYTES
    ):
        raise ValueError("Comparison scratch must be contiguous bool storage of at most 1 MiB")
    compared = mismatches = 0
    for source, installed in pairs:
        if source.shape != installed.shape or source.dtype != installed.dtype:
            raise ValueError("Installed shape/dtype differs from the frozen source")
        if any(t.device != scratch.device or not t.is_contiguous() for t in (source, installed)):
            raise ValueError("Comparison views and scratch must be contiguous on the same device")
        expected = source.detach().view(-1).view(torch.uint8)
        actual = installed.detach().view(-1).view(torch.uint8)
        if compared + expected.numel() > expected_bytes:
            raise ValueError("Replay exceeds the independently expected coverage")
        for offset in range(0, expected.numel(), scratch.numel()):
            count = min(scratch.numel(), expected.numel() - offset)
            work = scratch.narrow(0, 0, count)
            torch.ne(expected.narrow(0, offset, count), actual.narrow(0, offset, count), out=work)
            mismatches += int(torch.count_nonzero(work).item())
        compared += expected.numel()
    if compared != expected_bytes:
        raise ValueError("Replay did not cover the independently expected bytes")
    return ByteComparison(compared, mismatches)
