"""Exact comparisons for an untimed frozen-sender replay.

The caller owns the transferred source views, installed receiver views and
scratch storage. This module neither gathers weights nor allocates replay
buffers. A native transport must separately prove the source stayed frozen and
that these views cover every installed parameter byte.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


MAX_SCRATCH_BYTES = 2**20


@dataclass(frozen=True)
class ByteComparison:
    compared_bytes: int
    mismatches: int


class ReceiverByteCoverage:
    """Audit receiver storage intervals against an independent parameter inventory.

    Inventory tensors must be the actual installed, contiguous parameters. The
    caller retains them throughout replay. Tied/aliased parameters must first be
    represented once; overlapping inventory entries are rejected, not counted
    twice. Pointer arithmetic is metadata only and performs no device copies.
    """

    def __init__(self, parameters: Mapping[str, torch.Tensor]):
        if not parameters:
            raise ValueError("Receiver parameter inventory must be nonempty")
        self._parameters = dict(parameters)
        self._ranges = {}
        self._seen = {name: [] for name in parameters}
        for name, tensor in parameters.items():
            if not isinstance(name, str) or not name or not tensor.is_contiguous() or not tensor.numel():
                raise ValueError("Receiver inventory requires named nonempty contiguous parameters")
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            for device, left, right in self._ranges.values():
                if device == tensor.device and max(left, start) < min(right, end):
                    raise ValueError("Receiver inventory aliases parameter storage")
            self._ranges[name] = (tensor.device, start, end)

    @property
    def expected_bytes(self) -> int:
        return sum(end - start for _, start, end in self._ranges.values())

    def observe(self, installed: torch.Tensor) -> None:
        if not installed.is_contiguous() or not installed.numel():
            raise ValueError("Receiver replay view must be nonempty and contiguous")
        start = installed.data_ptr()
        end = start + installed.numel() * installed.element_size()
        matches = [
            name
            for name, (device, left, right) in self._ranges.items()
            if device == installed.device and left <= start < end <= right
        ]
        if len(matches) != 1:
            raise ValueError("Replay view is outside the independently registered receiver parameters")
        name = matches[0]
        for left, right in self._seen[name]:
            if max(left, start) < min(right, end):
                raise ValueError("Replay compares an installed receiver byte more than once")
        self._seen[name].append((start, end))

    def finish(self) -> int:
        for name, (device, start, end) in self._ranges.items():
            tensor = self._parameters[name]
            if (
                tensor.device != device
                or tensor.data_ptr() != start
                or tensor.numel() * tensor.element_size() != end - start
            ):
                raise ValueError("Receiver parameter storage changed during replay")
            cursor = start
            for left, right in sorted(self._seen[name]):
                if left != cursor:
                    raise ValueError("Replay missed installed receiver bytes")
                cursor = right
            if cursor != end:
                raise ValueError("Replay missed installed receiver bytes")
        return self.expected_bytes


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
