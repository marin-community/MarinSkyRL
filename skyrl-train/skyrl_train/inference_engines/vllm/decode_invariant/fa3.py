"""FA3 calls whose output for each Grug query row equals a one-row decode step of that row."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

FLASH_ATTN_VERSION = 3
# FA3's key tile on Hopper for this head dimension, bf16 and the causal and local kernels (``tile_size_fwd_sm90``):
# 128 keys whatever the step, so at one split a row's reduction over its keys follows its own position alone.
FA3_HEAD_DIM = 128


@dataclass(frozen=True)
class Fa3Requests:
    """FA3 varlen requests over the same query rows, in their order: ``query_start`` ``[E + 1]`` and ``key_lengths``
    ``[E]`` (int32), and ``owner`` ``[E]``, the original request of each new one (to gather its block-table row)."""

    query_start: torch.Tensor
    key_lengths: torch.Tensor
    owner: torch.Tensor


def fa3_window_requests(query_start: Sequence[int], key_lengths: Sequence[int], window: int) -> Fa3Requests:
    """Split each varlen request so that each row at a position of at least ``window`` is a request of its own.

    A request's query rows are the last ``q`` of its ``k`` keys. FA3's local kernel starts a query tile's keys at the
    window start of the tile's first row, which equals the row's own window start only for the first row.
    """
    query_start = np.asarray(query_start, dtype=np.int64)
    rows = np.arange(query_start[-1])
    request = np.repeat(np.arange(query_start.size - 1), np.diff(query_start))
    position = np.asarray(key_lengths, dtype=np.int64)[request] - query_start[request + 1] + rows
    starts = np.flatnonzero((rows == query_start[request]) | (position >= window))
    return Fa3Requests(
        query_start=torch.from_numpy(np.append(starts, query_start[-1])).to(torch.int32),
        key_lengths=torch.from_numpy(position[np.append(starts[1:], rows.size) - 1] + 1).to(torch.int32),
        owner=torch.from_numpy(request[starts]),
    )
