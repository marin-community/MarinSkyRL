"""FA3 calls that give each Grug attention row the bytes of the same row in a decode step.

FA3 (vLLM's FlashAttention fork, Hopper) adds a row's key-block splits in an order set by the split count and by the
rows that share its query tile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

# FA3's query tile rows (``tile_size_fwd_sm90``) for Grug on H100: head dim 128, bf16, packed query heads.
FA3_BLOCK_M = 128
# ``prepare_varlen_num_blocks`` splits dynamically only for batches one CTA covers; ``mha_fwd`` runs a varlen call of
# more requests with one split.
FA3_DYNAMIC_SPLIT_MAX_BATCH = 992
# The split count of every FA3 request of a decode-invariant engine: FA3 cuts a query tile's key blocks ``[0, n)`` into
# runs of ``ceil(n / 4)`` blocks (``BlockMN::get_n_block_min_max``), so a row's bytes follow its own block count alone
# when every row of its tile has the same block count.
FA3_INVARIANT_SPLITS = 4


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _padded_requests(requests: int) -> int:
    return _ceil_div(requests, 4) * 4


def fa3_fixed_split_metadata_size(requests: int) -> int:
    """The length of ``fa3_fixed_split_metadata``'s tensor for a call of ``requests`` requests."""
    return 3 * _padded_requests(requests) + 1


def fa3_fixed_split_metadata(
    query_start: torch.Tensor, query_heads_per_kv_head: int, kv_heads: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """FA3's scheduler metadata for a causal or local varlen call that runs every request with
    ``FA3_INVARIANT_SPLITS`` splits, whatever else the call holds.

    The layout is ``mha_fwd``'s for packed query heads, dynamic splits and the head swizzle: per request its packed
    query rows, its split count and its heads per L2 section, each vector padded to a multiple of 4, then the tile
    semaphore, which starts at zero and which FA3's combine kernel zeroes after the call. A call of more than
    ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests runs unsplit whatever the metadata. ``out``, when given, receives the
    metadata and must hold at least ``fa3_fixed_split_metadata_size(requests)`` elements.
    """
    requests = query_start.numel() - 1
    rounded = _padded_requests(requests)
    size = fa3_fixed_split_metadata_size(requests)
    if out is None:
        metadata = torch.zeros(size, dtype=torch.int32, device=query_start.device)
    else:
        metadata = out[:size]
        out.zero_()
    metadata[:requests] = (query_start[1:] - query_start[:-1]) * query_heads_per_kv_head
    metadata[rounded : rounded + requests] = FA3_INVARIANT_SPLITS
    metadata[2 * rounded : 2 * rounded + requests] = kv_heads
    return metadata


@dataclass(frozen=True)
class Fa3Requests:
    """FA3 varlen requests over the same query rows, in their order: ``query_start`` ``[E + 1]`` and ``key_lengths``
    ``[E]`` (int32), and ``owner`` ``[E]``, the original request of each new one (to gather its block-table row)."""

    query_start: torch.Tensor
    key_lengths: torch.Tensor
    owner: torch.Tensor


def fa3_invariant_requests(
    query_start: Sequence[int], key_lengths: Sequence[int], window: int | None, alignment: int
) -> Fa3Requests:
    """Split each varlen request so that every row's FA3 bytes are those of the row decoded alone.

    A request's query rows are the last ``q`` of its ``k`` keys (positions ``k - q .. k - 1``). A request of several
    rows starts at a multiple of ``alignment`` positions (one tile of packed query heads), its rows before the first
    such position becoming a request of their own. With ``window``, each row at a position of at least ``window`` is a
    request of its own. Takes host values; returns CPU tensors.
    """
    query_start = np.asarray(query_start, dtype=np.int64)
    key_lengths = np.asarray(key_lengths, dtype=np.int64)
    rows = query_start[1:] - query_start[:-1]
    first = key_lengths - rows
    head = rows if window is None else np.clip(window - first, 0, rows)
    lead = np.where(rows > 1, np.minimum(head, -first % alignment), 0)
    # Per request: the rows before its first aligned position, its other rows before the window, each later row.
    has_lead, has_rest = (lead > 0).astype(np.int64), (head > lead).astype(np.int64)
    counts = has_lead + has_rest + rows - head
    owner = np.repeat(np.arange(rows.size), counts)
    index = np.arange(owner.size) - np.repeat(np.cumsum(counts) - counts, counts)
    is_lead = index < has_lead[owner]
    is_rest = ~is_lead & (index < (has_lead + has_rest)[owner])
    later = head[owner] + index - (has_lead + has_rest)[owner]
    begin = np.where(is_lead, 0, np.where(is_rest, lead[owner], later))
    stop = np.where(is_lead, lead[owner], np.where(is_rest, head[owner], later + 1))
    return Fa3Requests(
        query_start=torch.from_numpy(np.append(query_start[owner] + begin, query_start[-1])).to(torch.int32),
        key_lengths=torch.from_numpy(first[owner] + stop).to(torch.int32),
        owner=torch.from_numpy(owner),
    )


class Fa3Call(NamedTuple):
    """One FA3 call: varlen requests ``[begin, end)``, which are one-row requests when ``one_row``."""

    begin: int
    end: int
    one_row: bool


def fa3_request_calls(query_start: Sequence[int], *, one_row_calls: bool, max_requests: int) -> list[Fa3Call]:
    """Group consecutive varlen requests into FA3 calls of at most ``max_requests`` requests; with ``one_row_calls``,
    the one-row requests and the longer ones go in separate calls."""
    rows = np.diff(np.asarray(query_start, dtype=np.int64))
    one_row = rows == 1 if one_row_calls else np.zeros(rows.size, dtype=bool)
    edges = [0, *(np.flatnonzero(one_row[1:] != one_row[:-1]) + 1).tolist(), rows.size]
    return [
        Fa3Call(begin, min(begin + max_requests, run_end), bool(one_row[run_begin]))
        for run_begin, run_end in zip(edges[:-1], edges[1:], strict=True)
        for begin in range(run_begin, run_end, max_requests)
    ]


def fa3_causal_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    scheduler_metadata: torch.Tensor,
    num_splits: int,
    cu_seqlens_k: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    leftpad_k: torch.Tensor | None = None,
    softcap: float = 0.0,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    """FA3's causal varlen forward into ``out``, called through the FA3 op: vLLM's Python wrapper takes ``cu_seqlens_k``
    or ``seqused_k``, not both, and never passes ``leftpad_k``."""
    torch.ops._vllm_fa3_C.fwd(
        query,
        key,
        value,
        None,  # k_new
        None,  # v_new
        None,  # q_v
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        None,  # kv_batch_idx
        leftpad_k,
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        True,  # is_causal
        -1,  # window_size_left
        -1,  # window_size_right
        softcap,
        True,  # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        None,  # pack_gqa
        0,  # sm_margin
        s_aux,
        1,  # cp_world_size
        0,  # cp_rank
        None,  # cp_tot_seqused_k
    )
    return out


def fa3_window_start_rows(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    leftpad_k: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    softmax_scale: float,
    scheduler_metadata: torch.Tensor,
    num_splits: int,
    softcap: float = 0.0,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-row paged FA3 requests of a sliding-window layer on FA3's causal kernel, each over its window alone:
    ``leftpad_k`` holds each request's window start (its key length minus the window, at least 0).

    With the fixed-split scheduler metadata the causal kernel reads the same key blocks from the window start, in the
    same splits, as the local kernel's one-row request, so every row gets the local kernel's bytes.
    """
    if cu_seqlens_q.numel() - 1 != leftpad_k.numel():
        raise ValueError("leftpad_k needs one window start per request")
    query, key_cache, value_cache = (
        x if x.stride(-1) == 1 else x.contiguous() for x in (query, key_cache, value_cache)
    )
    return fa3_causal_varlen(
        query,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        page_table=block_table,
        leftpad_k=leftpad_k,
        softcap=softcap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
    )
