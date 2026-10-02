"""FA3 calls that give each Grug attention row the bytes of the same row in a decode step.

FA3 (vLLM's FlashAttention fork, Hopper) cuts a query tile's key blocks into splits and adds the split results, so a
row's bytes depend on the split count and on which rows share its tile. The decode-invariant vLLM engine
(``inference_engines.vllm.decode_invariant``) runs every request with ``FA3_INVARIANT_SPLITS`` splits through
precomputed scheduler metadata (``fa3_fixed_split_metadata``), starts every prefill request at a multiple of one tile
of packed query heads and runs every sliding-window row past the window as a one-row request
(``fa3_invariant_requests``), on FA3's causal kernel from the row's window start (``fa3_window_start_rows``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# FA3 (vllm-project/flash-attention 506341a1, the build vLLM 70ea9ae8f260 pins) for Grug on H100: head dim 128, bf16,
# causal or causal sliding window, packed query heads. ``tile_size_fwd_sm90`` gives 128-row query tiles.
FA3_BLOCK_M = 128
# ``prepare_varlen_num_blocks`` splits dynamically only for batches one CTA covers; ``mha_fwd`` runs a varlen call of
# more requests with one split.
FA3_DYNAMIC_SPLIT_MAX_BATCH = 992
# The split count of every FA3 request of a decode-invariant engine: FA3 cuts a query tile's key blocks ``[0, n)`` into
# runs of ``ceil(n / 4)`` blocks (``BlockMN::get_n_block_min_max``), so a row's bytes follow its own block count alone
# when every row of its tile has the same block count. Four splits run a lone decode row about as fast as FA3's
# dynamic split and cost 2-12% at 64 requests per step.
FA3_INVARIANT_SPLITS = 4


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def fa3_fixed_split_metadata(
    query_start: torch.Tensor, query_heads_per_kv_head: int, kv_heads: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    """FA3's scheduler metadata for a causal or local varlen call that runs every request with
    ``FA3_INVARIANT_SPLITS`` splits, whatever else the call holds.

    FA3's own metadata (``prepare_varlen_num_blocks``) divides the call's total key blocks over the SMs, so a request's
    split count follows the batch. The layout is ``mha_fwd``'s for packed query heads, dynamic splits and the head
    swizzle: per request its packed query rows, its split count and its heads per L2 section (a scheduling hint), each
    vector padded to a multiple of 4, then the tile semaphore, which starts at zero and which FA3's combine kernel
    zeroes after the call. A call of more than ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests runs unsplit whatever the
    metadata. ``out`` (a persistent buffer, for CUDA graphs) receives the metadata when given.
    """
    requests = query_start.numel() - 1
    rounded = _ceil_div(requests, 4) * 4
    if out is None:
        metadata = torch.zeros(3 * rounded + 1, dtype=torch.int32, device=query_start.device)
    else:
        metadata = out[: 3 * rounded + 1]
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

    A request's query rows are the last ``q`` of its ``k`` keys (positions ``k - q .. k - 1``). With a fixed split
    count FA3 cuts the key blocks of each query tile, so every row of a tile must have the tile's block count: a
    request of several rows starts at a multiple of ``alignment`` positions (one tile of packed query heads), its rows
    before the first such position becoming a request of their own. On a sliding-window layer (``window``) FA3 also
    aligns key blocks to the window start of a tile's first row, so each row at a position of at least ``window`` is a
    request of its own, as in a decode step. Takes host values; returns CPU tensors.
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


def fa3_request_calls(
    query_start: Sequence[int], *, one_row_calls: bool, max_requests: int
) -> list[tuple[int, int, bool]]:
    """Group consecutive varlen requests into FA3 calls of at most ``max_requests`` requests: ``(first, end, one_row)``
    request ranges. With ``one_row_calls`` the one-row requests and the longer ones go in separate calls (on a
    sliding-window layer the one-row requests run on FA3's causal kernel, ``fa3_window_start_rows``)."""
    rows = np.diff(np.asarray(query_start, dtype=np.int64))
    one_row = rows == 1 if one_row_calls else np.zeros(rows.size, dtype=bool)
    edges = [0, *(np.flatnonzero(one_row[1:] != one_row[:-1]) + 1).tolist(), rows.size]
    return [
        (begin, min(begin + max_requests, run_end), bool(one_row[run_begin]))
        for run_begin, run_end in zip(edges[:-1], edges[1:], strict=True)
        for begin in range(run_begin, run_end, max_requests)
    ]


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

    With the fixed-split scheduler metadata the causal kernel reads the same 128-key blocks from the window start, and
    cuts them into the same splits, as the local kernel's one-row request, so every row has the local kernel's bytes.
    It runs faster: one request past the window takes about 15 µs per call against 19 µs. vLLM's Python wrapper does
    not pass ``leftpad_k``, so this calls the FA3 op with the wrapper's other arguments.
    """
    if cu_seqlens_q.numel() - 1 != leftpad_k.numel():
        raise ValueError("leftpad_k needs one window start per request")
    query, key_cache, value_cache = (
        x if x.stride(-1) == 1 else x.contiguous() for x in (query, key_cache, value_cache)
    )
    torch.ops._vllm_fa3_C.fwd(
        query,
        key_cache,
        value_cache,
        None,  # k_new
        None,  # v_new
        None,  # q_v
        out,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,
        1,  # max_seqlen_q
        max_seqlen_k,
        block_table,
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
