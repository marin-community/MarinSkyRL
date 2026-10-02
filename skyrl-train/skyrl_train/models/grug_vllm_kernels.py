"""Compiled vLLM's kernels and expert-parallel combine, which the Grug trainer runs under
``Numerics.EXACT``.

``fa3_attention_sbhd`` runs vLLM's own FA3 forward (``vllm.vllm_flash_attn``, the build the rollout engine serves with)
on the trainer's query, key and value, computing every row as a decode-invariant engine
(``inference_engines.vllm.decode_invariant``) computes it in decode and prefill steps alike: every request with
``FA3_INVARIANT_SPLITS`` splits, and on sliding-window layers every row past the window as a one-row request.
``vllm_ep_combine`` adds each token's routed expert outputs the way vLLM's expert-parallel combine does: every EP rank
sums the token's slots it owns in fp32 and rounds once, and a bf16 ring reduction adds the rank partials, starting after
the rank that holds the token's request and ending at it. ``vllm_topk_experts`` selects each token's experts with
vLLM's call, ``vllm_expert_outputs`` computes each token-expert slot with vLLM's fused-MoE Triton kernels,
``vllm_qkv_projection`` the attention's q, k and v as vLLM's three GEMMs, and ``vllm_token_logprobs`` the
log-probabilities with model runner V2's kernel.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

from skyrl_train.models.grug_fa3_invariant import (
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_causal_varlen,
    fa3_fixed_split_metadata,
)

# ``vllm_expert_outputs`` addresses each expert's weights in units of this many elements from the lowest-addressed one.
EXPERT_OFFSET_ELEMENTS = 16
# ``_expert_offset_table``'s tables by expert-weight addresses; cleared when it holds this many (parameters that move,
# for example when the model is offloaded and reloaded, leave old entries behind).
_EXPERT_OFFSET_TABLE_LIMIT = 256
_EXPERT_OFFSET_TABLES: dict[tuple, tuple[int, torch.Tensor]] = {}
# Each sequence's serving vLLM data-parallel rank and vLLM's expert-parallel size, while ``serving_engine_ranks`` holds
# them.
_SERVING_RANKS: tuple[torch.Tensor, int] | None = None


@functools.lru_cache(maxsize=64)
def _uniform_requests(rows: int, requests: int, query_heads_per_kv_head: int, kv_heads: int, device: torch.device):
    """``requests`` varlen requests of ``rows`` rows each: their query starts and their fixed-split metadata."""
    query_start = torch.arange(0, (requests + 1) * rows, rows, dtype=torch.int32, device=device)
    return query_start, fa3_fixed_split_metadata(query_start, query_heads_per_kv_head, kv_heads)


def _fa3_forward(query, key, value, *, rows: int, requests: int, window: int | None, scale: float, out=None):
    """FA3 over ``requests`` sequences of ``rows`` rows, each one varlen request from position 0 with
    ``FA3_INVARIANT_SPLITS`` splits."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    query_start, metadata = _uniform_requests(
        rows, requests, query.shape[1] // key.shape[1], key.shape[1], query.device
    )
    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=rows,
        cu_seqlens_q=query_start,
        max_seqlen_k=rows,
        cu_seqlens_k=query_start,
        softmax_scale=scale,
        causal=True,
        window_size=None if window is None else [window - 1, 0],
        fa_version=3,
        num_splits=FA3_INVARIANT_SPLITS,
        scheduler_metadata=metadata,
        out=out,
    )


def _fa3_window_rows_forward(query, key, value, *, rows: int, requests: int, window: int, scale: float):
    """Sliding-window FA3 over ``requests`` sequences of ``rows`` rows as a decode-invariant engine computes each row:
    every request with ``FA3_INVARIANT_SPLITS`` splits, and each row past the window alone, its 128-key blocks starting
    at its own window start.

    The rows before the window read every key from position 0 and run as one local request per sequence. Each later row
    runs as a one-row request of FA3's causal kernel over exactly its window of keys: ``cu_seqlens_k`` holds the window
    starts and ``seqused_k`` the window, so the causal kernel walks the same blocks from the same first key, and cuts
    them into the same splits, as the local kernel's one-row request that a decode step runs, on 64-row tiles where the
    local kernel runs 128 (FA3 picks one MMA warpgroup for few query rows only off sliding-window layers).
    ``max_seqlen_q`` is 2: at 1, FA3 runs these requests non-causal, whose 176-key blocks group the keys otherwise. The
    later rows go in calls of at most ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests, the most FA3 splits.
    """
    device = query.device
    group, kv_heads = query.shape[1] // key.shape[1], key.shape[1]
    output = torch.empty_like(query)
    later = rows - window
    for index in range(requests):
        first = index * rows
        _fa3_forward(
            query[first : first + window],
            key[first : first + window],
            value[first : first + window],
            rows=window,
            requests=1,
            window=window,
            scale=scale,
            out=output[first : first + window],
        )
        for begin in range(0, later, FA3_DYNAMIC_SPLIT_MAX_BATCH):
            count = min(FA3_DYNAMIC_SPLIT_MAX_BATCH, later - begin)
            query_start, metadata = _uniform_requests(1, count, group, kv_heads, device)
            # Row p's keys are rows p - window + 1 .. p of the same sequence.
            window_starts = torch.arange(first + begin + 1, first + begin + count + 2, dtype=torch.int32, device=device)
            rows_out = slice(first + window + begin, first + window + begin + count)
            fa3_causal_varlen(
                query[rows_out],
                key,
                value,
                output[rows_out],
                cu_seqlens_q=query_start,  # one row per request
                cu_seqlens_k=window_starts,  # each request's first key
                seqused_k=_window_lengths(count, window, device),
                max_seqlen_q=2,
                max_seqlen_k=window,
                softmax_scale=scale,
                scheduler_metadata=metadata,
                num_splits=FA3_INVARIANT_SPLITS,
            )
    return output


@functools.lru_cache(maxsize=16)
def _window_lengths(count: int, window: int, device: torch.device) -> torch.Tensor:
    return torch.full((count,), window, dtype=torch.int32, device=device)


def fa3_attention_sbhd(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, window: int | None, scale: float
) -> torch.Tensor:
    """vLLM's FA3 forward on Megatron's ``[S, B, heads, dim]`` tensors, every row computed as a decode-invariant engine
    computes it; returns ``[S, B, heads * dim]``.

    Each sequence is one varlen request with ``FA3_INVARIANT_SPLITS`` splits: right padding changes no valid row, since a
    causal row reads no later key and its key blocks start at key 0. On a sliding-window layer each row past the window
    is a one-row request (``_fa3_window_rows_forward``).
    """
    sequence, batch, heads, head_dim = query.shape
    flat = [t.transpose(0, 1).reshape(batch * sequence, *t.shape[2:]).contiguous() for t in (query, key, value)]
    if window is not None and sequence > window:
        output = _fa3_window_rows_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale)
    else:
        output = _fa3_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale)
    return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()


def vllm_topk_experts(biased_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """The ``[rows, top_k]`` experts vLLM's Grug router selects from the fp32 biased logits, in its slot order: the
    first ``top_k`` of ``torch.topk`` of ``top_k + 1``, which breaks exact ties as the engine does."""
    return torch.topk(biased_logits, k=top_k + 1, dim=-1).indices[:, :top_k]


def vllm_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Model runner V2's log-probability of ``token_ids[i]`` under row ``i`` of the ``[rows, vocab]`` ``logits``.

    vLLM's model runner V2 computes every prompt and sampled log-probability with ``compute_token_logprobs``: one
    Triton program per row takes the row's maximum and the sum of ``exp(logit - max)`` over 1,024-wide vocabulary
    blocks, and returns ``logit - max - log(sum)``. Each row's value depends on that row alone.
    """
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"vLLM's log-probability kernel reads rows of unit vocabulary stride, got {logits.stride()}")
    return compute_token_logprobs(logits, token_ids.reshape(-1, 1))[:, 0]


@contextmanager
def serving_engine_ranks(ranks: torch.Tensor, expert_parallel_size: int) -> Iterator[None]:
    """Give the forwards in the block each sequence's serving vLLM data-parallel rank (``[B]``) and vLLM's
    expert-parallel size, which decide the order of vLLM's expert-parallel combine (``vllm_ep_combine``)."""
    global _SERVING_RANKS
    if ranks.numel() and ranks.max() >= expert_parallel_size:
        raise ValueError(
            f"each sequence's serving engine rank must be a vLLM data-parallel rank below {expert_parallel_size}, "
            f"got {ranks.tolist()}"
        )
    previous, _SERVING_RANKS = _SERVING_RANKS, (ranks, expert_parallel_size)
    try:
        yield
    finally:
        _SERVING_RANKS = previous


def serving_row_ranks(rows: int) -> tuple[torch.Tensor, int]:
    """Each of ``rows`` router rows' serving rank, and vLLM's expert-parallel size: row ``s * B + b`` holds position
    ``s`` of sequence ``b``."""
    if _SERVING_RANKS is None:
        raise RuntimeError(
            "vLLM's expert-parallel combine needs each sequence's serving engine rank (serving_engine_ranks)"
        )
    ranks, expert_parallel_size = _SERVING_RANKS
    if rows % ranks.numel():
        raise ValueError(f"{rows} router rows do not cover {ranks.numel()} sequences evenly")
    return ranks.repeat(rows // ranks.numel()), expert_parallel_size


def vllm_ep_combine(
    permuted: torch.Tensor,
    routing_map: torch.Tensor,
    selected: torch.Tensor,
    home_ranks: torch.Tensor,
    ep_size: int,
) -> torch.Tensor:
    """Each token's routed output as vLLM's all-gather/reduce-scatter expert-parallel combine adds it.

    ``permuted`` holds the weighted expert outputs in Megatron's permuted order (by expert, then token);
    ``routing_map`` is the ``[tokens, experts]`` selection, ``selected`` the ``[tokens, top_k]`` experts in
    vLLM's slot order and ``home_ranks`` the vLLM data-parallel rank holding each token's request. vLLM's
    EP rank ``r`` owns experts ``[r * E / R, (r + 1) * E / R)``; its ``moe_sum`` adds the token's slots it
    owns, in slot order, to an fp32 zero and rounds once. NCCL's ring reduce-scatter (a ring reduce per
    rank when the ranks' token counts differ) adds the rank partials in bf16 from ``home + 1`` round to
    ``home``; a rank that owns none of the token's experts adds an exact zero.

    The permuted rows run by expert, then by token, so a token-expert pair's row is the number of selected
    pairs at or before it in that order, less one: one cumulative sum over the expert-major routing map, which
    never waits on the device. On CUDA one Triton kernel adds every token's slots (``grug_ep_combine_kernel``).
    """
    tokens, experts = routing_map.shape
    top_k = selected.shape[1]
    if experts % ep_size:
        raise ValueError(f"{experts} experts do not split over {ep_size} vLLM EP ranks")
    if permuted.shape[0] != tokens * top_k:
        raise ValueError(f"{permuted.shape[0]} permuted rows for {tokens} tokens of {top_k} slots")
    if tokens == 0:
        return permuted.new_empty(0, permuted.shape[1])
    pairs_through = routing_map.t().contiguous().view(-1).cumsum(0, dtype=torch.int32).view(experts, tokens)
    if permuted.is_cuda:
        from skyrl_train.models.grug_ep_combine_kernel import ep_combine

        return ep_combine(permuted, selected.long(), home_ranks.long(), pairs_through, ep_size)
    selected = selected.long()
    slot_rows = pairs_through[selected, torch.arange(tokens).view(-1, 1)] - 1
    ring_positions = (selected // (experts // ep_size) - home_ranks.long().view(-1, 1) - 1) % ep_size
    values = permuted[slot_rows].float()
    total = torch.zeros(tokens, permuted.shape[1], dtype=torch.float32, device=permuted.device)
    for position in range(ep_size):
        partial = torch.zeros_like(total)
        for slot in range(top_k):
            on_rank = (ring_positions[:, slot] == position).view(-1, 1)
            partial = torch.where(on_rank, partial + values[:, slot], partial)
        total = (total + partial.to(permuted.dtype).float()).to(permuted.dtype).float()
    return total.to(permuted.dtype)


def vllm_expert_outputs(
    hidden: torch.Tensor,
    tokens_per_expert: Sequence[int],
    probs: torch.Tensor,
    fc1_weights: Sequence[torch.Tensor],
    fc2_weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Each dispatched row's routed-expert output as vLLM's ``TritonExperts`` computes one token-expert slot.

    ``hidden`` ``[rows, hidden]`` holds each local expert's rows in turn (``tokens_per_expert`` of them),
    ``probs`` the rows' fp32 route weights, and ``fc1_weights`` / ``fc2_weights`` each expert's
    ``[2 * ffn, hidden]`` gate-and-up and ``[hidden, ffn]`` down projection. As ``TritonExperts.apply`` does,
    this aligns the rows by expert (``moe_align_block_size``), runs ``fused_moe_kernel`` for gate-and-up,
    ``silu_and_mul``, and ``fused_moe_kernel`` for the down projection with the route weight multiplied into the
    fp32 accumulator before one bf16 rounding, all with vLLM's launch config for the shapes. Every row is one
    slot (``top_k`` 1). A slot's bytes depend only on its row, its expert and its weight: the kernel adds the K
    blocks in order into one fp32 accumulator without split-K, whatever the config or the other rows.

    The kernel reads the experts' weights where the trainer keeps them, one tensor per expert: it addresses a
    block's expert at ``B + expert_ids[block] * B.stride(0)``, so each block gets its expert's offset from the
    lowest-addressed expert weight, in units of ``EXPERT_OFFSET_ELEMENTS``, as its expert id, with that unit as
    the expert stride. Triton specializes a stride divisible by 16, so it can assume the same weight-address
    alignment as for vLLM's own stacked weights.
    """
    from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
    from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    rows, width = hidden.shape
    experts = len(fc1_weights)
    gate_up_width = fc1_weights[0].shape[0]
    if probs.dtype != torch.float32 or probs.shape != (rows,):
        raise ValueError(
            f"vllm_expert_outputs needs one fp32 route weight per row, got {probs.dtype}{tuple(probs.shape)}"
        )
    if len(fc2_weights) != experts or len(tokens_per_expert) != experts:
        raise ValueError("vllm_expert_outputs needs one fc1 weight, one fc2 weight and one row count per expert")
    output = hidden.new_empty(rows, width)
    if rows == 0:
        return output
    config = try_get_optimal_moe_config(
        (experts, gate_up_width, width),
        (experts, width, gate_up_width // 2),
        1,
        FUSED_MOE_UNQUANTIZED_CONFIG.config_name(hidden.dtype),
        rows,
    )
    counts = torch.as_tensor(tokens_per_expert, dtype=torch.long, device="cpu")
    expert_of_row = torch.repeat_interleave(torch.arange(experts, dtype=torch.int32), counts)
    expert_of_row = expert_of_row.pin_memory().to(hidden.device, non_blocking=True).view(rows, 1)
    sorted_rows, block_experts, padded_rows = moe_align_block_size(expert_of_row, config["BLOCK_SIZE_M"], experts)
    gate_up = hidden.new_empty(rows, 1, gate_up_width)
    _fused_moe_gemm(hidden.contiguous(), fc1_weights, gate_up, None, sorted_rows, block_experts, padded_rows, config)
    activation = hidden.new_empty(rows, gate_up_width // 2)
    torch.ops._C.silu_and_mul(activation, gate_up.view(rows, gate_up_width))
    down = output.view(rows, 1, width)
    _fused_moe_gemm(activation, fc2_weights, down, probs.view(rows, 1), sorted_rows, block_experts, padded_rows, config)
    return output


def expert_weight_offsets(weights: Sequence[torch.Tensor], base: torch.Tensor) -> list[int]:
    """Each weight's address minus ``base``'s, in units of ``EXPERT_OFFSET_ELEMENTS`` elements.

    ``base`` must be the lowest-addressed weight; every weight must be contiguous with ``base``'s shape and dtype
    and sit a whole number of units above it, so that element ``offset * EXPERT_OFFSET_ELEMENTS`` counted from
    ``base``'s first element is the weight's first element.
    """
    if any(
        weight.shape != base.shape or weight.dtype != base.dtype or not weight.is_contiguous() for weight in weights
    ):
        raise ValueError("vllm_expert_outputs needs contiguous expert weights of one shape and dtype")
    unit = EXPERT_OFFSET_ELEMENTS * base.element_size()
    distances = [weight.data_ptr() - base.data_ptr() for weight in weights]
    if any(distance < 0 or distance % unit for distance in distances):
        raise ValueError(f"vllm_expert_outputs needs expert weights at whole {unit}-byte steps above the lowest one")
    return [distance // unit for distance in distances]


def vllm_qkv_projection(
    x: torch.Tensor, fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int
) -> torch.Tensor:
    """Megatron's fused QKV projection computed as compiled vLLM computes q, k and v: three GEMMs.

    Megatron's fused weight holds, for each of ``groups`` KV groups, the group's ``query_width`` query rows, then
    its key and its value rows (``head_dim`` each). vLLM's q, k and v weights are those rows gathered across the
    groups. The result has the fused projection's layout, so Megatron's split reads q, k and v from it.
    """
    grouped = fused_weight.view(groups, query_width + 2 * head_dim, -1)
    weights = (
        grouped[:, :query_width],
        grouped[:, query_width : query_width + head_dim],
        grouped[:, query_width + head_dim :],
    )
    parts = [torch.nn.functional.linear(x, weight.reshape(-1, weight.shape[-1])) for weight in weights]
    return torch.cat([part.view(*x.shape[:-1], groups, -1) for part in parts], dim=-1).view(*x.shape[:-1], -1)


def _expert_offset_table(weights: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest-addressed expert weight and the experts' ``expert_weight_offsets`` table on ``device``.

    The table depends only on the weights' addresses, which the trainer's expert parameters keep from call to call,
    so it is built once per set of addresses (and again after the parameters move).
    """
    addresses = tuple(weight.data_ptr() for weight in weights)
    key = (device, addresses, tuple(weights[0].shape), weights[0].dtype)
    cached = _EXPERT_OFFSET_TABLES.get(key)
    if cached is None:
        lowest = min(range(len(weights)), key=addresses.__getitem__)
        table = torch.tensor(expert_weight_offsets(weights, weights[lowest]), dtype=torch.int64, device=device)
        if len(_EXPERT_OFFSET_TABLES) >= _EXPERT_OFFSET_TABLE_LIMIT:
            _EXPERT_OFFSET_TABLES.clear()
        cached = _EXPERT_OFFSET_TABLES[key] = (lowest, table)
    lowest, table = cached
    return weights[lowest], table


def _fused_moe_gemm(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    output: torch.Tensor,
    route_weights: torch.Tensor | None,
    sorted_rows: torch.Tensor,
    block_experts: torch.Tensor,
    padded_rows: torch.Tensor,
    config: dict,
) -> None:
    """vLLM's ``invoke_fused_moe_triton_kernel`` on per-expert weight tensors (see ``vllm_expert_outputs``)."""
    from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel
    from vllm.triton_utils import tl

    base, table = _expert_offset_table(weights, inputs.device)
    rows, columns = base.shape
    # Blocks past the padded row count keep whatever moe_align_block_size left there; the kernel returns before
    # reading their expert, so any in-range value serves.
    block_offsets = table[block_experts.long().clamp(0, len(weights) - 1)]
    stacked = torch.as_strided(base.detach(), (1, rows, columns), (EXPERT_OFFSET_ELEMENTS, columns, 1))
    invoke_fused_moe_triton_kernel(
        inputs,
        stacked,
        output,
        None,
        None,
        route_weights,
        sorted_rows,
        block_offsets,
        padded_rows,
        route_weights is not None,
        1,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
