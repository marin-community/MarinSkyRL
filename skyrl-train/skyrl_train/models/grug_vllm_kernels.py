"""Compiled vLLM's kernels and expert-parallel combine, which the Grug trainer runs under ``Numerics.EXACT``.

Each function computes its values as a decode-invariant engine (``inference_engines.vllm.decode_invariant``) does:
``fa3_attention_sbhd`` the attention, ``vllm_qkv_projection`` the q, k and v projections, ``vllm_topk_experts`` the
router's selection, ``vllm_expert_outputs`` the routed experts, ``vllm_ep_combine`` the expert-parallel combine and
``vllm_token_logprobs`` the log-probabilities. The combine's order reads each sequence's serving engine rank from
``serving_engine_ranks``, and ``vllm_qkv_projection`` keeps its repacked weights within ``repacked_weights``.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from skyrl_train.models.grug_fa3_invariant import (
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_causal_varlen,
    fa3_fixed_split_metadata,
)

# ``vllm_expert_outputs`` addresses each expert's weights in units of this many elements from the lowest-addressed one,
# a stride that Triton specializes as divisible by 16, as it does the stride of vLLM's own stacked expert weights.
EXPERT_OFFSET_ELEMENTS = 16
# ``_expert_offset_table``'s tables by expert-weight addresses, cleared at this many since moved parameters leave
# stale entries.
_EXPERT_OFFSET_TABLE_LIMIT = 256
_EXPERT_OFFSET_TABLES: dict[tuple, tuple[int, torch.Tensor]] = {}
# Each sequence's serving vLLM data-parallel rank and vLLM's expert-parallel size, while ``serving_engine_ranks`` holds
# them.
_SERVING_RANKS: tuple[torch.Tensor, int] | None = None
# A sequence without a serving rank: no model call produced its tokens, and no loss reads them.
NO_SERVING_RANK = -1


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
    """Sliding-window FA3 over ``requests`` sequences of ``rows`` rows, each row computed as a decode-invariant engine
    computes it.

    The first ``window`` rows of each sequence run as one request; each later row runs as a one-row request of FA3's
    causal kernel over exactly its window of keys, in calls of at most ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests.
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
                max_seqlen_q=2,  # at 1, FA3 runs the requests non-causal, which blocks the keys differently
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

    Each sequence is one varlen request, so right padding changes no valid row.
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
    """Model runner V2's log-probability (``compute_token_logprobs``) of ``token_ids[i]`` under row ``i`` of the
    ``[rows, vocab]`` ``logits``.

    Each row's value depends on that row alone. Raises ``ValueError`` unless ``logits`` is 2-D with unit vocabulary
    stride.
    """
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"vLLM's log-probability kernel reads rows of unit vocabulary stride, got {logits.stride()}")
    return compute_token_logprobs(logits, token_ids.reshape(-1, 1))[:, 0]


@contextmanager
def serving_engine_ranks(ranks: torch.Tensor, data_parallel_size: int, expert_parallel_size: int) -> Iterator[None]:
    """Give the forwards in the block each sequence's serving vLLM data-parallel rank (``[B]``, below
    ``data_parallel_size``, or ``NO_SERVING_RANK``) and vLLM's expert-parallel size, which decide the order of vLLM's
    expert-parallel combine (``vllm_ep_combine``). With one expert-parallel rank the combine is the same from every
    serving rank."""
    global _SERVING_RANKS
    if ranks.numel() and (ranks.min() < NO_SERVING_RANK or ranks.max() >= data_parallel_size):
        raise ValueError(
            f"each sequence's serving engine rank must be a vLLM data-parallel rank below {data_parallel_size} or "
            f"{NO_SERVING_RANK}, got {ranks.tolist()}"
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
    EP rank ``r`` owns experts ``[r * E / R, (r + 1) * E / R)`` and adds the token's slots it owns, in slot
    order, to an fp32 zero and rounds once; the ring reduce-scatter adds the rank partials in bf16 from
    ``home + 1`` round to ``home``.
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
    """Each dispatched row's routed-expert output, weighted by its route weight, as vLLM's ``TritonExperts`` computes
    one token-expert slot.

    ``hidden`` ``[rows, hidden]`` holds each local expert's rows in turn (``tokens_per_expert`` of them),
    ``probs`` the rows' fp32 route weights, and ``fc1_weights`` / ``fc2_weights`` each expert's
    ``[2 * ffn, hidden]`` gate-and-up and ``[hidden, ffn]`` down projection, laid out as
    ``expert_weight_offsets`` requires. A slot's bytes depend only on its row, its expert and its route weight.
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


# vLLM's q, k and v weights repacked from each fused QKV weight (by its storage and layout), while ``repacked_weights``
# holds them.
_REPACKED_QKV_WEIGHTS: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@contextmanager
def repacked_weights() -> Iterator[None]:
    """Keep the q, k and v weights ``vllm_qkv_projection`` repacks for the block, one trainer forward or
    forward-backward of unchanging parameters, and drop them before and after it."""
    _REPACKED_QKV_WEIGHTS.clear()
    try:
        yield
    finally:
        _REPACKED_QKV_WEIGHTS.clear()


def _qkv_slices(query_width: int, head_dim: int) -> tuple[slice, slice, slice]:
    """Each KV group's query, key and value rows of Megatron's fused QKV weight."""
    return (
        slice(0, query_width),
        slice(query_width, query_width + head_dim),
        slice(query_width + head_dim, query_width + 2 * head_dim),
    )


def _repacked_qkv_weights(
    fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """vLLM's contiguous q, k and v weights gathered from ``fused_weight``'s groups, repacked once per
    ``repacked_weights`` block for each fused weight."""
    key = (fused_weight.data_ptr(), tuple(fused_weight.shape), fused_weight.dtype, groups, query_width, head_dim)
    weights = _REPACKED_QKV_WEIGHTS.get(key)
    if weights is None:
        grouped = fused_weight.detach().view(groups, query_width + 2 * head_dim, -1)
        weights = tuple(grouped[:, rows].reshape(-1, grouped.shape[-1]) for rows in _qkv_slices(query_width, head_dim))
        _REPACKED_QKV_WEIGHTS[key] = weights
    return weights


class _QkvProjection(torch.autograd.Function):
    """vLLM's three GEMMs on the repacked weights, differentiated as ``F.linear`` of each part on its weight slice.

    The backward computes each part's gradients as autograd computes ``F.linear``'s, writes the weight gradients
    into the fused layout, and adds the input gradients in autograd's order: the value projection's first, the
    query projection's last.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int):
        weights = _repacked_qkv_weights(fused_weight, groups, query_width, head_dim)
        ctx.save_for_backward(x, *weights)
        ctx.layout = (groups, query_width, head_dim)
        parts = [F.linear(x, weight) for weight in weights]
        return torch.cat([part.view(*x.shape[:-1], groups, -1) for part in parts], dim=-1).view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        x, *weights = ctx.saved_tensors
        groups, query_width, head_dim = ctx.layout
        rows = x.reshape(-1, x.shape[-1])
        grouped_grad = grad.reshape(rows.shape[0], groups, query_width + 2 * head_dim)
        grad_fused = weights[0].new_zeros(groups * (query_width + 2 * head_dim), rows.shape[1])
        grouped_fused = grad_fused.view(groups, query_width + 2 * head_dim, -1)
        grad_x = None
        for part, weight in reversed(list(zip(_qkv_slices(query_width, head_dim), weights, strict=True))):
            grad_part = grouped_grad[:, :, part].reshape(rows.shape[0], -1)
            term = grad_part.mm(weight)
            grad_x = term if grad_x is None else grad_x + term
            grouped_fused[:, part] = rows.t().mm(grad_part).t().view(groups, -1, rows.shape[1])
        return grad_x.view(x.shape), grad_fused, None, None, None


def vllm_qkv_projection(
    x: torch.Tensor, fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int
) -> torch.Tensor:
    """Megatron's fused QKV projection computed as compiled vLLM computes q, k and v: three GEMMs.

    Megatron's fused weight holds, for each of ``groups`` KV groups, the group's ``query_width`` query rows, then
    its key and its value rows (``head_dim`` each). vLLM's q, k and v weights are those rows gathered across the
    groups, repacked once per ``repacked_weights`` block. The result has the fused projection's layout, so
    Megatron's split reads q, k and v from it, and its gradient is ``F.linear(x, fused_weight)``'s.
    """
    return _QkvProjection.apply(x, fused_weight, groups, query_width, head_dim)


def _expert_offset_table(weights: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest-addressed expert weight and the experts' ``expert_weight_offsets`` table on ``device``, cached by
    the weights' addresses."""
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
