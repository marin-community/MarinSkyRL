"""Compiled vLLM's kernels and expert-parallel combine, for the Grug trainer's numerics flags.

``fa3_attention`` runs vLLM's own FA3 forward (``vllm.vllm_flash_attn``, the build the rollout engine
serves with) on the trainer's query, key and value. Its bytes depend on how many key-block splits FA3
uses for a request, which vLLM decides per engine step (``fa3_split_counts``). ``ep_sum`` adds each
token's routed expert outputs the way vLLM's expert-parallel combine does: every EP rank sums the
token's slots it owns in fp32 and rounds once, and a bf16 ring reduction adds the rank partials,
starting after the rank that holds the token's request and ending at it. ``vllm_experts`` computes each
token-expert slot with vLLM's fused-MoE Triton kernels (``vllm_expert_outputs``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

# FA3 (vllm-project/flash-attention 506341a1, the build vLLM 70ea9ae8f260 pins) for Grug on H100: head dim
# 128, bf16, causal or causal sliding window, packed query heads. ``tile_size_fwd_sm90`` gives 128 x 128
# tiles, or 64-row tiles when one MMA warpgroup is used (full attention with at most 64 packed rows).
FA3_BLOCK_M = 128
FA3_BLOCK_M_ONE_WARPGROUP = 64
FA3_BLOCK_N = 128
# ``prepare_varlen_num_blocks`` splits dynamically only for batches one CTA covers.
FA3_DYNAMIC_SPLIT_MAX_BATCH = 992
# ``attention_config.flash_attn_max_num_splits_for_cuda_graph``: the split cap vLLM passes on CUDA-graph steps.
FA3_MAX_SPLITS_FOR_CUDA_GRAPH = 32
# The probe engines' largest CUDA-graph capture size (``max_cudagraph_capture_size``).
VLLM_MAX_CUDA_GRAPH_TOKENS = 512
H100_SMS = 132
# ``vllm_experts`` addresses each expert's weights in units of this many elements from the lowest-addressed one.
EXPERT_OFFSET_ELEMENTS = 16
# The probe engines' ``max_num_batched_tokens``: the rows of a full vLLM prefill step.
VLLM_MAX_BATCHED_TOKENS = 8192


@dataclass(frozen=True)
class Fa3Request:
    """One request of an engine step: its query rows in the step and its key length after the step."""

    query_tokens: int
    key_tokens: int


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _heuristic_splits(total_mblocks: int, num_sms: int, num_n_blocks: int) -> int:
    """FA3's ``num_splits_heuristic`` for causal or local attention, in its fp32 arithmetic."""
    if total_mblocks >= np.float32(0.8) * np.float32(num_sms):
        return 1
    if num_n_blocks <= 4:
        return 1
    efficiency = []
    for splits in range(1, min(128, num_sms, num_n_blocks) + 1):
        waves = np.float32(np.float32(total_mblocks * splits) / np.float32(num_sms))
        efficiency.append(float(np.float32(waves / np.float32(math.ceil(waves)))))
    best = max(efficiency)
    return next(splits for splits, value in enumerate(efficiency, start=1) if value >= 0.85 * best)


def fa3_split_counts(
    step: Sequence[Fa3Request],
    *,
    kv_heads: int,
    query_heads_per_kv_head: int,
    window: int | None,
    step_tokens: int | None = None,
    num_sms: int = H100_SMS,
) -> list[int]:
    """How many key-block splits vLLM's FA3 forward uses for each request of one engine step and layer type.

    vLLM passes ``num_splits=32`` on steps of at most 512 scheduled tokens and 0 above
    (``FlashAttentionMetadataBuilder.build``). FA3 turns 0 into a host heuristic on the step's longest
    query and key (``get_num_splits``); a count above one then becomes a per-request count on the device,
    from the step's total query-block x key-block count (``prepare_varlen_num_blocks_kernel``). vLLM's
    build always packs query heads. ``window`` is the sliding window of the layer (``None``: full).
    """
    if not step:
        return []
    if len(step) > FA3_DYNAMIC_SPLIT_MAX_BATCH:
        return [1] * len(step)
    step_tokens = sum(request.query_tokens for request in step) if step_tokens is None else step_tokens
    max_query = max(request.query_tokens for request in step)
    max_key = max(request.key_tokens for request in step)
    packed = query_heads_per_kv_head
    block_m = FA3_BLOCK_M_ONE_WARPGROUP if window is None and max_query * packed <= 64 else FA3_BLOCK_M
    if step_tokens <= VLLM_MAX_CUDA_GRAPH_TOKENS:
        static = FA3_MAX_SPLITS_FOR_CUDA_GRAPH
    else:
        loaded = max_key if window is None else max(0, min(max_key, window + block_m))
        static = _heuristic_splits(
            kv_heads * _ceil_div(max_query * packed, block_m), num_sms, _ceil_div(loaded, FA3_BLOCK_N)
        )
    if static <= 1:
        return [1] * len(step)
    blocks = [
        (_ceil_div(request.query_tokens * packed, block_m), _ceil_div(request.key_tokens, FA3_BLOCK_N))
        for request in step
    ]
    total = np.float32(sum(m * n for m, n in blocks))
    per_sm = math.ceil(np.float32(np.float32(total * np.float32(1.1)) * np.float32(kv_heads)) / np.float32(num_sms))
    return [max(min(_ceil_div(n, per_sm), static), 1) for _, n in blocks]


@dataclass(frozen=True)
class Fa3Plan:
    """The FA3 split count and valid length of each sequence of a scoring micro-batch, in batch order."""

    splits: tuple[int, ...]
    lengths: tuple[int, ...]


_plan: Fa3Plan | None = None


@contextmanager
def fa3_attention_plan(splits: Sequence[int], lengths: Sequence[int]) -> Iterator[None]:
    """Give ``fa3_attention`` the split count vLLM used for each sequence of the enclosed scoring forwards.

    Without a plan every sequence runs unsplit, which is what vLLM does on steps that fill the GPU.
    """
    global _plan
    if len(splits) != len(lengths):
        raise ValueError("fa3_attention plan needs one split count per sequence length")
    previous, _plan = _plan, Fa3Plan(tuple(int(v) for v in splits), tuple(int(v) for v in lengths))
    try:
        yield
    finally:
        _plan = previous


def _fa3_forward(query, key, value, *, rows: int, requests: int, window: int | None, scale: float, splits: int):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    cu_seqlens = torch.arange(0, (requests + 1) * rows, rows, dtype=torch.int32, device=query.device)
    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=rows,
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=rows,
        cu_seqlens_k=cu_seqlens,
        softmax_scale=scale,
        causal=True,
        window_size=None if window is None else [window - 1, 0],
        fa_version=3,
        num_splits=splits,
    )


def fa3_attention_sbhd(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, window: int | None, scale: float
) -> torch.Tensor:
    """vLLM's FA3 forward on Megatron's ``[S, B, heads, dim]`` tensors; returns ``[S, B, heads * dim]``.

    Without a plan each sequence is one unsplit varlen request of ``S`` rows: right padding changes no
    valid row, since a causal row reads no later key and its key blocks start at key 0. With a plan
    (``fa3_attention_plan``) each sequence runs alone on its valid rows with vLLM's split count.
    """
    sequence, batch, heads, head_dim = query.shape
    flat = [t.transpose(0, 1).reshape(batch * sequence, *t.shape[2:]).contiguous() for t in (query, key, value)]
    if _plan is None:
        output = _fa3_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale, splits=1)
        return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()
    if torch.is_grad_enabled():
        raise NotImplementedError("fa3_attention split plans are for scoring forwards; training runs unsplit")
    if len(_plan.splits) != batch:
        raise ValueError(f"fa3_attention plan has {len(_plan.splits)} sequences for a micro-batch of {batch}")
    output = torch.zeros(batch, sequence, heads, head_dim, dtype=query.dtype, device=query.device)
    for index, (splits, length) in enumerate(zip(_plan.splits, _plan.lengths, strict=True)):
        rows = [t[index * sequence : index * sequence + length] for t in flat]
        output[index, :length] = _fa3_forward(*rows, rows=length, requests=1, window=window, scale=scale, splits=splits)
    return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()


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
    """
    tokens, experts = routing_map.shape
    top_k = selected.shape[1]
    if experts % ep_size:
        raise ValueError(f"{experts} experts do not split over {ep_size} vLLM EP ranks")
    pairs = routing_map.t().nonzero()
    if pairs.shape[0] != permuted.shape[0]:
        raise ValueError(f"{permuted.shape[0]} permuted rows for {pairs.shape[0]} selected (token, expert) pairs")
    row_of = torch.full((tokens, experts), -1, dtype=torch.long, device=permuted.device)
    row_of[pairs[:, 1], pairs[:, 0]] = torch.arange(pairs.shape[0], device=permuted.device)
    slot_rows = row_of.gather(1, selected.long())
    if (slot_rows < 0).any():
        raise ValueError("ep_sum slot order names an expert the routing map did not select")
    ring_position = (selected.long() // (experts // ep_size) - home_ranks.long().view(-1, 1) - 1) % ep_size
    # A stable order by ring position keeps vLLM's slot order inside each rank's partial.
    order = torch.sort(ring_position * top_k + torch.arange(top_k, device=selected.device), dim=1).indices
    position, rows = ring_position.gather(1, order), slot_rows.gather(1, order)
    zero = torch.zeros(tokens, permuted.shape[1], dtype=torch.float32, device=permuted.device)
    total = zero.to(permuted.dtype)
    partial = zero + permuted[rows[:, 0]].float()
    for slot in range(1, top_k):
        value = permuted[rows[:, slot]].float()
        new_rank = (position[:, slot] != position[:, slot - 1]).view(-1, 1)
        closed = (total.float() + partial.to(permuted.dtype).float()).to(permuted.dtype)
        total = torch.where(new_rank, closed, total)
        partial = torch.where(new_rank, zero + value, partial + value)
    return (total.float() + partial.to(permuted.dtype).float()).to(permuted.dtype)


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
        raise ValueError(f"vllm_experts needs one fp32 route weight per row, got {probs.dtype}{tuple(probs.shape)}")
    if len(fc2_weights) != experts or len(tokens_per_expert) != experts:
        raise ValueError("vllm_experts needs one fc1 weight, one fc2 weight and one row count per expert")
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
        raise ValueError("vllm_experts needs contiguous expert weights of one shape and dtype")
    unit = EXPERT_OFFSET_ELEMENTS * base.element_size()
    distances = [weight.data_ptr() - base.data_ptr() for weight in weights]
    if any(distance < 0 or distance % unit for distance in distances):
        raise ValueError(f"vllm_experts needs expert weights at whole {unit}-byte steps above the lowest one")
    return [distance // unit for distance in distances]


def fixed_rows_linear(x: torch.Tensor, weight: torch.Tensor, rows: int) -> torch.Tensor:
    """``F.linear(x, weight)`` computed in calls of exactly ``rows`` rows, the last one zero-padded.

    A GEMM library picks its kernel, and with it the order it sums each output in, from the call's row count, so
    this gives every row the bytes of a ``rows``-row call whatever the batch holds.
    """
    flat = x.reshape(-1, x.shape[-1])
    outputs = []
    for chunk in flat.split(rows):
        padded = chunk.new_zeros(rows, chunk.shape[1])
        padded[: chunk.shape[0]] = chunk
        outputs.append(torch.nn.functional.linear(padded, weight)[: chunk.shape[0]])
    return torch.cat(outputs).view(*x.shape[:-1], weight.shape[0])


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

    base = min(weights, key=lambda weight: weight.data_ptr())
    rows, columns = base.shape
    table = torch.tensor(expert_weight_offsets(weights, base), dtype=torch.int64)
    table = table.pin_memory().to(inputs.device, non_blocking=True)
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
