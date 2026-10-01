"""Compiled vLLM's kernels and expert-parallel combine, for the Grug trainer's numerics flags.

``fa3_attention`` runs vLLM's own FA3 forward (``vllm.vllm_flash_attn``, the build the rollout engine
serves with) on the trainer's query, key and value. Its bytes depend on how many key-block splits FA3
uses for a request, which vLLM decides per engine step (``fa3_split_counts``). ``ep_sum`` adds each
token's routed expert outputs the way vLLM's expert-parallel combine does: every EP rank sums the
token's slots it owns in fp32 and rounds once, and a bf16 ring reduction adds the rank partials,
starting after the rank that holds the token's request and ending at it. ``vllm_experts`` computes each
token-expert slot with vLLM's fused-MoE Triton kernels (``vllm_expert_outputs``).

A decode-invariant engine (``inference_engines.vllm.decode_invariant``) runs FA3 with one split and every
sliding-window row past the window as a one-row request (``window_row_requests``); ``fa3_window_rows`` computes the
trainer's rows that way.

``vllm_steps`` reproduces the engine step of a logged re-read: each sequence's prefill ran alone in one vLLM step
(``VllmStep``), which fixes its FA3 split counts, the row count of the fp32 router GEMM, and the row counts of the
LM-head GEMMs that vLLM's model runner V2 runs for prompt log-probabilities (1,024-row chunks) and for the sampled
position (one row). ``vllm_token_logprobs`` is model runner V2's log-probability kernel.
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
# ``_expert_offset_table``'s tables by expert-weight addresses; cleared when it holds this many (parameters that move,
# for example when the model is offloaded and reloaded, leave old entries behind).
_EXPERT_OFFSET_TABLE_LIMIT = 256
_EXPERT_OFFSET_TABLES: dict[tuple, tuple[int, torch.Tensor]] = {}
# The probe engines' ``max_num_batched_tokens``: the rows of a full vLLM prefill step.
VLLM_MAX_BATCHED_TOKENS = 8192
# Model runner V2 computes prompt log-probabilities from LM-head calls of this many rows
# (``compute_prompt_logprobs_with_chunking``).
VLLM_PROMPT_LOGPROB_CHUNK = 1024


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
class VllmStep:
    """The vLLM engine step that ran one sequence's re-read prefill, the sequence alone in the step.

    ``tokens`` is the prefix the step scheduled (the sequence's tokens without its last) and ``rows`` the row count
    the step's model forward ran: ``tokens`` padded to a CUDA-graph capture size, or ``tokens`` itself above the
    largest. ``tokens == 0`` marks a batch row without a logged step (padding past the probe's samples).
    """

    tokens: int
    rows: int


_steps: tuple[VllmStep, ...] | None = None


@contextmanager
def vllm_step_plan(steps: Sequence[VllmStep]) -> Iterator[None]:
    """Give ``vllm_steps`` the logged vLLM step of each sequence of the enclosed forward, in micro-batch order."""
    global _steps
    previous, _steps = _steps, tuple(steps)
    try:
        yield
    finally:
        _steps = previous


def planned_vllm_steps(batch: int) -> tuple[VllmStep, ...]:
    """The step plan of the current scoring forward, which must hold one step per sequence of its micro-batch."""
    if _steps is None:
        raise RuntimeError("vllm_steps numerics need each sequence's logged vLLM step (a re-read replay mode)")
    if torch.is_grad_enabled():
        raise NotImplementedError("vllm_steps numerics reproduce a logged scoring step; a training forward has none")
    if len(_steps) != batch:
        raise ValueError(f"the vLLM step plan holds {len(_steps)} sequences for a micro-batch of {batch}")
    return _steps


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


@dataclass(frozen=True)
class WindowRowRequests:
    """FA3 varlen requests over the same query rows: each request's rows before the window, then one per later row.

    ``query_start`` ``[E + 1]`` and ``key_lengths`` ``[E]`` are int32; ``owner`` ``[E]`` is the original request of each
    new request, to gather its block-table row. The query rows keep their order, so the output rows do too.
    """

    query_start: torch.Tensor
    key_lengths: torch.Tensor
    owner: torch.Tensor


def window_row_requests(query_start: torch.Tensor, key_lengths: torch.Tensor, window: int) -> WindowRowRequests:
    """Split each varlen request so every query row at a key position of at least ``window`` is a request of its own.

    A request's query rows are the last ``q`` of its ``k`` keys (positions ``k - q .. k - 1``). On a sliding-window
    layer FA3 aligns key blocks to the window start of a query tile's first row; a lone row is its own tile, as in a
    decode step. Rows before position ``window`` read every key from position 0, so their tiles need no split.
    Waits for the device once (the number of new requests).
    """
    query_start = query_start.long()
    requests = query_start.numel() - 1
    key_lengths = key_lengths[:requests].long()
    query_lengths = query_start[1:] - query_start[:-1]
    first_position = key_lengths - query_lengths
    # Rows before the window stay one request; each later row becomes one. A request without rows before the window
    # contributes only its one-row requests.
    head = torch.minimum((window - first_position).clamp(min=0), query_lengths)
    has_head = (head > 0).long()
    counts = has_head + query_lengths - head
    total = int(counts.sum())
    device = query_start.device
    owner = torch.repeat_interleave(torch.arange(requests, device=device), counts, output_size=total)
    # The new request's index among its owner's: -1 is the rows before the window, i >= 0 the i-th later row.
    later_row = torch.arange(total, device=device) - (torch.cumsum(counts, 0) - counts)[owner] - has_head[owner]
    before = later_row < 0
    starts = query_start[owner] + torch.where(before, 0, head[owner] + later_row)
    keys = first_position[owner] + head[owner] + torch.where(before, 0, later_row + 1)
    return WindowRowRequests(
        query_start=torch.cat([starts, query_start[-1:]]).to(torch.int32),
        key_lengths=keys.to(torch.int32),
        owner=owner,
    )


def _fa3_window_rows_forward(query, key, value, *, rows: int, requests: int, window: int, scale: float):
    """Sliding-window FA3 over ``requests`` sequences of ``rows`` rows, each row past the window computed as a
    decode-invariant engine computes it: alone, its 128-key blocks starting at its own window start.

    The rows before the window read every key from position 0 and run as one local request per sequence. Each later row
    runs as a one-row request of FA3's causal kernel over exactly its window of keys: ``cu_seqlens_k`` holds the window
    starts and ``seqused_k`` the window, so the causal kernel walks the same blocks from the same first key as the
    local kernel's one-row request that a decode step runs, on 64-row tiles where the local kernel runs 128 (FA3 picks
    one MMA warpgroup for few query rows only off sliding-window layers). ``max_seqlen_q`` is 2: at 1, FA3 runs these
    requests non-causal, whose 176-key blocks group the keys otherwise. The bytes equal the one-row local requests' on
    every row (harness ``fa3_window_check``). vLLM's Python wrapper accepts ``cu_seqlens_k`` or ``seqused_k``, not both,
    so the causal requests call the FA3 op with the wrapper's arguments.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = query.device
    output = torch.empty_like(query)
    heads_out = torch.arange(0, 2 * window, window, dtype=torch.int32, device=device)
    later = rows - window
    later_queries = torch.arange(later + 1, dtype=torch.int32, device=device)
    seqused_k = torch.full((later,), window, dtype=torch.int32, device=device)
    for index in range(requests):
        first = index * rows
        flash_attn_varlen_func(
            q=query[first : first + window],
            k=key[first : first + window],
            v=value[first : first + window],
            max_seqlen_q=window,
            cu_seqlens_q=heads_out,
            max_seqlen_k=window,
            cu_seqlens_k=heads_out,
            softmax_scale=scale,
            causal=True,
            window_size=[window - 1, 0],
            fa_version=3,
            num_splits=1,
            out=output[first : first + window],
        )
        # Row p's keys are rows p - window + 1 .. p of the same sequence.
        window_starts = torch.arange(first + 1, first + later + 2, dtype=torch.int32, device=device)
        torch.ops._vllm_fa3_C.fwd(
            query[first + window : first + rows],
            key,
            value,
            None,  # k_new
            None,  # v_new
            None,  # q_v
            output[first + window : first + rows],
            later_queries,  # cu_seqlens_q: one row per request
            window_starts,  # cu_seqlens_k: each request's first key
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            seqused_k,
            2,  # max_seqlen_q
            window,  # max_seqlen_k
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            scale,
            True,  # is_causal
            -1,  # window_size_left
            -1,  # window_size_right
            0.0,  # softcap
            True,  # is_rotary_interleaved
            None,  # scheduler_metadata
            1,  # num_splits
            None,  # pack_gqa
            0,  # sm_margin
            None,  # s_aux
            1,  # cp_world_size
            0,  # cp_rank
            None,  # cp_tot_seqused_k
        )
    return output


def fa3_attention_sbhd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int | None,
    scale: float,
    steps: Sequence[VllmStep] | None = None,
    window_rows: bool = False,
) -> torch.Tensor:
    """vLLM's FA3 forward on Megatron's ``[S, B, heads, dim]`` tensors; returns ``[S, B, heads * dim]``.

    Without ``steps`` each sequence is one unsplit varlen request of ``S`` rows: right padding changes no
    valid row, since a causal row reads no later key and its key blocks start at key 0. With ``steps`` each
    sequence runs alone on the prefix its vLLM step scheduled, with the split count FA3 chose for a request
    alone in a step of that many tokens (``fa3_split_counts``); its later rows stay zero, as no logged
    position reads them. A sequence without a step (``tokens == 0``) runs unsplit on all ``S`` rows. With
    ``window_rows`` a sliding-window layer runs each row past the window as a one-row request
    (``_fa3_window_rows_forward``), as a decode-invariant engine does.
    """
    sequence, batch, heads, head_dim = query.shape
    flat = [t.transpose(0, 1).reshape(batch * sequence, *t.shape[2:]).contiguous() for t in (query, key, value)]
    if window_rows and steps is not None:
        raise NotImplementedError("fa3_window_rows numerics do not combine with a logged step plan")
    if window_rows and window is not None and sequence > window:
        output = _fa3_window_rows_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale)
        return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()
    if steps is None:
        output = _fa3_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale, splits=1)
        return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()
    kv_heads = key.shape[2]
    output = torch.zeros(batch, sequence, heads, head_dim, dtype=query.dtype, device=query.device)
    for index, step in enumerate(steps):
        length = step.tokens or sequence
        splits = 1
        if step.tokens:
            (splits,) = fa3_split_counts(
                [Fa3Request(step.tokens, step.tokens)],
                kv_heads=kv_heads,
                query_heads_per_kv_head=heads // kv_heads,
                window=window,
                step_tokens=step.tokens,
            )
        rows = [t[index * sequence : index * sequence + length] for t in flat]
        output[index, :length] = _fa3_forward(*rows, rows=length, requests=1, window=window, scale=scale, splits=splits)
    return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()


def step_rows_linear(x: torch.Tensor, weight: torch.Tensor, steps: Sequence[VllmStep]) -> torch.Tensor:
    """``F.linear(x, weight)`` of ``[S, B, K]`` rows, each sequence's rows computed as its vLLM step computed them.

    A GEMM library picks its kernel, and with it each output's summation order, from the call's row count. vLLM ran
    sequence ``b`` alone in a step of ``steps[b].rows`` rows with its tokens first, so its first rows run here as one
    call of exactly that many rows, zero rows after its own. Rows past ``steps[b].rows``, which no logged token reads,
    and sequences without a step come from one plain call.
    """
    output = torch.nn.functional.linear(x, weight)
    for index, step in enumerate(steps):
        if not step.rows:
            continue
        used = min(step.rows, x.shape[0])
        padded = x.new_zeros(step.rows, x.shape[-1])
        padded[:used] = x[:used, index]
        output[:used, index] = torch.nn.functional.linear(padded, weight)[:used]
    return output


def step_lm_head_logits(hidden: torch.Tensor, weight: torch.Tensor, steps: Sequence[VllmStep]) -> torch.Tensor:
    """LM-head logits of ``[S, B, H]`` hidden states with each logged position computed as vLLM's step computed it.

    For a step of ``t`` tokens vLLM's model runner V2 takes prompt log-probabilities from LM-head calls on rows
    ``0 .. t - 1`` in chunks of ``VLLM_PROMPT_LOGPROB_CHUNK`` rows, and the sampled position's log-probability from a
    one-row call on row ``t - 1``. Rows ``0 .. t - 2`` hold the chunks' logits and row ``t - 1`` the one-row call's;
    every other row comes from one plain call.
    """
    output = torch.nn.functional.linear(hidden, weight)
    for index, step in enumerate(steps):
        tokens = step.tokens
        if not tokens:
            continue
        rows = hidden[:, index]
        for start in range(0, tokens, VLLM_PROMPT_LOGPROB_CHUNK):
            end = min(start + VLLM_PROMPT_LOGPROB_CHUNK, tokens)
            output[start:end, index] = torch.nn.functional.linear(rows[start:end].contiguous(), weight)
        output[tokens - 1, index] = torch.nn.functional.linear(rows[tokens - 1 : tokens].contiguous(), weight)[0]
    return output


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
