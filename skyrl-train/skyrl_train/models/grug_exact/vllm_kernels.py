"""vLLM's attention, fused-MoE, expert-selection and log-probability kernels, called as a decode-invariant engine
calls them."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch

from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import (
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_causal_varlen,
    fa3_fixed_split_metadata,
)

HOPPER_COMPUTE_CAPABILITY_MAJOR = 9
# ``expert_outputs`` addresses each expert's weights in units of this many elements from the lowest-addressed one, a
# stride that Triton specializes as divisible by 16, as it does the stride of vLLM's own stacked expert weights.
EXPERT_OFFSET_ELEMENTS = 16


def require_hopper_gpu() -> None:
    """Raise unless this process's GPU is a Hopper, which the FA3 kernels run on."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != HOPPER_COMPUTE_CAPABILITY_MAJOR:
        name = torch.cuda.get_device_name() if torch.cuda.is_available() else "no GPU"
        raise RuntimeError(f"exact numerics run on Hopper GPUs (FA3); this process has {name}")


@functools.lru_cache(maxsize=64)
def _uniform_requests(rows: int, requests: int, query_heads_per_kv_head: int, kv_heads: int, device: torch.device):
    """``requests`` varlen requests of ``rows`` rows each: their query starts and their fixed-split metadata."""
    query_start = torch.arange(0, (requests + 1) * rows, rows, dtype=torch.int32, device=device)
    return query_start, fa3_fixed_split_metadata(query_start, query_heads_per_kv_head, kv_heads)


def _fa3_forward(query, key, value, *, rows: int, requests: int, window: int | None, scale: float, out=None):
    """FA3 over ``requests`` sequences of ``rows`` rows, each one varlen request from position 0 with
    ``FA3_INVARIANT_SPLITS`` splits."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: PLC0415 - vLLM is a GPU-only extra

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


@functools.lru_cache(maxsize=16)
def _window_lengths(count: int, window: int, device: torch.device) -> torch.Tensor:
    return torch.full((count,), window, dtype=torch.int32, device=device)


def _fa3_window_rows_forward(query, key, value, *, rows: int, requests: int, window: int, scale: float):
    """Sliding-window FA3 over ``requests`` sequences of ``rows`` rows, each row computed as a decode-invariant engine
    computes it: the rows before the window's end as one request, each later row as a request over its window."""
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


def topk_experts(biased_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """The ``[rows, top_k]`` experts vLLM's Grug router selects from the fp32 biased logits, in its slot order: the
    first ``top_k`` of ``torch.topk`` of ``top_k + 1``, which breaks exact ties as the engine does."""
    return torch.topk(biased_logits, k=top_k + 1, dim=-1).indices[:, :top_k]


def token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Model runner V2's log-probability (``compute_token_logprobs``) of ``token_ids[i]`` under row ``i`` of the
    ``[rows, vocab]`` ``logits``; each row's value depends on that row alone."""
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs  # noqa: PLC0415 - vLLM is a GPU-only extra

    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"vLLM's log-probability kernel reads rows of unit vocabulary stride, got {logits.stride()}")
    return compute_token_logprobs(logits, token_ids.reshape(-1, 1))[:, 0]


def expert_weight_offsets(weights: Sequence[torch.Tensor], base: torch.Tensor) -> list[int]:
    """Each weight's address minus that of ``base``, the lowest-addressed weight, in units of
    ``EXPERT_OFFSET_ELEMENTS`` elements."""
    if any(
        weight.shape != base.shape or weight.dtype != base.dtype or not weight.is_contiguous() for weight in weights
    ):
        raise ValueError("expert_outputs needs contiguous expert weights of one shape and dtype")
    unit = EXPERT_OFFSET_ELEMENTS * base.element_size()
    distances = [weight.data_ptr() - base.data_ptr() for weight in weights]
    if any(distance < 0 or distance % unit for distance in distances):
        raise ValueError(f"expert_outputs needs expert weights at whole {unit}-byte steps above the lowest one")
    return [distance // unit for distance in distances]


@functools.lru_cache(maxsize=256)
def _offset_table(offsets: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _expert_offset_table(weights: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest-addressed expert weight and the experts' ``expert_weight_offsets`` table on ``device``, copied to
    the device once per distinct table."""
    lowest = min(weights, key=lambda weight: weight.data_ptr())
    return lowest, _offset_table(tuple(expert_weight_offsets(weights, lowest)), device)


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
    """vLLM's ``invoke_fused_moe_triton_kernel`` on per-expert weight tensors (see ``expert_outputs``)."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: PLC0415 - vLLM is a GPU-only extra
        invoke_fused_moe_triton_kernel,
    )
    from vllm.triton_utils import tl  # noqa: PLC0415 - vLLM is a GPU-only extra

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


def expert_outputs(
    hidden: torch.Tensor,
    tokens_per_expert: Sequence[int],
    probs: torch.Tensor,
    fc1_weights: Sequence[torch.Tensor],
    fc2_weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Each dispatched row's routed-expert output, weighted by its route weight, as the decode-invariant engine's
    fused-MoE kernels compute one token-expert slot.

    ``hidden`` ``[rows, hidden]`` holds each local expert's rows in turn (``tokens_per_expert`` of them),
    ``probs`` the rows' fp32 route weights, and ``fc1_weights`` / ``fc2_weights`` each expert's
    ``[2 * ffn, hidden]`` gate-and-up and ``[hidden, ffn]`` down projection, laid out as
    ``expert_weight_offsets`` requires. A slot's value depends only on its row, its expert and its route weight.
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: PLC0415 - GPU-only extra
        moe_align_block_size,
    )

    from skyrl_train.inference_engines.vllm.decode_invariant.model import (  # noqa: PLC0415 - GPU-only extra
        INVARIANT_MOE_CONFIG,
    )

    rows, width = hidden.shape
    experts = len(fc1_weights)
    gate_up_width = fc1_weights[0].shape[0]
    if probs.dtype != torch.float32 or probs.shape != (rows,):
        raise ValueError(f"expert_outputs needs one fp32 route weight per row, got {probs.dtype}{tuple(probs.shape)}")
    if len(fc2_weights) != experts or len(tokens_per_expert) != experts:
        raise ValueError("expert_outputs needs one fc1 weight, one fc2 weight and one row count per expert")
    output = hidden.new_empty(rows, width)
    if rows == 0:
        return output
    config = INVARIANT_MOE_CONFIG
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
