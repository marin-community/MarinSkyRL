"""Row-invariant kernels that the decode-invariant vLLM engine and the Grug trainer both run.

A GEMM library picks its kernel, and with it the order it adds each output's products, from the call's row count. The
Triton GEMM here uses one tile shape and no split-K and adds the K blocks of every output in order into one fp32
accumulator, so a row's bytes depend on that row and the weight alone, whatever else the call holds.

``invariant_router_logits`` is Grug's fp32 router GEMM. The router input is bf16 and the router weight holds bf16
values, so every product is exact in fp32; the kernel multiplies on bf16 tensor cores and adds each 64-wide K block's
sum to the fp32 accumulator.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One tile shape for every call: the bytes of a row do not depend on the row count.
ROUTER_BLOCK_M = 64
ROUTER_BLOCK_N = 64
ROUTER_BLOCK_K = 64


@triton.jit
def _router_logits_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    rows,
    columns,
    depth,
    x_row_stride,
    w_row_stride,
    out_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``out[m, n] = sum_k x[m, k] * w[n, k]``: bf16 products, each K block summed apart, then added in K order."""
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    depth_offsets = tl.arange(0, BLOCK_K)
    row_mask = row_offsets[:, None] < rows
    column_mask = column_offsets[None, :] < columns
    x_block = x_ptr + row_offsets[:, None].to(tl.int64) * x_row_stride + depth_offsets[None, :]
    w_block = w_ptr + column_offsets[None, :].to(tl.int64) * w_row_stride + depth_offsets[:, None]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, depth, BLOCK_K):
        depth_mask = (start + depth_offsets) < depth
        x = tl.load(x_block, mask=row_mask & depth_mask[None, :], other=0.0).to(tl.bfloat16)
        w = tl.load(w_block, mask=column_mask & depth_mask[:, None], other=0.0).to(tl.bfloat16)
        accumulator += tl.dot(x, w)
        x_block += BLOCK_K
        w_block += BLOCK_K
    out = out_ptr + row_offsets[:, None].to(tl.int64) * out_row_stride + column_offsets[None, :]
    tl.store(out, accumulator, mask=row_mask & column_mask)


def invariant_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grug's fp32 router logits ``x @ weight.T`` for ``x`` ``[rows, K]`` and ``weight`` ``[N, K]``, both bf16-valued.

    Each row's bytes depend on that row and the weight alone. Inputs may be bf16 or fp32 holding bf16 values; an fp32
    value that bf16 cannot hold exactly would be rounded, so callers check the weight once (``check_bf16_values``).
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"router logits need [rows, K] and [N, K], got {tuple(x.shape)} and {tuple(weight.shape)}")
    x, weight = x.contiguous(), weight.contiguous()
    rows, depth = x.shape
    columns = weight.shape[0]
    out = torch.empty(rows, columns, dtype=torch.float32, device=x.device)
    if rows == 0:
        return out
    grid = (triton.cdiv(rows, ROUTER_BLOCK_M), triton.cdiv(columns, ROUTER_BLOCK_N))
    _router_logits_kernel[grid](
        x,
        weight,
        out,
        rows,
        columns,
        depth,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        BLOCK_M=ROUTER_BLOCK_M,
        BLOCK_N=ROUTER_BLOCK_N,
        BLOCK_K=ROUTER_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out


def check_bf16_values(name: str, tensor: torch.Tensor) -> None:
    """Raise unless every value of ``tensor`` is a bf16 value (``invariant_router_logits`` multiplies in bf16)."""
    if not torch.equal(tensor, tensor.to(torch.bfloat16).to(tensor.dtype)):
        raise ValueError(f"{name} holds values bf16 cannot represent; the invariant router GEMM multiplies in bf16")
