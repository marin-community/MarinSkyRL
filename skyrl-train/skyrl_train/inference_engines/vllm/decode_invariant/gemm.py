"""A Triton GEMM with one tile shape at every row count, so a row's result does not depend on the other rows.

The decode-invariant vLLM engine computes Grug's router logits and head gates with it, through the
``skyrl::invariant_gemm`` torch custom op, which stays opaque inside vLLM's compiled graph.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Every call uses one tile shape and GEMM_SPLITS fixed K slices, whose sums are added left to right.
GEMM_BLOCK_M = 64
GEMM_BLOCK_N = 64
GEMM_BLOCK_K = 64
GEMM_SPLITS = 8
GEMM_SUM_BLOCK = 1024


@triton.jit
def _gemm_partials_kernel(
    x_ptr,
    w_ptr,
    partial_ptr,
    rows,
    columns,
    depth,
    split_depth,
    x_row_stride,
    w_row_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``partial[s, m, n]``: the sum over K slice ``s`` of ``x[m, k] * w[n, k]``, accumulated in fp32 one
    ``BLOCK_K`` block after another."""
    row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    split = tl.program_id(2)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    depth_offsets = split * split_depth + tl.arange(0, BLOCK_K)
    row_mask = row_offsets[:, None] < rows
    column_mask = column_offsets[None, :] < columns
    x_block = x_ptr + row_offsets[:, None].to(tl.int64) * x_row_stride + depth_offsets[None, :]
    w_block = w_ptr + column_offsets[None, :].to(tl.int64) * w_row_stride + depth_offsets[:, None]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, split_depth, BLOCK_K):
        depth_mask = (split * split_depth + start + tl.arange(0, BLOCK_K)) < depth
        x = tl.load(x_block, mask=row_mask & depth_mask[None, :], other=0.0).to(tl.bfloat16)
        w = tl.load(w_block, mask=column_mask & depth_mask[:, None], other=0.0).to(tl.bfloat16)
        accumulator += tl.dot(x, w)
        x_block += BLOCK_K
        w_block += BLOCK_K
    partial = partial_ptr + (split * rows + row_offsets[:, None]).to(tl.int64) * columns + column_offsets[None, :]
    tl.store(partial, accumulator, mask=row_mask & column_mask)


@triton.jit
def _gemm_sum_kernel(partial_ptr, out_ptr, elements, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    """``out = partial[0] + partial[1] + ... + partial[SPLITS - 1]``, left to right."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    total = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLITS):
        total += tl.load(partial_ptr + split * elements + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, total, mask=mask)


def invariant_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` in fp32 for ``x`` ``[rows, K]`` and ``weight`` ``[N, K]``, both bf16-valued.

    Each row's result depends on that row and the weight alone: the kernel multiplies on bf16 tensor cores with one
    tile shape at every row count, accumulates each of eight fixed K slices block by block, then adds the slice sums
    left to right. Inputs may be bf16 or fp32 holding bf16 values; callers check the weight once
    (``check_bf16_values``).
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"invariant_gemm needs [rows, K] and [N, K], got {tuple(x.shape)} and {tuple(weight.shape)}")
    depth = x.shape[1]
    if depth % (GEMM_SPLITS * GEMM_BLOCK_K):
        raise ValueError(f"invariant_gemm needs K divisible by {GEMM_SPLITS * GEMM_BLOCK_K}, got {depth}")
    x, weight = x.contiguous(), weight.contiguous()
    rows, columns = x.shape[0], weight.shape[0]
    out = torch.empty(rows, columns, dtype=torch.float32, device=x.device)
    if rows == 0:
        return out
    partials = torch.empty(GEMM_SPLITS, rows, columns, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(rows, GEMM_BLOCK_M), triton.cdiv(columns, GEMM_BLOCK_N), GEMM_SPLITS)
    _gemm_partials_kernel[grid](
        x,
        weight,
        partials,
        rows,
        columns,
        depth,
        depth // GEMM_SPLITS,
        x.stride(0),
        weight.stride(0),
        BLOCK_M=GEMM_BLOCK_M,
        BLOCK_N=GEMM_BLOCK_N,
        BLOCK_K=GEMM_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    elements = rows * columns
    _gemm_sum_kernel[(triton.cdiv(elements, GEMM_SUM_BLOCK),)](
        partials, out, elements, SPLITS=GEMM_SPLITS, BLOCK=GEMM_SUM_BLOCK, num_warps=4
    )
    return out


@torch.library.custom_op("skyrl::invariant_gemm", mutates_args=())
def invariant_gemm_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``invariant_gemm`` as a torch custom op, opaque to vLLM's compiled graph."""
    return invariant_gemm(x, weight)


@invariant_gemm_op.register_fake
def _invariant_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


def check_bf16_values(name: str, tensor: torch.Tensor) -> None:
    """Raise unless every value of ``tensor`` is a bf16 value (``invariant_gemm`` multiplies in bf16)."""
    if not torch.equal(tensor, tensor.to(torch.bfloat16).to(tensor.dtype)):
        raise ValueError(f"{name} holds values bf16 cannot represent; the invariant router GEMM multiplies in bf16")
