"""Row-invariant kernels that the decode-invariant vLLM engine and the Grug trainer both run.

A GEMM library picks its kernel, and with it the order it adds each output's products, from the call's row count. The
Triton GEMM here uses one tile shape and one partition of K for every call, so a row's bytes depend on that row and the
weight alone, whatever else the call holds.

``invariant_router_logits`` is Grug's fp32 router GEMM. The router input is bf16 and the router weight holds bf16
values, so every product is exact in fp32; the kernel multiplies on bf16 tensor cores, sums each 64-wide K block apart,
adds the blocks of each of eight fixed K slices in order, then adds the eight slice sums left to right.

``launcher_preference`` orders an Inductor kernel's launch configs, which sum in different orders: the engine's
autotuner and the trainer's copies of compiled vLLM's kernels both launch the first.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One tile shape and one K partition for every call: the bytes of a row do not depend on the row count. K is cut into
# ROUTER_SPLITS fixed slices so that a call of a few rows (a decode step) runs on enough programs; each slice's sum
# is kept apart and the slices are added in order.
ROUTER_BLOCK_M = 64
ROUTER_BLOCK_N = 64
ROUTER_BLOCK_K = 64
ROUTER_SPLITS = 8
ROUTER_SUM_BLOCK = 1024


@triton.jit
def _router_partials_kernel(
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
    """``partial[s, m, n]``: the sum over K slice ``s`` of ``x[m, k] * w[n, k]``, bf16 products, each K block summed
    apart and added in order into one fp32 accumulator."""
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
def _router_sum_kernel(partial_ptr, out_ptr, elements, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    """``out = partial[0] + partial[1] + ... + partial[SPLITS - 1]``, left to right."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    total = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLITS):
        total += tl.load(partial_ptr + split * elements + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, total, mask=mask)


def invariant_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grug's fp32 router logits ``x @ weight.T`` for ``x`` ``[rows, K]`` and ``weight`` ``[N, K]``, both bf16-valued.

    Each row's bytes depend on that row and the weight alone. Inputs may be bf16 or fp32 holding bf16 values; an fp32
    value that bf16 cannot hold exactly would be rounded, so callers check the weight once (``check_bf16_values``).
    """
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError(f"router logits need [rows, K] and [N, K], got {tuple(x.shape)} and {tuple(weight.shape)}")
    depth = x.shape[1]
    if depth % (ROUTER_SPLITS * ROUTER_BLOCK_K):
        raise ValueError(f"router logits need K divisible by {ROUTER_SPLITS * ROUTER_BLOCK_K}, got {depth}")
    x, weight = x.contiguous(), weight.contiguous()
    rows, columns = x.shape[0], weight.shape[0]
    out = torch.empty(rows, columns, dtype=torch.float32, device=x.device)
    if rows == 0:
        return out
    partials = torch.empty(ROUTER_SPLITS, rows, columns, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(rows, ROUTER_BLOCK_M), triton.cdiv(columns, ROUTER_BLOCK_N), ROUTER_SPLITS)
    _router_partials_kernel[grid](
        x,
        weight,
        partials,
        rows,
        columns,
        depth,
        depth // ROUTER_SPLITS,
        x.stride(0),
        weight.stride(0),
        BLOCK_M=ROUTER_BLOCK_M,
        BLOCK_N=ROUTER_BLOCK_N,
        BLOCK_K=ROUTER_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    elements = rows * columns
    _router_sum_kernel[(triton.cdiv(elements, ROUTER_SUM_BLOCK),)](
        partials, out, elements, SPLITS=ROUTER_SPLITS, BLOCK=ROUTER_SUM_BLOCK, num_warps=4
    )
    return out


def launcher_preference(launcher) -> tuple[int, int, int, int]:
    """An Inductor launcher's rank in the order the decode-invariant engine's autotuner takes in place of timing:
    largest reduction block first, then most warps, then most rows per program, then most stages."""
    config = launcher.config
    return (
        -config.kwargs.get("R0_BLOCK", 0),
        -config.num_warps,
        -config.kwargs.get("XBLOCK", 0),
        -config.num_stages,
    )


def check_bf16_values(name: str, tensor: torch.Tensor) -> None:
    """Raise unless every value of ``tensor`` is a bf16 value (``invariant_router_logits`` multiplies in bf16)."""
    if not torch.equal(tensor, tensor.to(torch.bfloat16).to(tensor.dtype)):
        raise ValueError(f"{name} holds values bf16 cannot represent; the invariant router GEMM multiplies in bf16")
