"""Compiled vLLM's Inductor kernels for Grug's norms, q/k chain, XSA and shared-expert activation, run by the trainer.

The sources are copied verbatim from the Inductor output code of Snowball's compiled vLLM engines (vLLM
``70ea9ae8f260``, torch ``2.13.0+cu132``, Triton ``3.7.1``, H100), must be copied again whenever vLLM, torch or Triton
change, and hard-code Snowball's shapes (``config/grug_vllm_shapes.py``). Each kernel launches through Inductor's own
runtime with the config the decode-invariant engine's autotuner takes (``launcher_preference``).
``_RESIDUAL_NORM_SQUARE_SUM`` and ``_EMBEDDING_PRODUCT_NORM_SQUARE_SUM`` also store the per-row sum of squares, and
``_NORM_FROM_SQUARE_SUM`` normalizes a stored row from such a sum with those kernels' own second loop.
"""

from __future__ import annotations

import functools

import torch
from torch._inductor.async_compile import AsyncCompile

from skyrl_train.config.grug_vllm_shapes import HEAD_DIM, HEADS, HIDDEN, KV_HEADS, ROTARY_POSITIONS, SHARED_WIDTH
from skyrl_train.models.grug_invariant_kernels import launcher_preference

# XSA and the 2*sigmoid head gate over each head's 128 dims (every attention layer).
_XSA_GATE = (
    "triton_red_fused__unsafe_view_add_clone_div_expand_mm_mul_sigmoid_sub_sum_unsqueeze_view_1",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_div_expand_mm_mul_sigmoid_sub_sum_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 327680, 'r0_': 136314880}, 'kernel_num_gb': 0.094765056, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_div_expand_mm_mul_sigmoid_sub_sum_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 20)
    x1 = xindex // 20
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*(x0 // 4) + 640*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp6 = tmp1 * tmp1
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tl.load(in_ptr2 + (x0 + 24*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_2 + 128*(x0 // 4) + 640*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.sigmoid(tmp10)
        tmp12 = tl.full([1, 1], 2.0, tl.float32)
        tmp13 = tmp11 * tmp12
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp8 + tmp15
        tmp17 = (tmp4 / tmp16)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp14 - tmp19
        tmp21 = tmp13 * tmp20
        tl.store(out_ptr2 + (r0_2 + 128*x3), tmp21, r0_mask & xmask)
""",
)

# The post-attention RMSNorm of the stored residual.
_RMS_NORM = (
    "triton_red_fused_rms_norm_2",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_rms_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 125834240}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.0838912, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_rms_norm_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp6 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 2560.0, tl.float32)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tl.full([1, 1], 1e-05, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp16, r0_mask & xmask)
""",
)

# A gated norm's ``norm * sigmoid(gate)``, in place.
_GATED_PRODUCT = (
    "triton_poi_fused_mul_sigmoid_4",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
""",
)

# The shared expert's activation ``silu(gate) * up``, in place in the gate projection's buffer.
_SHARED_SWIGLU = (
    "triton_poi_fused_mul_silu_6",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_mul_silu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 167772160}, 'kernel_num_gb': 0.12582912, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
""",
)

# ``h + (routed + shared)`` stored in place, then the next layer's input norm (variance of the unrounded sum).
_RESIDUAL_NORM = (
    "triton_red_fused_add_rms_norm_7",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_rms_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 293606400}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.20972032, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_add_rms_norm_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp4, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 2560.0, tl.float32)
        tmp13 = (tmp8 / tmp12)
        tmp14 = tl.full([1, 1], 1e-05, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp20, r0_mask & xmask)
""",
)

# The last layer's ``h + (routed + shared)`` normalized unrounded by the final norm, in place.
_FINAL_NORM = (
    "triton_red_fused_add_rms_norm_7",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_add_rms_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 209720320}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.16777728, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_add_rms_norm_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 2560.0, tl.float32)
        tmp17 = (tmp8 / tmp16)
        tmp18 = tl.full([1, 1], 1e-05, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp24, r0_mask & xmask)
""",
)

# Per-head sums of squares of q and k (RoPE layers).
_QK_SQUARE_SUMS = (
    "triton_per_fused_8",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': None, 'xnumel_0': None, 'no_x_dim_1': None, 'xnumel_1': None}, 'kernel_name': 'triton_per_fused_8', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'max_persistent_rblock': 128}
)
@triton.jit
def triton_per_fused_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 128
        R0_BLOCK_0: tl.constexpr = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_0
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_index = tl.arange(0, R0_BLOCK_0)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK_0])
        tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
        tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
        tl.store(out_ptr0 + (x0), tmp6, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 128
        R0_BLOCK_1: tl.constexpr = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_index = tl.arange(0, R0_BLOCK_1)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        x2 = xindex
        tmp7 = tl.load(in_ptr1 + (r0_3 + 128*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK_1])
        tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
        tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
        tl.store(out_ptr1 + (x2), tmp13, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 20, 1), (20, 1, 163840), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192, 5, 1), (5, 1, 40960), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 163840, 40960,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_8.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.053248000000000004
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
""",
)

# k normalized and rotated (both halves), q's rotary half normalized and rotated.
_QK_ROPE = (
    "triton_poi_fused_9",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x3 = xindex // 64
        x2 = xindex // 320
        tmp65 = tl.load(in_ptr0 + (64 + x0 + 128*x3), xmask).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], 32, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + (128*x3 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full([1], 128.0, tl.float32)
        tmp11 = (tmp9 / tmp10)
        tmp12 = tl.full([1], 1e-06, tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp8 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = (tl.full([XBLOCK], 65536, tl.int32)).to(tl.int32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp17 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp17)
        tl.device_assert(((0 <= tl.broadcast_to(tmp21, [XBLOCK])) & (tl.broadcast_to(tmp21, [XBLOCK]) < 65536)) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp21, [XBLOCK]) < 65536")
        tmp23 = tl.load(in_ptr3 + (64*tmp21 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tmp16 * tmp23
        tmp25 = tl.load(in_ptr0 + (32 + 128*x3 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 * tmp14
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.load(in_ptr3 + (32 + 64*tmp21 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tmp28 * tmp29
        tmp31 = tmp24 - tmp30
        tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
        tmp33 = tl.where(tmp6, tmp31, tmp32)
        tmp34 = tmp0 >= tmp5
        tmp35 = tl.full([1], 64, tl.int64)
        tmp36 = tmp0 < tmp35
        tmp37 = tl.load(in_ptr0 + (32 + 128*x3 + ((-32) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tl.load(in_ptr1 + (x3), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.full([1], 128.0, tl.float32)
        tmp41 = (tmp39 / tmp40)
        tmp42 = tl.full([1], 1e-06, tl.float32)
        tmp43 = tmp41 + tmp42
        tmp44 = libdevice.rsqrt(tmp43)
        tmp45 = tmp38 * tmp44
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tl.load(in_ptr2 + (x2), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
        tmp48 = (tl.full([XBLOCK], 65536, tl.int32)).to(tl.int32)
        tmp49 = tmp47 + tmp48
        tmp50 = tmp47 < 0
        tmp51 = tl.where(tmp50, tmp49, tmp47)
        tl.device_assert(((0 <= tl.broadcast_to(tmp51, [XBLOCK])) & (tl.broadcast_to(tmp51, [XBLOCK]) < 65536)) | ~(tmp34 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp51, [XBLOCK]) < 65536")
        tmp53 = tl.load(in_ptr3 + (64*tmp51 + ((-32) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tmp46 * tmp53
        tmp55 = tl.load(in_ptr0 + (128*x3 + ((-32) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp55.to(tl.float32)
        tmp57 = tmp56 * tmp44
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tl.load(in_ptr3 + (32 + 64*tmp51 + ((-32) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp60 = tmp58 * tmp59
        tmp61 = tmp54 + tmp60
        tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
        tmp63 = tl.where(tmp34, tmp61, tmp62)
        tmp64 = tl.where(tmp6, tmp33, tmp63)
        tmp66 = tmp65.to(tl.float32)
        tmp68 = tl.full([1], 128.0, tl.float32)
        tmp69 = (tmp67 / tmp68)
        tmp70 = tl.full([1], 1e-06, tl.float32)
        tmp71 = tmp69 + tmp70
        tmp72 = libdevice.rsqrt(tmp71)
        tmp73 = tmp66 * tmp72
        tmp74 = tmp73.to(tl.float32)
        tl.store(out_ptr0 + (x0 + 128*x3), tmp64, xmask)
        tl.store(out_ptr1 + (x0 + 128*x3), tmp74, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x4 = (xindex % 64)
        x7 = xindex // 64
        x6 = xindex // 1280
        x8 = xindex
        tmp75 = (x4).to(tl.int32)
        tmp76 = tl.full([1], 0, tl.int64)
        tmp77 = tmp75 >= tmp76
        tmp78 = (x4).to(tl.int64)
        tmp79 = (tmp78).to(tl.int64)
        tmp80 = tl.full([1], 32, tl.int64)
        tmp81 = tmp79 < tmp80
        tmp82 = tl.load(in_ptr4 + (128*x7 + (x4)), tmp81 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp83 = tmp82.to(tl.float32)
        tmp84 = tl.load(in_ptr5 + (x7), tmp81 & xmask, eviction_policy='evict_last', other=0.0)
        tmp85 = tl.full([1], 128.0, tl.float32)
        tmp86 = (tmp84 / tmp85)
        tmp87 = tl.full([1], 1e-06, tl.float32)
        tmp88 = tmp86 + tmp87
        tmp89 = libdevice.rsqrt(tmp88)
        tmp90 = tmp83 * tmp89
        tmp91 = tmp90.to(tl.float32)
        tmp92 = tl.load(in_ptr2 + (x6), tmp81 & xmask, eviction_policy='evict_last', other=0.0)
        tmp93 = (tl.full([XBLOCK], 65536, tl.int32)).to(tl.int32)
        tmp94 = tmp92 + tmp93
        tmp95 = tmp92 < 0
        tmp96 = tl.where(tmp95, tmp94, tmp92)
        tl.device_assert(((0 <= tl.broadcast_to(tmp96, [XBLOCK])) & (tl.broadcast_to(tmp96, [XBLOCK]) < 65536)) | ~(tmp81 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp96, [XBLOCK]) < 65536")
        tmp98 = tl.load(in_ptr3 + (64*tmp96 + (x4)), tmp81 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tmp91 * tmp98
        tmp100 = tl.load(in_ptr4 + (32 + 128*x7 + (x4)), tmp81 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp101 = tmp100.to(tl.float32)
        tmp102 = tmp101 * tmp89
        tmp103 = tmp102.to(tl.float32)
        tmp104 = tl.load(in_ptr3 + (32 + 64*tmp96 + (x4)), tmp81 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp105 = tmp103 * tmp104
        tmp106 = tmp99 - tmp105
        tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
        tmp108 = tl.where(tmp81, tmp106, tmp107)
        tmp109 = tmp75 >= tmp80
        tmp110 = tl.full([1], 64, tl.int64)
        tmp111 = tmp75 < tmp110
        tmp112 = tl.load(in_ptr4 + (32 + 128*x7 + ((-32) + x4)), tmp109 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp113 = tmp112.to(tl.float32)
        tmp114 = tl.load(in_ptr5 + (x7), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
        tmp115 = tl.full([1], 128.0, tl.float32)
        tmp116 = (tmp114 / tmp115)
        tmp117 = tl.full([1], 1e-06, tl.float32)
        tmp118 = tmp116 + tmp117
        tmp119 = libdevice.rsqrt(tmp118)
        tmp120 = tmp113 * tmp119
        tmp121 = tmp120.to(tl.float32)
        tmp122 = tl.load(in_ptr2 + (x6), tmp109 & xmask, eviction_policy='evict_last', other=0.0)
        tmp123 = (tl.full([XBLOCK], 65536, tl.int32)).to(tl.int32)
        tmp124 = tmp122 + tmp123
        tmp125 = tmp122 < 0
        tmp126 = tl.where(tmp125, tmp124, tmp122)
        tl.device_assert(((0 <= tl.broadcast_to(tmp126, [XBLOCK])) & (tl.broadcast_to(tmp126, [XBLOCK]) < 65536)) | ~(tmp109 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp126, [XBLOCK]) < 65536")
        tmp128 = tl.load(in_ptr3 + (64*tmp126 + ((-32) + x4)), tmp109 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp129 = tmp121 * tmp128
        tmp130 = tl.load(in_ptr4 + (128*x7 + ((-32) + x4)), tmp109 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp131 = tmp130.to(tl.float32)
        tmp132 = tmp131 * tmp119
        tmp133 = tmp132.to(tl.float32)
        tmp134 = tl.load(in_ptr3 + (32 + 64*tmp126 + ((-32) + x4)), tmp109 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp135 = tmp133 * tmp134
        tmp136 = tmp129 + tmp135
        tmp137 = tl.full(tmp136.shape, 0.0, tmp136.dtype)
        tmp138 = tl.where(tmp109, tmp136, tmp137)
        tmp139 = tl.where(tmp81, tmp108, tmp138)
        tl.store(out_ptr2 + (x8), tmp139, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 640), (640, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 5, 1), (5, 1, 40960), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((65536, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_5 = rand_strided((8192, 20, 1), (20, 1, 163840), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8192, 5, 64), (640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_7 = rand_strided((8192, 5, 64), (640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_8 = rand_strided((8192, 20, 64), (1280, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 2621440, 10485760,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_9.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.17186816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
""",
)

# The final query: the stored rotary half and the normalized pass-through half, times the two query factors.
_Q_SCALE = (
    "triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_view_10",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 147456000}, 'kernel_num_gb': 0.10551296, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_mean_mul_pow_rsqrt_slice_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    tmp0 = ((x2 % 128)).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ((x2 % 128)).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (64*(x0 // 128) + 1280*x1 + ((x0 % 128))), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 128, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (64 + 128*(x0 // 128) + 2560*x1 + ((-64) + ((x0 % 128)))), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x2 // 128), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full([1], 128.0, tl.float32)
    tmp15 = (tmp13 / tmp14)
    tmp16 = tl.full([1], 1e-06, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp12 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp8, tmp20, tmp21)
    tmp23 = tl.where(tmp6, tmp7, tmp22)
    tmp24 = tl.full([1], 1.5703274004183787, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tl.full([1], 1.0, tl.float32)
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr0 + (x2), tmp27, xmask)
""",
)

# Full-attention layers: k normalized in place, q normalized and scaled.
_QK_FULL = (
    "triton_per_fused_8",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i64', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'autotune_grouping': True, 'default_config': None, 'no_x_dim_0': None, 'xnumel_0': None, 'no_x_dim_1': None, 'xnumel_1': None}, 'kernel_name': 'triton_per_fused_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'max_persistent_rblock': 128}
)
@triton.jit
def triton_per_fused_8(in_out_ptr0, in_ptr0, out_ptr2, ks0, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 128
        R0_BLOCK_0: tl.constexpr = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_0
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_index = tl.arange(0, R0_BLOCK_0)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK_0])
        tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
        tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
        tmp7 = tl.full([1, 1], 128.0, tl.float32)
        tmp8 = (tmp6 / tmp7)
        tmp9 = tl.full([1, 1], 1e-06, tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp1 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp13, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 128
        R0_BLOCK_1: tl.constexpr = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK_1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_index = tl.arange(0, R0_BLOCK_1)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        x2 = xindex
        tmp14 = tl.load(in_ptr0 + (r0_3 + 128*x2), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr0 + (((r0_3 + 128*x2) % (2560*ks0))), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK_1])
        tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
        tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 128.0, tl.float32)
        tmp24 = (tmp20 / tmp23)
        tmp25 = tl.full([1, 1], 1e-06, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.full([1, 1], 1.5703274004183787, tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tl.full([1, 1], 1.0, tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (((r0_3 + 128*x2) % (2560*ks0))), tmp33, r0_mask & xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((8192, 5, 128), (640, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_2 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = 8192
    return arg_0, arg_1, arg_2, arg_3, 40960, 163840,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_8.run(*args, stream=raw_stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_per_fused_8.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device='cuda',rep=40)
    num_gb = 0.1048576
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
""",
)

# Embedding lookup and the embedding RMSNorm.
_EMBEDDING_NORM = (
    "triton_red_fused__to_copy_embedding_rms_norm_0",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_embedding_rms_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 32768, 'r0_': 83891200}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.083923968, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_embedding_rms_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp1) & (tmp1 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp1 < 128256")
        tmp3 = tl.load(in_ptr1 + (r0_1 + 2560*tmp1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp9) & (tmp9 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp9 < 128256")
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2560*tmp9), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 2560.0, tl.float32)
        tmp14 = (tmp7 / tmp13)
        tmp15 = tl.full([1, 1], 1e-05, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp21, r0_mask & xmask)
""",
)

# The embedding gated norm's product stored in place, then layer 0's input norm (variance of the unrounded product).
_EMBEDDING_PRODUCT_NORM = (
    "triton_red_fused_mul_rms_norm_sigmoid_2",
    r"""
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mul_rms_norm_sigmoid_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 251663360}, 'add_persistent_rblock': True, 'kernel_num_gb': 0.16777728, 'kernel_flop': 0, 'backend_hash': 'B1F9651A75F5D2DD6203FECC047C63B4DA82AA1EC10FE90B895553194611C6F8', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused_mul_rms_norm_sigmoid_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp3, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.full([1, 1], 2560.0, tl.float32)
        tmp12 = (tmp7 / tmp11)
        tmp13 = tl.full([1, 1], 1e-05, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 * tmp18
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp19, r0_mask & xmask)
""",
)


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old!r} in a vendored kernel source")
    return text.replace(old, new)


def _with_square_sum_output(source: tuple[str, str], square_sum: str, argument_index: int) -> tuple[str, str]:
    """The kernel with one more output: its per-row fp32 sum of squares ``square_sum``, stored after the reduction."""
    name, text = source
    derived = f"{name}_square_sum"
    text = _replace_once(text, f"def {name}(", f"def {derived}(")
    text = _replace_once(text, f"'kernel_name': '{name}'", f"'kernel_name': '{derived}'")
    text = _replace_once(
        text, "'r0_numel': 'i32', 'XBLOCK'", "'r0_numel': 'i32', 'out_ptr_square_sum': '*fp32', 'XBLOCK'"
    )
    text = _replace_once(
        text, "xnumel, r0_numel, XBLOCK : tl.constexpr", "xnumel, r0_numel, out_ptr_square_sum, XBLOCK : tl.constexpr"
    )
    text = _replace_once(
        text,
        "[['tt.divisibility', 16]]}]}",
        f"[['tt.divisibility', 16]], ({argument_index},): [['tt.divisibility', 16]]}}]}}",
    )
    reduction = f"    {square_sum} = tl.sum(_{square_sum}, 1)[:, None]\n"
    text = _replace_once(text, reduction, f"{reduction}    tl.store(out_ptr_square_sum + (x0), {square_sum}, xmask)\n")
    return derived, text


# ``_RESIDUAL_NORM`` and ``_EMBEDDING_PRODUCT_NORM`` with their sum of squares stored (argument 7 and 6).
_RESIDUAL_NORM_SQUARE_SUM = _with_square_sum_output(_RESIDUAL_NORM, "tmp8", 7)
_EMBEDDING_PRODUCT_NORM_SQUARE_SUM = _with_square_sum_output(_EMBEDDING_PRODUCT_NORM, "tmp7", 6)


# The second loop of ``_RESIDUAL_NORM`` (and of ``_EMBEDDING_PRODUCT_NORM``, whose second loop is the same arithmetic),
# normalizing the stored bf16 row with a sum of squares read from memory instead of the one the first loop reduced.
def _norm_from_square_sum() -> tuple[str, str]:
    name = "triton_red_rms_norm_from_square_sum"
    header, decorated = _RESIDUAL_NORM[1].split("@triton_heuristics.reduction(")
    decorator = decorated.split("@triton.jit")[0]
    for old, new in (
        (
            "{'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16',",
            "{'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16',",
        ),
        ("(4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]", "(5,): [['tt.divisibility', 16]]"),
        ("'kernel_name': 'triton_red_fused_add_rms_norm_7'", f"'kernel_name': '{name}'"),
        ("'mutated_arg_names': ['in_out_ptr0']", "'mutated_arg_names': []"),
    ):
        decorator = _replace_once(decorator, old, new)
    body = f"""@triton.jit
def {name}(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 2560.0, tl.float32)
        tmp13 = (tmp8 / tmp12)
        tmp14 = tl.full([1, 1], 1e-05, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp20, r0_mask & xmask)
"""
    return name, f"{header}@triton_heuristics.reduction({decorator}{body}"


_NORM_FROM_SQUARE_SUM = _norm_from_square_sum()

# Inductor's padded head-gate GEMM writes 24 columns, and the vendored XSA kernel reads the gate at that row stride.
GATE_COLUMNS = 24
ROTARY_DIM = 64


@functools.cache
def _kernel(name: str, text: str, device: int):
    """``name`` compiled from ``text`` by Inductor's runtime for ``device``, launching the config the decode-invariant
    engine's autotuner takes for it: the first of its candidates in ``launcher_preference`` order.

    The autotune cache stays off so that no ``.best_config`` left by another process narrows the candidates, and every
    other candidate is dropped before the first launch, so no launch benchmarks configs.
    """
    text = _replace_once(
        text, "DeviceProperties(type='cuda', index=0,", f"DeviceProperties(type='cuda', index={device},"
    )
    text = _replace_once(text, "'autotune_local_cache': True", "'autotune_local_cache': False")
    compiler = AsyncCompile()
    scope = {"kernel": compiler.triton(name, text, device_str="cuda")}
    compiler.wait(scope)
    kernel = scope["kernel"]
    kernel.launchers = [min(kernel.launchers, key=launcher_preference)]
    return kernel


def _run(source: tuple[str, str], *arguments) -> None:
    device = torch.cuda.current_device()
    _kernel(*source, device).run(*arguments, stream=torch.cuda.current_stream(device).cuda_stream)


@functools.cache
def _unit_weight(device: torch.device) -> torch.Tensor:
    return torch.ones(HIDDEN, dtype=torch.bfloat16, device=device)


def _rows(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """``tensor``'s values as contiguous bf16 ``[rows, width]``, outside autograd (the kernels compute values only)."""
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"vLLM kernels take bf16 rows, got {tensor.dtype}{tuple(tensor.shape)}")
    return tensor.detach().reshape(-1, width).contiguous()


def xsa_head_gate(attention: torch.Tensor, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's XSA and ``2 * sigmoid`` head gate: ``[rows, 2560]`` from attention, ``[rows, 640]`` value and
    ``[rows, 20]`` gate."""
    attention, value = _rows(attention, HIDDEN), _rows(value, KV_HEADS * HEAD_DIM)
    rows = attention.shape[0]
    padded_gate = gate.new_zeros(rows, GATE_COLUMNS)
    padded_gate[:, :HEADS] = gate.detach().reshape(rows, HEADS)
    output = torch.empty_like(attention)
    _run(
        _XSA_GATE,
        attention,
        value,
        padded_gate,
        output,
        HEADS * rows,
        HEAD_DIM,
    )
    return output


@functools.cache
def rotary_table(device: int, base: float) -> torch.Tensor:
    """vLLM's bf16 rotary cos/sin table, built on the GPU as its ``RotaryEmbedding`` builds it at model load."""
    with torch.device(f"cuda:{device}"):
        inv_freq = 1.0 / (base ** (torch.arange(0, ROTARY_DIM, 2, dtype=torch.float) / ROTARY_DIM))
        positions = torch.arange(ROTARY_POSITIONS, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def query_key_rope(
    query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor, rotary_base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled vLLM's q/k RMS norm, half RoPE and query scale on a sliding-window layer's raw q and k projections.

    ``query`` is ``[rows, 2560]``, ``key`` ``[rows, 640]`` and ``positions`` each row's int64 position. Returns the
    query ``[rows, 2560]`` and key ``[rows, 640]`` that vLLM hands to attention.
    """
    query, key = _rows(query, HIDDEN), _rows(key, KV_HEADS * HEAD_DIM)
    rows = query.shape[0]
    device = query.device.index
    query_sums = torch.empty(rows * HEADS, dtype=torch.float32, device=query.device)
    key_sums = torch.empty(rows * KV_HEADS, dtype=torch.float32, device=query.device)
    _run(
        _QK_SQUARE_SUMS,
        query,
        key,
        query_sums,
        key_sums,
        HEADS * rows,
        KV_HEADS * rows,
    )
    key_out = torch.empty(rows, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=query.device)
    query_rotary = torch.empty(rows, HEADS, ROTARY_DIM, dtype=torch.bfloat16, device=query.device)
    _run(
        _QK_ROPE,
        key,
        key_sums,
        positions.to(torch.int64).contiguous(),
        rotary_table(device, rotary_base),
        query,
        query_sums,
        key_out[:, :, :ROTARY_DIM],
        key_out[:, :, ROTARY_DIM:],
        query_rotary,
        KV_HEADS * ROTARY_DIM * rows,
        HEADS * ROTARY_DIM * rows,
    )
    query_out = torch.empty_like(query)
    _run(_Q_SCALE, query_rotary, query, query_sums, query_out, HIDDEN * rows)
    return query_out, key_out.view(rows, KV_HEADS * HEAD_DIM)


def query_key_full(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled vLLM's q/k RMS norm and query scale on a full-attention layer's raw q ``[rows, 2560]`` and k
    ``[rows, 640]``."""
    query = _rows(query, HIDDEN)
    key_out = _rows(key, KV_HEADS * HEAD_DIM).clone()
    rows = query.shape[0]
    query_out = torch.empty_like(query)
    _run(
        _QK_FULL,
        key_out,
        query,
        query_out,
        rows,
        KV_HEADS * rows,
        HEADS * rows,
    )
    return query_out, key_out


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's post-attention RMSNorm of the stored bf16 residual ``[rows, 2560]``."""
    hidden = _rows(hidden, HIDDEN)
    output = torch.empty_like(hidden)
    _run(_RMS_NORM, hidden, weight, output, hidden.shape[0], HIDDEN)
    return output


def gated_product(normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's gated-norm output ``normalized * sigmoid(gate)``, one rounding."""
    output = _rows(normalized, HIDDEN).clone()
    gate = _rows(gate, HIDDEN)
    _run(_GATED_PRODUCT, output, gate, output.numel())
    return output


def shared_activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's shared-expert activation ``silu(gate) * up`` of the two ``[rows, 2560]`` projections, one
    rounding."""
    if gate.shape[-1] != SHARED_WIDTH or up.shape != gate.shape:
        raise ValueError(f"gate and up of width {SHARED_WIDTH} expected, got {tuple(gate.shape)} and {tuple(up.shape)}")
    output = _rows(gate, SHARED_WIDTH).clone()
    _run(_SHARED_SWIGLU, output, _rows(up, SHARED_WIDTH), output.numel())
    return output


def residual_square_sum(
    residual: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The bf16 layer output ``residual + (routed + shared)`` and the fp32 per-row sum of squares of the unrounded sum.

    Both come from compiled vLLM's fused residual-add and input-norm kernel, run with a unit norm weight since its
    normalized output is discarded.
    """
    output = _rows(residual, HIDDEN).clone()
    rows = output.shape[0]
    square_sum = torch.empty(rows, dtype=torch.float32, device=output.device)
    scratch = torch.empty_like(output)
    _run(
        _RESIDUAL_NORM_SQUARE_SUM,
        output,
        _rows(routed, HIDDEN),
        _rows(shared, HIDDEN),
        _unit_weight(output.device),
        scratch,
        rows,
        HIDDEN,
        square_sum,
    )
    return output, square_sum


def gated_product_square_sum(normalized: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The embedding gated norm's bf16 product and the fp32 per-row sum of squares of the unrounded product.

    Both come from compiled vLLM's fused product and layer-0 input-norm kernel, normalizing with a unit weight as
    ``residual_square_sum`` does.
    """
    output = _rows(normalized, HIDDEN).clone()
    rows = output.shape[0]
    square_sum = torch.empty(rows, dtype=torch.float32, device=output.device)
    scratch = torch.empty_like(output)
    _run(
        _EMBEDDING_PRODUCT_NORM_SQUARE_SUM,
        output,
        _rows(gate, HIDDEN),
        _unit_weight(output.device),
        scratch,
        rows,
        HIDDEN,
        square_sum,
    )
    return output, square_sum


def rms_norm_from_square_sum(hidden: torch.Tensor, square_sum: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's input norm of the stored bf16 residual ``[rows, 2560]`` given the kernel's sum of squares."""
    hidden = _rows(hidden, HIDDEN)
    if square_sum.dtype != torch.float32 or square_sum.shape != (hidden.shape[0],):
        raise ValueError(f"one fp32 sum of squares per row expected, got {square_sum.dtype}{tuple(square_sum.shape)}")
    output = torch.empty_like(hidden)
    _run(
        _NORM_FROM_SQUARE_SUM,
        hidden,
        square_sum,
        weight,
        output,
        hidden.shape[0],
        HIDDEN,
    )
    return output


def final_norm(
    residual: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Compiled vLLM's final RMSNorm of the unrounded last-layer sum ``residual + (routed + shared)``."""
    output = _rows(residual, HIDDEN).clone()
    _run(
        _FINAL_NORM,
        output,
        _rows(routed, HIDDEN),
        _rows(shared, HIDDEN),
        weight,
        output.shape[0],
        HIDDEN,
    )
    return output


def embedding_norm(embeddings: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compiled vLLM's embedding RMSNorm of the looked-up bf16 embedding rows ``[rows, 2560]``.

    The kernel gathers rows of a table by token id; given the looked-up rows as the table and each row's own index
    as its id, it reads the same values.
    """
    table = _rows(embeddings, HIDDEN)
    rows = table.shape[0]
    output = torch.empty_like(table)
    identity = torch.arange(rows, dtype=torch.int32, device=table.device)
    _run(_EMBEDDING_NORM, identity, table, weight, output, rows, HIDDEN)
    return output
