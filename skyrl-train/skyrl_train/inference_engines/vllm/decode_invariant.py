"""A decode-invariant compiled vLLM engine for Grug: each token's bytes do not depend on the step that computes it.

``install`` patches vLLM in the calling process:

- ``_patch_fixed_splits``: every FA3 request runs ``FA3_INVARIANT_SPLITS`` key splits, and cascade attention is off.
- ``_patch_request_plan``: a step's FA3 call runs as the calls ``_invariant_calls`` plans, which compute each row as a
  decode step computes it.
- ``_patch_router``: a Triton GEMM with one tile shape computes the fp32 router logits.
- ``_patch_autotune``: Inductor's autotuner takes each kernel's launch configs in a fixed order instead of timing them.

Inductor pads the head-gate GEMM in every engine process (``force_shape_pad``, which
``create_ray_wrapped_inference_engines`` sets for a decode-invariant engine). The engine's other kernels give a row
the same bytes at every row count. ``decode_invariant_worker.DecodeInvariantWorkerWrap`` is the engine's worker
extension; importing it installs the patches in each worker process before the model loads.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import NamedTuple

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from vllm.model_executor.models import grugmoe
from vllm.v1.attention.backends import flash_attn

from skyrl_train.models.grug_invariant_kernels import check_bf16_values, invariant_router_logits, launcher_preference
from skyrl_train.models.grug_fa3_invariant import (
    FA3_BLOCK_M,
    FA3_DYNAMIC_SPLIT_MAX_BATCH,
    FA3_INVARIANT_SPLITS,
    fa3_fixed_split_metadata,
    fa3_fixed_split_metadata_size,
    fa3_invariant_requests,
    fa3_request_calls,
    fa3_window_start_rows,
)

_installed = False
# ``(window, window starts)`` while ``invariant_forward`` runs a call of one-row sliding-window requests, whose varlen
# call ``_causal_window_varlen`` runs.
_window_call: tuple[int, torch.Tensor] | None = None


@torch.library.custom_op("skyrl::grug_router_logits", mutates_args=())
def grug_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grug's router logits from the row-invariant GEMM, an opaque op inside vLLM's compiled graph."""
    return invariant_router_logits(x, weight)


@grug_router_logits.register_fake
def _grug_router_logits_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


def _patch_fixed_splits() -> None:
    """Every FA3 call of the engine runs ``FA3_INVARIANT_SPLITS`` splits for each request, on CUDA-graph and eager
    steps, with cascade attention off."""
    builder = flash_attn.FlashAttentionMetadataBuilder
    original_init, original_build = builder.__init__, builder.build

    def fixed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.max_num_splits = FA3_INVARIANT_SPLITS
        self.skyrl_scheduler_buffer = None
        self.skyrl_scheduler_metadata = None
        self.skyrl_query_lengths = None
        if self.use_full_cuda_graph:
            # Full CUDA graphs replay FA3 calls that read the scheduler metadata from this buffer.
            batch = max(self.vllm_config.scheduler_config.max_num_seqs, self.max_cudagraph_size or 0)
            size = fa3_fixed_split_metadata_size(batch)
            self.skyrl_scheduler_buffer = torch.zeros(size, dtype=torch.int32, device=self.device)

    def fixed_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        if metadata.use_cascade or metadata.dcp_context_kv_lens is not None:
            raise NotImplementedError("fixed FA3 splits run plain FA3 varlen steps only")
        metadata.max_num_splits = FA3_INVARIANT_SPLITS
        query_start = common_attn_metadata.query_start_loc_cpu
        if query_start.numel() != metadata.query_start_loc.numel():
            raise ValueError("the step's host and device query starts hold different request counts")
        query_lengths = (query_start[1:] - query_start[:-1]).numpy().tobytes()
        # The metadata depends on the query lengths alone, and FA3's combine kernel zeroes its tile semaphore.
        if query_lengths != self.skyrl_query_lengths:
            self.skyrl_scheduler_metadata = fa3_fixed_split_metadata(
                metadata.query_start_loc,
                self.num_heads_q // self.num_heads_kv,
                self.num_heads_kv,
                out=self.skyrl_scheduler_buffer,
            )
            self.skyrl_query_lengths = query_lengths
        metadata.scheduler_metadata = self.skyrl_scheduler_metadata
        return metadata

    builder.__init__, builder.build = fixed_init, fixed_build
    builder.use_cascade_attention = lambda self, *args, **kwargs: False


def _patch_request_plan() -> None:
    """Run a step's FA3 call as the calls ``_invariant_calls`` plans."""
    original_forward = flash_attn.FlashAttentionImpl.forward
    original_varlen = flash_attn.flash_attn_varlen_func

    def invariant_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        global _window_call
        calls = None if attn_metadata is None else _invariant_calls(self, attn_metadata)
        if calls is None:
            return original_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        for rows, metadata, window_call in calls:
            _window_call = window_call
            try:
                original_forward(
                    self, layer, query[rows], key, value, kv_cache, metadata, output[rows], *args, **kwargs
                )
            finally:
                _window_call = None
        return output

    def window_start_varlen(*args, **kwargs):
        if _window_call is None:
            return original_varlen(*args, **kwargs)
        return _causal_window_varlen(*_window_call, *args, **kwargs)

    flash_attn.FlashAttentionImpl.forward = invariant_forward
    flash_attn.flash_attn_varlen_func = window_start_varlen


def _causal_window_varlen(window: int, window_starts: torch.Tensor, *args, **kwargs):
    """vLLM's FA3 varlen call of one-row requests of a sliding-window layer, run by ``fa3_window_start_rows``."""
    window_size = kwargs.get("window_size")
    if (
        args
        or kwargs.get("fa_version") != 3
        or kwargs.get("max_seqlen_q") != 1
        or kwargs.get("causal") is not True
        or list(window_size or ()) != [window - 1, 0]
        or kwargs.get("cu_seqlens_k") is not None
        or any(kwargs.get(name) is not None for name in ("alibi_slopes", "dynamic_causal", "mask_mod", "aux_tensors"))
    ):
        raise NotImplementedError("window starts apply to FA3 varlen calls of one-row causal sliding-window requests")
    return fa3_window_start_rows(
        kwargs["q"],
        kwargs["k"],
        kwargs["v"],
        kwargs["out"],
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        seqused_k=kwargs["seqused_k"],
        leftpad_k=window_starts,
        max_seqlen_k=kwargs["max_seqlen_k"],
        block_table=kwargs["block_table"],
        softmax_scale=kwargs["softmax_scale"],
        scheduler_metadata=kwargs["scheduler_metadata"],
        num_splits=kwargs["num_splits"],
        softcap=kwargs["softcap"],
        q_descale=kwargs.get("q_descale"),
        k_descale=kwargs.get("k_descale"),
        v_descale=kwargs.get("v_descale"),
        s_aux=kwargs.get("s_aux"),
    )


class _StepCall(NamedTuple):
    """One FA3 call of a step: its query rows, its attention metadata, and for a call of one-row requests on FA3's
    causal kernel the ``(window, window starts)`` it runs with."""

    rows: slice
    metadata: object
    window_call: tuple[int, torch.Tensor] | None


def _invariant_calls(impl, metadata) -> list[_StepCall] | None:
    """The FA3 calls that compute each row of ``impl``'s layer as a decode step computes it, or ``None`` when the step's
    call runs as built."""
    left, right = impl.sliding_window
    window = left + 1 if left >= 0 and right == 0 else None
    cached = metadata.__dict__.setdefault("_skyrl_calls", {})
    if window in cached:
        return cached[window]
    requests = metadata.query_start_loc.numel() - 1
    single_rows = metadata.max_query_len <= 1
    if window is not None and single_rows and requests <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
        # Window starts are computed on the device, so captured CUDA graphs replay them, and cached on the step's
        # key-length tensor, which every KV-cache group of the step shares.
        by_window = metadata.seq_lens.__dict__.setdefault("_skyrl_window_starts", {})
        if window not in by_window:
            by_window[window] = torch.clamp(metadata.seq_lens[:requests] - window, min=0)
        cached[window] = [_StepCall(slice(None), metadata, (window, by_window[window]))]
        return cached[window]
    if single_rows and requests <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
        cached[window] = None
        return None
    if metadata.use_cascade or metadata.dcp_context_kv_lens is not None:
        raise NotImplementedError("decode-invariant FA3 requests rewrite plain FA3 varlen steps only")
    group = impl.num_heads // impl.num_kv_heads
    if FA3_BLOCK_M % group:
        raise NotImplementedError(f"decode-invariant FA3 requests need query heads per KV head dividing {FA3_BLOCK_M}")
    query_start = metadata.query_start_loc.tolist()
    planned = fa3_invariant_requests(query_start, metadata.seq_lens[:requests].tolist(), window, FA3_BLOCK_M // group)
    total = planned.key_lengths.numel()
    if total == requests and total <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
        cached[window] = None
        return None
    device = metadata.query_start_loc.device
    starts = planned.query_start.tolist()
    calls = []
    for begin, end, one_row in fa3_request_calls(
        starts, one_row_calls=window is not None, max_requests=FA3_DYNAMIC_SPLIT_MAX_BATCH
    ):
        key_lengths = planned.key_lengths[begin:end]
        call = copy.copy(metadata)
        call.num_actual_tokens = starts[end] - starts[begin]
        call.query_start_loc = (planned.query_start[begin : end + 1] - starts[begin]).to(device)
        call.seq_lens = key_lengths.to(device)
        call.block_table = metadata.block_table[planned.owner[begin:end].to(device)]
        call.max_query_len = max(b - a for a, b in zip(starts[begin:end], starts[begin + 1 : end + 1], strict=True))
        call.max_seq_len = int(key_lengths.max())
        call.scheduler_metadata = fa3_fixed_split_metadata(call.query_start_loc, group, impl.num_kv_heads)
        window_call = (window, torch.clamp(key_lengths - window, min=0).to(device)) if one_row else None
        calls.append(_StepCall(slice(starts[begin], starts[end]), call, window_call))
    cached[window] = calls
    return calls


def _patch_router() -> None:
    """Grug's router logits from ``skyrl::grug_router_logits``: the fp32 router is Grug's only fp32-weight linear."""
    original_apply = grugmoe._apply_grug_linear

    def apply_grug_linear(layer, x):
        if layer.weight.dtype == torch.float32:
            return torch.ops.skyrl.grug_router_logits(x, layer.weight)
        return original_apply(layer, x)

    grugmoe._apply_grug_linear = apply_grug_linear
    original_load_weights = grugmoe.GrugMoeForCausalLM.load_weights

    def checked_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = original_load_weights(self, weights)
        check_router_weights(self)
        return loaded

    grugmoe.GrugMoeForCausalLM.load_weights = checked_load_weights


def check_router_weights(model: torch.nn.Module) -> None:
    """Raise unless every materialized router weight holds bf16 values; a weight sync's meta parameters are
    skipped until it finishes."""
    for name, parameter in model.named_parameters():
        if name.endswith(".mlp.router.weight") and not parameter.is_meta:
            check_bf16_values(name, parameter.data)


def _patch_autotune() -> None:
    """Inductor's autotuner ranks a kernel's launch configs by ``launcher_preference`` instead of timing them."""

    def benchmark_all_configs(self, *args, **kwargs):
        ranked = sorted(self.launchers, key=launcher_preference)
        return {launcher: float(rank) for rank, launcher in enumerate(ranked)}

    CachingAutotuner.benchmark_all_configs = benchmark_all_configs


def install() -> None:
    """Patch vLLM in this process; a second call does nothing."""
    global _installed
    if _installed:
        return
    _patch_fixed_splits()
    _patch_request_plan()
    _patch_router()
    _patch_autotune()
    _installed = True
