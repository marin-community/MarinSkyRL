"""A decode-invariant compiled vLLM engine for Grug: each token's bytes do not depend on the step that computes it.

Compiled vLLM gives a token other bytes in a decode step than in a prefill step, or in steps of another composition,
in these places (a one-H100 sweep of each kernel at Snowball's shapes, one row to 8,192 rows per call):

- FA3 splits a request's keys by the step: 32 splits at most on CUDA-graph steps (a per-request count on the device,
  from the step's total key blocks), FA3's heuristic above. Rows computed with another split count add their key
  blocks in another order. Every FA3 request runs ``FA3_INVARIANT_SPLITS`` splits, through scheduler metadata the
  metadata builder writes for every step (``FIXED_SPLITS``). FA3 cuts each query tile's key blocks into the splits, so
  a prefill request starts at a multiple of 32 positions (one tile of packed query heads), its rows before the first
  such position running as a request of their own, and a call holds at most 992 requests, the most FA3 splits.
- On a sliding-window layer FA3 aligns its key blocks to the window start of each query tile's first row. A decode
  row is a tile of its own; inside a prefill tile the same row past the window adds other block groups. Prefill rows
  past the window run as one-row requests, as decode steps run them (``WINDOW_ROWS``). With fixed splits, every
  one-row request of a sliding-window layer runs on FA3's causal kernel from its window start
  (``fa3_window_start_rows``), which reads the same key blocks in the same splits as the local kernel, in less time.
- cuBLAS picks the fp32 router GEMM's kernel from the step's row count. A Triton GEMM with one tile shape computes the
  router logits instead (``ROUTER``).
- The 20-output head-gate GEMM: Inductor's ``pad_mm`` pass times it against a zero-padded 24-output copy when each
  engine process compiles, and the unpadded GEMM's cuBLAS kernel follows the row count. Every process compiles the
  padded GEMM: the engine's compilation setting ``force_shape_pad``, which ``create_ray_wrapped_inference_engines``
  sets for a decode-invariant engine.
- Inductor's autotuner picks some reductions' launch configs by timing them in each process. Every process takes the
  first config in a fixed order (``AUTOTUNE``), so every engine launches the same configs.

Every other kernel of a token's forward (the bf16 cuBLAS GEMMs, Inductor's fused kernels, the Triton experts, the LM
head and the log-probability kernel) gives the same bytes at every row count measured, from one row to 8,192.

``install`` applies the patches in the calling process. ``decode_invariant_worker.DecodeInvariantWorkerWrap`` is the
engine's worker extension: vLLM imports it in every worker process before the model loads, compiles and captures CUDA
graphs, and importing it installs every part. vLLM stays compiled.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

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
    fa3_invariant_requests,
    fa3_request_calls,
    fa3_window_start_rows,
)

FIXED_SPLITS = "fixed_splits"
WINDOW_ROWS = "window_rows"
ROUTER = "router"
AUTOTUNE = "autotune"
PARTS = (FIXED_SPLITS, WINDOW_ROWS, ROUTER, AUTOTUNE)

_installed: set[str] = set()
# ``(window, window starts)`` while ``invariant_forward`` runs a call of one-row sliding-window requests: vLLM's FA3
# forward makes one varlen call, which ``_causal_window_varlen`` runs on FA3's causal kernel.
_window_call: tuple[int, torch.Tensor] | None = None
# Marks the forward patch that both ``FIXED_SPLITS`` and ``WINDOW_ROWS`` install.
_REQUEST_PLAN = "request_plan"


@torch.library.custom_op("skyrl::grug_router_logits", mutates_args=())
def grug_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grug's router logits from the row-invariant GEMM, an opaque op inside vLLM's compiled graph."""
    return invariant_router_logits(x, weight)


@grug_router_logits.register_fake
def _grug_router_logits_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


def _fixed_splits() -> None:
    """Every FA3 call of the engine runs ``FA3_INVARIANT_SPLITS`` splits for each request, on CUDA-graph steps
    (captured with them) and on eager steps; vLLM's cascade attention, another computation for requests that share a
    prefix, stays off.

    The metadata builder writes FA3's scheduler metadata (``fa3_fixed_split_metadata``), into a buffer that lives as
    long as the builder when full CUDA graphs replay the step's FA3 calls. The metadata follows the step's query lengths
    alone, so the builder writes it only when they change (a run of decode steps of the same requests reuses it; FA3's
    combine kernel zeroes the tile semaphore after each call).
    """
    builder = flash_attn.FlashAttentionMetadataBuilder
    original_init, original_build = builder.__init__, builder.build

    def fixed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.max_num_splits = FA3_INVARIANT_SPLITS
        self.skyrl_scheduler_buffer = None
        self.skyrl_scheduler_metadata = None
        self.skyrl_query_lengths = None
        if self.use_full_cuda_graph:
            batch = max(self.vllm_config.scheduler_config.max_num_seqs, self.max_cudagraph_size or 0)
            size = 3 * -(-batch // 4) * 4 + 1
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
    _install_request_plan()


def _window_rows() -> None:
    """On sliding-window layers, prefill rows past the window run as one-row requests (``fa3_invariant_requests``)."""
    _install_request_plan()


def _install_request_plan() -> None:
    """Run a step's FA3 call as the calls that compute each row as a decode step computes it.

    Prefill and mixed steps, which vLLM runs outside CUDA graphs, and decode steps of more than
    ``FA3_DYNAMIC_SPLIT_MAX_BATCH`` requests run as the requests ``fa3_invariant_requests`` gives, in calls of at most
    that many requests, each with its own metadata. With ``FIXED_SPLITS`` and ``WINDOW_ROWS``, the one-row requests of
    a sliding-window layer go in calls of their own on FA3's causal kernel, which start each row's keys at its window
    start (``fa3_window_start_rows``); a decode step of at most that many requests, which full CUDA graphs capture, runs
    as one such call with window starts computed on the device from its key lengths. Other decode steps of at most that
    many requests run as built. The calls are built once per step's metadata and window.
    """
    if _REQUEST_PLAN in _installed:
        return
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
    _installed.add(_REQUEST_PLAN)


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


def _invariant_calls(impl, metadata):
    """The FA3 calls of ``impl``'s layer for the step of ``metadata``: ``(query rows, metadata, window call)``
    triples, the window call ``(window, window starts)`` for a call of one-row requests on FA3's causal kernel, or
    ``None`` when the step's call runs as built."""
    left, right = impl.sliding_window
    window = left + 1 if WINDOW_ROWS in _installed and left >= 0 and right == 0 else None
    fixed = FIXED_SPLITS in _installed
    causal_rows = fixed and window is not None
    cached = metadata.__dict__.setdefault("_skyrl_calls", {})
    if window in cached:
        return cached[window]
    requests = metadata.query_start_loc.numel() - 1
    single_rows = metadata.max_query_len <= 1
    if causal_rows and single_rows and requests <= FA3_DYNAMIC_SPLIT_MAX_BATCH:
        # A decode step: the window starts follow the key lengths on the device, so a captured CUDA graph replays them.
        # Every KV-cache group's metadata of a step holds the step's one key-length tensor, so the step computes them
        # once for all sliding-window layers.
        by_window = metadata.seq_lens.__dict__.setdefault("_skyrl_window_starts", {})
        if window not in by_window:
            by_window[window] = torch.clamp(metadata.seq_lens[:requests] - window, min=0)
        cached[window] = [(slice(None), metadata, (window, by_window[window]))]
        return cached[window]
    needs_window_rows = window is not None and not single_rows and metadata.max_seq_len > window
    needs_alignment = fixed and not single_rows
    if not (needs_window_rows or needs_alignment or (fixed and requests > FA3_DYNAMIC_SPLIT_MAX_BATCH)):
        cached[window] = None
        return None
    if metadata.use_cascade or metadata.dcp_context_kv_lens is not None:
        raise NotImplementedError("decode-invariant FA3 requests rewrite plain FA3 varlen steps only")
    group = impl.num_heads // impl.num_kv_heads
    if FA3_BLOCK_M % group:
        raise NotImplementedError(f"decode-invariant FA3 requests need query heads per KV head dividing {FA3_BLOCK_M}")
    query_start = metadata.query_start_loc.tolist()
    planned = fa3_invariant_requests(
        query_start, metadata.seq_lens[:requests].tolist(), window, FA3_BLOCK_M // group if fixed else 1
    )
    total = planned.key_lengths.numel()
    if total == requests and (not fixed or total <= FA3_DYNAMIC_SPLIT_MAX_BATCH):
        cached[window] = None
        return None
    device = metadata.query_start_loc.device
    starts = planned.query_start.tolist()
    calls = []
    groups = fa3_request_calls(
        starts, one_row_calls=causal_rows, max_requests=FA3_DYNAMIC_SPLIT_MAX_BATCH if fixed else total
    )
    for begin, end, one_row in groups:
        key_lengths = planned.key_lengths[begin:end]
        call = copy.copy(metadata)
        call.num_actual_tokens = starts[end] - starts[begin]
        call.query_start_loc = (planned.query_start[begin : end + 1] - starts[begin]).to(device)
        call.seq_lens = key_lengths.to(device)
        call.block_table = metadata.block_table[planned.owner[begin:end].to(device)]
        call.max_query_len = max(b - a for a, b in zip(starts[begin:end], starts[begin + 1 : end + 1], strict=True))
        call.max_seq_len = int(key_lengths.max())
        call.scheduler_metadata = (
            fa3_fixed_split_metadata(call.query_start_loc, group, impl.num_kv_heads) if fixed else None
        )
        window_call = (window, torch.clamp(key_lengths - window, min=0).to(device)) if one_row else None
        calls.append((slice(starts[begin], starts[end]), call, window_call))
    cached[window] = calls
    return calls


def _router() -> None:
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
    """Raise unless every materialized router weight holds bf16 values (the invariant router GEMM multiplies in bf16).

    A weight sync loads into meta parameters and materializes them when it finishes, so the worker checks again then.
    """
    for name, parameter in model.named_parameters():
        if name.endswith(".mlp.router.weight") and not parameter.is_meta:
            check_bf16_values(name, parameter.data)


def _deterministic_autotune() -> None:
    """Inductor's autotuner ranks a kernel's configs by a fixed order (largest reduction block, then warps, then
    rows per program, then stages) instead of timing them, so every engine process launches the same config."""

    def benchmark_all_configs(self, *args, **kwargs):
        ranked = sorted(self.launchers, key=launcher_preference)
        return {launcher: float(rank) for rank, launcher in enumerate(ranked)}

    CachingAutotuner.benchmark_all_configs = benchmark_all_configs


_PATCHES = {FIXED_SPLITS: _fixed_splits, WINDOW_ROWS: _window_rows, ROUTER: _router, AUTOTUNE: _deterministic_autotune}


def installed_parts() -> list[str]:
    """The parts installed in this process, in ``PARTS`` order."""
    return [part for part in PARTS if part in _installed]


def install(parts: Iterable[str] = PARTS) -> None:
    """Patch vLLM in this process with the named parts; each part applies once."""
    for part in parts:
        if part not in _PATCHES:
            raise ValueError(f"unknown decode-invariant part {part!r}; choose from {PARTS}")
        if part not in _installed:
            _PATCHES[part]()
            _installed.add(part)
