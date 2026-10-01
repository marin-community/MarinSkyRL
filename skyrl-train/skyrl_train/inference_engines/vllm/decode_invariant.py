"""A decode-invariant compiled vLLM engine for Grug: each token's bytes do not depend on the step that computes it.

Compiled vLLM gives a token other bytes in a decode step than in a prefill step, or in steps of another composition,
in these places (a one-H100 sweep of each kernel at Snowball's shapes, one row to 8,192 rows per call):

- FA3 splits a request's keys by the step: 32 splits at most on CUDA-graph steps (a per-request count on the device),
  FA3's heuristic above. Rows computed with another split count add their key blocks in another order. Every FA3
  call runs one split (``ONE_SPLIT``).
- On a sliding-window layer FA3 aligns its key blocks to the window start of each query tile's first row. A decode
  row is a tile of its own; inside a prefill tile the same row past the window adds other block groups. Prefill rows
  past the window run as one-row requests, as decode steps run them (``WINDOW_ROWS``).
- cuBLAS picks the fp32 router GEMM's kernel from the step's row count. A Triton GEMM with one tile shape computes the
  router logits instead (``ROUTER``).
- The 20-output head-gate GEMM: Inductor's ``pad_mm`` pass times it against a zero-padded 24-output copy when each
  engine process compiles, and the unpadded GEMM's cuBLAS kernel follows the row count. Every process compiles the
  padded GEMM: the engine's compilation setting ``force_shape_pad``, which ``create_ray_wrapped_inference_engines``
  sets for a decode-invariant engine.
- Inductor's autotuner picks some reductions' launch configs by timing them in each process. Every process takes the
  same config (``AUTOTUNE``), so the re-read engine's recorded configs hold for every engine.

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

from skyrl_train.models.grug_invariant_kernels import check_bf16_values, invariant_router_logits
from skyrl_train.models.grug_vllm_kernels import window_row_requests

ONE_SPLIT = "one_split"
WINDOW_ROWS = "window_rows"
ROUTER = "router"
AUTOTUNE = "autotune"
PARTS = (ONE_SPLIT, WINDOW_ROWS, ROUTER, AUTOTUNE)

_installed: set[str] = set()


@torch.library.custom_op("skyrl::grug_router_logits", mutates_args=())
def grug_router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grug's router logits from the row-invariant GEMM, an opaque op inside vLLM's compiled graph."""
    return invariant_router_logits(x, weight)


@grug_router_logits.register_fake
def _grug_router_logits_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


def _one_split() -> None:
    """Every FA3 call of the engine runs one split, on CUDA-graph steps (captured with it) and on eager steps."""
    builder = flash_attn.FlashAttentionMetadataBuilder
    original_init, original_build = builder.__init__, builder.build

    def one_split_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.max_num_splits = 1

    def one_split_build(self, *args, **kwargs):
        metadata = original_build(self, *args, **kwargs)
        metadata.max_num_splits = 1
        return metadata

    builder.__init__, builder.build = one_split_init, one_split_build


def _window_rows() -> None:
    """On sliding-window layers, prefill rows past the window run as one-row requests (``window_row_requests``).

    Decode steps already run each row alone, and steps whose sequences all end within the window need nothing; the
    rewrite happens only on steps with more than one query row of some request and a key length past the window,
    which vLLM runs outside CUDA graphs. The new requests are built once per step's metadata and window.
    """
    original_forward = flash_attn.FlashAttentionImpl.forward

    def window_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        left, right = self.sliding_window
        if attn_metadata is not None and left >= 0 and right == 0:
            window = left + 1
            if attn_metadata.max_query_len > 1 and attn_metadata.max_seq_len > window:
                attn_metadata = _window_row_metadata(attn_metadata, window)
        return original_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)

    flash_attn.FlashAttentionImpl.forward = window_forward


def _window_row_metadata(metadata, window: int):
    cached = getattr(metadata, "_skyrl_window_rows", None)
    if cached is not None and cached[0] == window:
        return cached[1]
    if metadata.use_cascade or metadata.dcp_context_kv_lens is not None or metadata.scheduler_metadata is not None:
        raise NotImplementedError("window_rows rewrites plain FA3 varlen steps only")
    requests = window_row_requests(metadata.query_start_loc, metadata.seq_lens, window)
    rewritten = copy.copy(metadata)
    rewritten.query_start_loc = requests.query_start
    rewritten.seq_lens = requests.key_lengths
    rewritten.block_table = metadata.block_table[requests.owner]
    metadata._skyrl_window_rows = (window, rewritten)
    return rewritten


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


def _launcher_preference(launcher) -> tuple:
    config = launcher.config
    return (
        -config.kwargs.get("R0_BLOCK", 0),
        -config.num_warps,
        -config.kwargs.get("XBLOCK", 0),
        -config.num_stages,
    )


def _deterministic_autotune() -> None:
    """Inductor's autotuner ranks a kernel's configs by a fixed order (largest reduction block, then warps, then
    rows per program, then stages) instead of timing them, so every engine process launches the same config."""

    def benchmark_all_configs(self, *args, **kwargs):
        ranked = sorted(self.launchers, key=_launcher_preference)
        return {launcher: float(rank) for rank, launcher in enumerate(ranked)}

    CachingAutotuner.benchmark_all_configs = benchmark_all_configs


_PATCHES = {ONE_SPLIT: _one_split, WINDOW_ROWS: _window_rows, ROUTER: _router, AUTOTUNE: _deterministic_autotune}


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
