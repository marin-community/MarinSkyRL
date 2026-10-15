"""The engine settings a decode-invariant vLLM engine needs, and the vLLM engine kwargs it adds to them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# vLLM's CUSTOM attention backend slot, which ``plugin.register_attention_backend`` points at
# ``attention_backend.DecodeInvariantAttentionBackend`` in every vLLM process.
DECODE_INVARIANT_ATTENTION_BACKEND = "CUSTOM"
# The attention backend a configuration may name for an engine that is built decode-invariant.
STOCK_ATTENTION_BACKEND = "FLASH_ATTN"
DECODE_INVARIANT_MODEL_CLASS = (
    "skyrl_train.inference_engines.vllm.decode_invariant.model:DecodeInvariantGrugMoeForCausalLM"
)
FLASH_ATTN_VERSION = 3
MOE_BACKEND = "triton"
ALL2ALL_BACKEND = "allgather_reducescatter"
# Inductor chooses each reduction kernel's launch configuration by a fixed rule instead of timing the candidates,
# pads every GEMM it would otherwise pad only after timing, and builds no combo kernels, whose pass times candidate
# kernels (vLLM turns the pass on when the configuration leaves the key unset).
INDUCTOR_COMPILE_CONFIG = {"deterministic": True, "force_shape_pad": True, "combo_kernels": False}
# Every step of at most 512 tokens runs in a CUDA graph padded to the next size; no size is 1, because vLLM's
# compiled graph gives a token computed alone different floating-point values than the same token in a step of two or
# more. The sizes are vLLM's default sizes for a 512-token ceiling without 1; vLLM drops the sizes above
# ``max_num_batched_tokens``.
CUDA_GRAPH_MODE = "FULL_AND_PIECEWISE"
CUDA_GRAPH_CAPTURE_SIZES = (2, 4, *range(8, 256, 8), *range(256, 513, 16))
# Engine kwargs ``decode_invariant_engine_init_kwargs`` sets.
RESERVED_ENGINE_INIT_KWARGS = ("attention_backend", "compilation_config", "model_class_overrides")


def verify_decode_invariant_config(
    *,
    backend: str,
    attention_backend: str | None,
    enforce_eager: bool,
    tensor_parallel_size: int,
    decode_context_parallel_size: int,
    engine_init_kwargs: Mapping[str, Any],
) -> None:
    """Raise ValueError unless these engine settings can build a decode-invariant engine."""
    if backend != "vllm":
        raise ValueError(f"a decode-invariant engine needs generator.backend=vllm, got {backend!r}")
    if attention_backend not in (None, STOCK_ATTENTION_BACKEND):
        raise ValueError(
            f"a decode-invariant engine needs generator.vllm_attention_backend unset or {STOCK_ATTENTION_BACKEND}, "
            f"got {attention_backend!r}"
        )
    if enforce_eager:
        raise ValueError("a decode-invariant engine needs generator.enforce_eager=false")
    if tensor_parallel_size != 1:
        raise ValueError(f"a decode-invariant engine needs tensor_parallel_size=1, got {tensor_parallel_size}")
    if decode_context_parallel_size != 1:
        raise ValueError(
            f"a decode-invariant engine needs decode_context_parallel_size=1, got {decode_context_parallel_size}"
        )
    reserved = [key for key in RESERVED_ENGINE_INIT_KWARGS if key in engine_init_kwargs]
    if reserved:
        raise ValueError(f"a decode-invariant engine sets engine_init_kwargs {', '.join(reserved)} itself")
    flash_attn_version = (engine_init_kwargs.get("attention_config") or {}).get("flash_attn_version")
    if flash_attn_version not in (None, FLASH_ATTN_VERSION):
        raise ValueError(
            "a decode-invariant engine needs engine_init_kwargs.attention_config.flash_attn_version unset or "
            f"{FLASH_ATTN_VERSION}, got {flash_attn_version}"
        )
    moe_backend = engine_init_kwargs.get(
        "moe_backend", (engine_init_kwargs.get("kernel_config") or {}).get("moe_backend")
    )
    if moe_backend != MOE_BACKEND:
        raise ValueError(
            f"a decode-invariant engine needs engine_init_kwargs.moe_backend={MOE_BACKEND}, got {moe_backend!r}"
        )
    all2all_backend = engine_init_kwargs.get("all2all_backend", ALL2ALL_BACKEND)
    if all2all_backend != ALL2ALL_BACKEND:
        raise ValueError(
            f"a decode-invariant engine needs engine_init_kwargs.all2all_backend={ALL2ALL_BACKEND}, got {all2all_backend!r}"
        )
    if engine_init_kwargs.get("speculative_config") is not None:
        raise ValueError("a decode-invariant engine runs without speculative decoding")
    if engine_init_kwargs.get("logits_processors"):
        raise ValueError(
            "a decode-invariant engine runs vLLM's model runner V2, which engine_init_kwargs.logits_processors "
            "turns off"
        )


def decode_invariant_engine_environment() -> dict[str, str]:
    """The environment a decode-invariant engine's processes start with: without NVLink SHARP or vLLM's NCCL symmetric
    memory, the expert-parallel combine runs NCCL's ring reduce-scatter, in the order ``models/grug_exact``'s
    ``ep_combine`` reproduces."""
    return {"NCCL_NVLS_ENABLE": "0", "VLLM_USE_NCCL_SYMM_MEM": "0"}


def decode_invariant_engine_init_kwargs() -> dict[str, Any]:
    """The vLLM engine kwargs that make an engine decode-invariant, to add to its ``engine_init_kwargs``, for an
    engine whose ``attention_backend`` is ``DECODE_INVARIANT_ATTENTION_BACKEND``."""
    # Imported here: skyrl_train.models imports skyrl_train.utils, whose config validation imports this module.
    from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE  # noqa: PLC0415

    return {
        "model_class_overrides": {GRUG_MOE_ARCHITECTURE: DECODE_INVARIANT_MODEL_CLASS},
        "compilation_config": {
            "inductor_compile_config": dict(INDUCTOR_COMPILE_CONFIG),
            "cudagraph_mode": CUDA_GRAPH_MODE,
            "cudagraph_capture_sizes": list(CUDA_GRAPH_CAPTURE_SIZES),
        },
    }
