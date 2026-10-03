"""What a decode-invariant vLLM engine (``inference_engines/vllm/decode_invariant.py``) runs on."""

from collections.abc import Mapping
from typing import Any

# vLLM's attention backend for a decode-invariant engine: the engine patches FA3's calls.
DECODE_INVARIANT_ATTENTION_BACKEND = "FLASH_ATTN"
# The FlashAttention version the engine's fixed splits and window starts are written for, which vLLM selects on Hopper
# when ``attention_config.flash_attn_version`` is unset.
DECODE_INVARIANT_FLASH_ATTN_VERSION = 3


def flash_attn_version_override(engine_init_kwargs: Mapping[str, Any]) -> int | None:
    """The FlashAttention version ``engine_init_kwargs`` forces on vLLM (``attention_config.flash_attn_version``),
    or None."""
    return (engine_init_kwargs.get("attention_config") or {}).get("flash_attn_version")


def decode_invariant_engine_problems(
    *,
    backend: str,
    attention_backend: str | None,
    flash_attn_version: int | None,
    enforce_eager: bool,
    tensor_parallel_size: int,
    decode_context_parallel_size: int,
) -> list[str]:
    """The ``generator`` settings a decode-invariant engine needs that these engine settings lack."""
    problems = []
    if backend != "vllm":
        problems.append("generator.backend=vllm")
    if attention_backend != DECODE_INVARIANT_ATTENTION_BACKEND:
        problems.append(f"generator.vllm_attention_backend={DECODE_INVARIANT_ATTENTION_BACKEND}")
    if flash_attn_version not in (None, DECODE_INVARIANT_FLASH_ATTN_VERSION):
        problems.append(
            "generator.engine_init_kwargs.attention_config.flash_attn_version unset or "
            f"{DECODE_INVARIANT_FLASH_ATTN_VERSION}"
        )
    if enforce_eager:
        problems.append("generator.enforce_eager=false")
    if tensor_parallel_size != 1:
        problems.append("generator.inference_engine_tensor_parallel_size=1")
    if decode_context_parallel_size != 1:
        problems.append("generator.inference_engine_decode_context_parallel_size=1")
    return problems
