"""What a decode-invariant vLLM engine (``inference_engines/vllm/decode_invariant.py``) runs on."""

# vLLM's attention backend for a decode-invariant engine: the engine patches FA3's calls.
DECODE_INVARIANT_ATTENTION_BACKEND = "FLASH_ATTN"


def decode_invariant_engine_problems(
    *,
    backend: str,
    attention_backend: str | None,
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
    if enforce_eager:
        problems.append("generator.enforce_eager=false")
    if tensor_parallel_size != 1:
        problems.append("generator.inference_engine_tensor_parallel_size=1")
    if decode_context_parallel_size != 1:
        problems.append("generator.inference_engine_decode_context_parallel_size=1")
    return problems
