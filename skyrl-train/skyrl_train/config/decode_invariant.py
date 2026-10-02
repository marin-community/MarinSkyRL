"""The policy's decode-invariant vLLM engines (``generator.decode_invariant``)."""

from omegaconf import DictConfig

# vLLM's attention backend for a decode-invariant engine: the engine patches FA3's calls.
DECODE_INVARIANT_ATTENTION_BACKEND = "FLASH_ATTN"


def validate_decode_invariant_config(cfg: DictConfig) -> None:
    """Accept a decode-invariant engine only in the geometry it is verified in: compiled vLLM engines this run starts,
    with FlashAttention, one GPU per tensor- and context-parallel group."""

    if not cfg.generator.decode_invariant:
        return
    generator = cfg.generator
    if generator.backend != "vllm" or not generator.run_engines_locally:
        raise ValueError("generator.decode_invariant=true needs vLLM engines that this run starts")
    if generator.get("vllm_attention_backend") != DECODE_INVARIANT_ATTENTION_BACKEND:
        raise ValueError(
            f"generator.decode_invariant=true needs generator.vllm_attention_backend={DECODE_INVARIANT_ATTENTION_BACKEND}"
        )
    if generator.enforce_eager:
        raise ValueError("generator.decode_invariant=true needs compiled engines (generator.enforce_eager=false)")
    if (
        generator.inference_engine_tensor_parallel_size != 1
        or generator.get("inference_engine_decode_context_parallel_size", 1) != 1
    ):
        raise ValueError(
            "generator.decode_invariant=true needs inference_engine_tensor_parallel_size=1 and "
            "inference_engine_decode_context_parallel_size=1"
        )
