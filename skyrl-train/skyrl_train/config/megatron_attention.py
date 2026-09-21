"""Run-level Transformer Engine attention selection for Megatron workers."""

from enum import StrEnum


class MegatronAttentionBackend(StrEnum):
    FA2 = "flash_attention_2"
    FA4 = "flash_attention_4"
    FUSED = "fused"


def resolve_megatron_attention_backend(configured: str, legacy_flash_attention: bool) -> MegatronAttentionBackend:
    """Resolve the existing trainer attention setting for Megatron."""
    if configured == "auto":
        return MegatronAttentionBackend.FA2 if legacy_flash_attention else MegatronAttentionBackend.FUSED
    try:
        backend = MegatronAttentionBackend(configured)
    except ValueError as error:
        raise ValueError(
            f"Megatron trainer.attn_backend={configured!r} is unsupported; choose flash_attention_4, "
            "flash_attention_2, or fused"
        ) from error
    if legacy_flash_attention:
        raise ValueError("Set trainer.flash_attn=false when trainer.attn_backend selects a Megatron backend")
    return backend


def transformer_engine_attention_environment(configured: str, legacy_flash_attention: bool) -> dict[str, str]:
    """Allow only the selected TE backend in a worker process."""
    backend = resolve_megatron_attention_backend(configured, legacy_flash_attention)
    if configured == "auto" and backend is MegatronAttentionBackend.FUSED:
        return {}
    if backend is MegatronAttentionBackend.FUSED:
        return {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"}
    return {
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FLASH_ATTN_V2": "1" if backend is MegatronAttentionBackend.FA2 else "0",
        "NVTE_FLASH_ATTN_V3": "0",
        "NVTE_FLASH_ATTN_V4": "1" if backend is MegatronAttentionBackend.FA4 else "0",
        "NVTE_FUSED_ATTN": "0",
        "NVTE_UNFUSED_ATTN": "0",
    }
