"""Check that a Qwen3.5 adapter can load into the stock fused-QKV model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open

SPLIT_QKV_TARGETS = {"in_proj_q", "in_proj_k", "in_proj_v"}
FUSED_QKV_TARGET = "in_proj_qkv"


@dataclass(frozen=True)
class FusedAdapterConfig:
    base_model_name_or_path: str | None
    rank: int
    alpha: int
    target_modules: frozenset[str]


def verify_fused_qkv_adapter(adapter_path: Path) -> FusedAdapterConfig:
    """Return the verified adapter shape, rejecting split-QKV weights that PEFT would partly load."""
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    targets = set(config["target_modules"])
    if targets & SPLIT_QKV_TARGETS or FUSED_QKV_TARGET not in targets:
        raise ValueError("Qwen3.5 adapter must use fused in_proj_qkv; convert split Q/K/V LoRA before loading")
    rank = config["r"]
    alpha = config["lora_alpha"]
    if config.get("rank_pattern", {}).get(FUSED_QKV_TARGET) != 3 * rank:
        raise ValueError("Qwen3.5 fused QKV LoRA rank must be three times the split rank")
    if config.get("alpha_pattern", {}).get(FUSED_QKV_TARGET) != 3 * alpha:
        raise ValueError("Qwen3.5 fused QKV LoRA alpha must preserve the split projection scaling")
    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt") as source:
        keys = source.keys()
    if any(f".{target}." in key for target in SPLIT_QKV_TARGETS for key in keys):
        raise ValueError("Qwen3.5 adapter still contains split Q/K/V LoRA weights")
    if not any(f".{FUSED_QKV_TARGET}." in key for key in keys):
        raise ValueError("Qwen3.5 adapter has no fused QKV LoRA weights")
    return FusedAdapterConfig(
        base_model_name_or_path=config.get("base_model_name_or_path"),
        rank=rank,
        alpha=alpha,
        target_modules=frozenset(targets),
    )
