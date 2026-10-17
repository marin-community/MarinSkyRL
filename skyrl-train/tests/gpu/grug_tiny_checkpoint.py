"""Tiny Grug checkpoint fixture shared by the Grug GPU tests."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM
from skyrl_train.models.grug_moe import GRUG_ROUTER_BIAS_SUFFIX

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TOY_SHAPE = dict(
    hidden_size=64,
    intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_local_experts=8,
    num_hidden_layers=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    sliding_window=16,
)


def router_bias_names(num_layers: int) -> list[str]:
    """Per-layer router-bias buffer names, which carry no gradient and are excluded."""
    return [f"model.layers.{index}{GRUG_ROUTER_BIAS_SUFFIX}" for index in range(num_layers)]


def write_tiny_checkpoint(
    path: Path,
    max_position_embeddings: int = 128,
    num_experts_per_tok: int = 2,
    shape: dict | None = None,
    vocab_size_multiple: int = 1,
) -> None:
    """Write the deterministic tiny Grug checkpoint used across the GPU gates."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    shape = TOY_SHAPE if shape is None else shape
    config = GrugMoeConfig(
        vocab_size=((len(tokenizer) + vocab_size_multiple - 1) // vocab_size_multiple) * vocab_size_multiple,
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=max_position_embeddings,
        initializer_range=0.02,
        qk_mult=1.37,
        qk_mult_long_scale=1.1,
        **shape,
    )
    torch.manual_seed(17)
    model = GrugMoeForCausalLM(config)
    with torch.no_grad():
        # Non-trivial gates and router biases so the Megatron port has to reproduce them.
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def write_tiny_hero_checkpoint(path: Path):
    """Create all Hero parameter families, retaining an ordinary saved tokenizer."""
    shape = dict(
        hidden_size=128,
        intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_local_experts=16,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        sliding_window=8,
    )
    write_tiny_checkpoint(path, shape=shape, num_experts_per_tok=8)
    config = GrugMoeConfig.from_pretrained(path)
    state = load_file(str(path / "model.safetensors"))
    torch.manual_seed(23)
    # The borrowed tokenizer has an odd vocabulary; Hero's real vocabulary is
    # divisible by the supported tensor-parallel sizes.
    vocab_size = (config.vocab_size + 127) // 128 * 128
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        state[name] = torch.nn.functional.pad(state[name], (0, 0, 0, vocab_size - config.vocab_size))
    config.vocab_size = vocab_size
    latent = 64
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for projection, dims in (("gate_proj", (64, latent)), ("up_proj", (64, latent)), ("down_proj", (latent, 64))):
            del state[f"{prefix}.mlp.experts.{projection}.weight"]
            for expert in range(config.num_local_experts):
                state[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = torch.randn(*dims) * 0.1
        for projection in ("gate_proj", "up_proj", "down_proj"):
            original = state.pop(f"{prefix}.shared_expert.{projection}.weight")
            for expert in range(2):
                state[f"{prefix}.shared_experts.{expert}.{projection}.weight"] = torch.randn_like(original) * 0.1
        state[f"{prefix}.mlp.latent_down_proj.weight"] = torch.randn(latent, config.hidden_size) * 0.1
        state[f"{prefix}.mlp.latent_up_proj.weight"] = torch.randn(config.hidden_size, latent) * 0.1
        state[f"{prefix}.mlp.latent_norm.weight"] = torch.ones(latent)
        for site, channels in (("self_attn.sconv_k", 64), ("sconv_attn", 128), ("sconv_mlp", 128)):
            weight = torch.randn(4, channels) * 0.1
            weight[0] += 1
            state[f"{prefix}.{site}.weight"] = weight
    config = GrugMoeConfig(
        **{
            **config.to_dict(),
            "latent_dim": latent,
            "num_shared_experts": 2,
            "local_kv_heads": 4,
            "global_kv_heads": 2,
            "rope_fused": True,
            "sconv": True,
            "grugmoe_artifact_schema_version": 2,
        }
    )
    config.save_pretrained(path)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    return state
