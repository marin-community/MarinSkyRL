"""Seeded Grug checkpoint and fixed prompts for a small RL probe run."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from marinskyrl.model_manifest import write_local_model_manifest
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
NUM_LAYERS = 8
NUM_EXPERTS = 8
TOY_SHAPE = dict(
    hidden_size=64,
    intermediate_size=64,
    shared_expert_intermediate_size=64,
    num_local_experts=NUM_EXPERTS,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    sliding_window=16,
)

TRAIN_PROMPTS = (
    "Write a short phrase about a blue square.",
    "Write a short phrase about a red circle.",
    "Write a short phrase about a green triangle.",
    "Write a short phrase about a yellow star.",
)
VALIDATION_PROMPTS = (
    "Write a short phrase about a silver moon.",
    "Write a short phrase about a golden sun.",
)


def write_tiny_checkpoint(
    path: Path,
    max_position_embeddings: int = 128,
    num_experts_per_tok: int = 2,
    shape: dict | None = None,
    vocab_size_multiple: int = 1,
) -> None:
    """Save the seeded Grug checkpoint shared by GPU and RL integration tests."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
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
        for module in model.modules():
            if module.__class__.__name__ == "GrugMoeGatedNorm":
                module.down_proj.weight.normal_(std=0.2)
                module.up_proj.weight.normal_(std=0.2)
        for layer in model.model.layers:
            layer.self_attn.attn_gate.weight.normal_(std=0.2)
            layer.mlp.router.bias.copy_(torch.linspace(-0.3, 0.3, config.num_local_experts))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def write_tiny_probe_fixture(path: Path) -> None:
    """Write one model and disjoint fixed train and validation prompt files."""
    model_path = path / "model"
    data_path = path / "data"
    model_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    write_tiny_checkpoint(model_path)
    write_local_model_manifest(model_path)
    for name, prompts in (("train", TRAIN_PROMPTS), ("validation", VALIDATION_PROMPTS)):
        rows = [
            {"prompt": [{"role": "user", "content": prompt}], "env_class": "mismatch_fixture"} for prompt in prompts
        ]
        Dataset.from_list(rows).to_parquet(str(data_path / f"{name}.parquet"))
