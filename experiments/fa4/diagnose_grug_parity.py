"""Locate the existing long-context Grug HF/Megatron parity failure.

Run one mode per process. The ablations remove the attention or MoE contribution
from the same saved checkpoint on both sides of the comparison.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.models.grug_moe import GrugMoeForCausalLM
from skyrl_train.utils import initialize_ray
from tests.gpu.test_grug_megatron import (
    SNOWBALL_LIKE_SHAPE,
    _config,
    _hf_response_logprobs,
    _init_policy,
    _megatron_response_logprobs,
    _padded_batch,
    _write_tiny_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("flash_attention_2", "flash_attention_4"), required=True)
    parser.add_argument(
        "--mode", choices=("full", "no_moe", "no_attention", "fixed_routes", "fixed_high_routes"), required=True
    )
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--prompt-length", type=int, default=2400)
    parser.add_argument("--response-length", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = {**SNOWBALL_LIKE_SHAPE, "sliding_window": args.window}

    with tempfile.TemporaryDirectory(prefix="grug-parity-") as directory:
        model_path = Path(directory) / "model"
        model_path.mkdir()
        _write_tiny_checkpoint(model_path, max_position_embeddings=16384, num_experts_per_tok=4, shape=shape)

        if args.mode != "full":
            model = GrugMoeForCausalLM.from_pretrained(model_path, dtype=torch.float32)
            with torch.no_grad():
                for layer in model.model.layers:
                    if args.mode == "no_attention":
                        layer.self_attn.o_proj.weight.zero_()
                    elif args.mode == "no_moe":
                        for name in ("gate_proj", "up_proj", "down_proj"):
                            getattr(layer.mlp.experts, name).weight.zero_()
                            getattr(layer.shared_expert, name).weight.zero_()
                    else:
                        # Keep MoE arithmetic while removing top-k flips from small logit differences.
                        layer.mlp.router.bias.fill_(-1000)
                        expert_start = 0 if args.mode == "fixed_routes" else layer.mlp.router.num_experts - 5
                        layer.mlp.router.bias[expert_start : expert_start + 5] = torch.tensor([1000, 900, 800, 700, 600])
            model.save_pretrained(model_path, safe_serialization=True)
            del model

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        batch = _padded_batch(
            tokenizer.pad_token_id,
            prompt_length=args.prompt_length,
            response_length=args.response_length,
            variable_lengths=True,
        )
        cfg = _config(str(model_path), world_size=1, pp=1, ep=1)
        cfg.trainer.attn_backend = args.backend
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        initialize_ray(cfg)
        try:
            expected = ray.get(_hf_response_logprobs.remote(str(model_path), batch))
            actual = _megatron_response_logprobs(_init_policy(cfg, 1), batch)
            valid = batch["response_mask"].bool()
            difference = (actual[valid] - expected[valid]).abs()
            result = {
                "backend": args.backend,
                "mode": args.mode,
                "shape": shape,
                "prompt_length": args.prompt_length,
                "response_length": args.response_length,
                "valid_tokens": difference.numel(),
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
