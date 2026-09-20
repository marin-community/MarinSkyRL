"""Run a deterministic Grug Megatron forward and PPO update with selected FA.

Invoke in separate processes with NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 and
NVTE_FLASH_ATTN_V4=0/1. Inspect the Ray worker's Transformer Engine backend
log as proof of the selected kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from skyrl_train.utils import initialize_ray
from skyrl_train.utils.utils import validate_cfg
from tests.gpu.grug_serving import rank0_validation_snapshot
from tests.gpu.test_grug_megatron import (
    ATTN_GATE_NAME,
    LONG_LAYER_Q_NAME,
    SNOWBALL_LIKE_SHAPE,
    TOY_SHAPE,
    _assert_logprobs_close,
    _config,
    _hf_response_logprobs,
    _init_policy,
    _megatron_response_logprobs,
    _padded_batch,
    _train_step,
    _write_tiny_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--shape", choices=("toy", "snowball"), default="toy")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if os.getenv("NVTE_FLASH_ATTN_V4") not in ("0", "1"):
        raise ValueError("Set NVTE_FLASH_ATTN_V4=0 for FA2 or 1 for FA4")
    if args.steps < 1:
        raise ValueError("steps must be positive")

    shape = TOY_SHAPE if args.shape == "toy" else SNOWBALL_LIKE_SHAPE
    prompt_length, response_length = (12, 8) if args.shape == "toy" else (2400, 300)
    with tempfile.TemporaryDirectory(prefix="fa4-grug-") as directory:
        model_path = Path(directory) / "model"
        model_path.mkdir()
        _write_tiny_checkpoint(model_path, max_position_embeddings=16384, shape=shape)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        batch = _padded_batch(
            tokenizer.pad_token_id,
            prompt_length=prompt_length,
            response_length=response_length,
            variable_lengths=args.shape == "snowball",
        )
        cfg = _config(str(model_path), world_size=args.world_size, pp=1, ep=1)
        cfg.trainer.flash_attn = True
        cfg.trainer.policy.megatron_config.context_parallel_size = args.world_size
        # The Megatron wrapper requires packed sequences for CP. Grug's active
        # configuration disables packing, so CP2 here is an experimental gate.
        cfg.trainer.use_sample_packing = args.world_size > 1
        if args.world_size > 1:
            # TE 2.19 rejects Grug's sliding window with the default p2p CP
            # transport. all_gather is one of its explicit supported modes.
            OmegaConf.update(
                cfg.trainer.policy.megatron_config.transformer_config_kwargs,
                "cp_comm_type",
                "all_gather",
                force_add=True,
            )
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1
        validate_cfg(cfg)
        initialize_ray(cfg)
        try:
            expected = ray.get(_hf_response_logprobs.remote(str(model_path), batch))
            policy = _init_policy(cfg, args.world_size)
            start = time.perf_counter()
            before_logprobs = _megatron_response_logprobs(policy, batch)
            forward_seconds = time.perf_counter() - start
            _assert_logprobs_close(before_logprobs, expected, batch["response_mask"])
            names = (LONG_LAYER_Q_NAME, ATTN_GATE_NAME)
            before_weights = rank0_validation_snapshot(policy, names)
            training_seconds = []
            statuses = []
            for _ in range(args.steps):
                start = time.perf_counter()
                status = _train_step(policy, batch)
                training_seconds.append(time.perf_counter() - start)
                statuses.append(
                    {
                        name: float(status[name])
                        for name in ("policy_loss", "raw_grad_norm", "policy_update_steps")
                    }
                )
                print(f"TRAIN_STATUS {statuses[-1]!r}", flush=True)
                assert math.isfinite(statuses[-1]["policy_loss"])
                # Grug disables clipping, so MCore may report zero instead of
                # computing a norm. The weight-change check below proves update.
                assert 0 <= statuses[-1]["raw_grad_norm"] < float("inf")
                assert statuses[-1]["policy_update_steps"] == 1
            after_weights = rank0_validation_snapshot(policy, names)
            after_logprobs = _megatron_response_logprobs(policy, batch)
            memory = ray.get(policy.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
            assert torch.isfinite(before_logprobs).all() and torch.isfinite(after_logprobs).all()
            assert all(torch.isfinite(weight).all() for weight in after_weights.values())
            assert any(not torch.equal(before_weights[name], after_weights[name]) for name in names)
            result = {
                "world_size": args.world_size,
                "context_parallel_size": args.world_size,
                "sample_packing": bool(cfg.trainer.use_sample_packing),
                "cp_comm_type": cfg.trainer.policy.megatron_config.transformer_config_kwargs.get("cp_comm_type"),
                "shape": args.shape,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "steps": args.steps,
                "torch": str(torch.__version__),
                "nvte_flash_attn_v4": os.environ["NVTE_FLASH_ATTN_V4"],
                "forward_seconds": forward_seconds,
                "training_seconds": training_seconds,
                "statuses": statuses,
                "memory": memory,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "result": result,
                    "reference_logprobs": expected.cpu(),
                    "before_logprobs": before_logprobs.cpu(),
                    "after_logprobs": after_logprobs.cpu(),
                    "response_mask": batch["response_mask"].cpu(),
                    "before_weights": before_weights,
                    "after_weights": after_weights,
                },
                args.output,
            )
            print(json.dumps(result, indent=2))
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
