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
    parser.add_argument("--cp-comm-type", choices=("default", "p2p", "all_gather"), default="all_gather")
    parser.add_argument("--capture-grads", action="store_true")
    parser.add_argument("--diagnose-parity-failure", action="store_true")
    args = parser.parse_args()
    if os.getenv("NVTE_FLASH_ATTN_V4") not in ("0", "1"):
        raise ValueError("Set NVTE_FLASH_ATTN_V4=0 for FA2 or 1 for FA4")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.capture_grads:
        os.environ["FA4_EXPERIMENT_GRAD_SNAPSHOT"] = "1"

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
        if args.world_size > 1 and args.cp_comm_type != "default":
            # TE 2.19 rejects Grug's sliding window with its default p2p CP
            # transport. The older baseline may require its own default mode.
            OmegaConf.update(
                cfg.trainer.policy.megatron_config.transformer_config_kwargs,
                "cp_comm_type",
                args.cp_comm_type,
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
            parity_passed = True
            try:
                _assert_logprobs_close(before_logprobs, expected, batch["response_mask"])
            except AssertionError:
                if not args.diagnose_parity_failure:
                    raise
                parity_passed = False
                print("PARITY_GATE_FAILED: continuing only to record diagnostic FA2/FA4 evidence", flush=True)
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
                if not all(math.isfinite(value) for value in statuses[-1].values()):
                    break
            after_weights = rank0_validation_snapshot(policy, names)
            weight_health = {
                name: {
                    "nonfinite": int((~torch.isfinite(after_weights[name])).sum().item()),
                    "changed": int((after_weights[name] != before_weights[name]).sum().item()),
                }
                for name in names
            }
            print(f"WEIGHT_HEALTH {weight_health!r}", flush=True)
            gradients = {}
            if args.capture_grads:
                snapshots = ray.get(policy.async_run_ray_method("pass_through", "fa4_experiment_gradient_snapshot"))
                gradients = next(snapshot["grads"] for snapshot in snapshots if snapshot["rank"] == 0)
                if not gradients or not any(torch.count_nonzero(grad).item() for grad in gradients.values()):
                    raise AssertionError("Gradient capture returned no nonzero attention gradients")
                print(
                    "GRADIENT_HEALTH",
                    {
                        name: {
                            "nonzero": int(torch.count_nonzero(grad).item()),
                            "nonfinite": int((~torch.isfinite(grad)).sum().item()),
                            "max_abs": float(grad.abs().max().item()),
                        }
                        for name, grad in gradients.items()
                    },
                    flush=True,
                )
            weights_finite = all(health["nonfinite"] == 0 for health in weight_health.values())
            after_logprobs = (
                _megatron_response_logprobs(policy, batch)
                if weights_finite
                else torch.full_like(before_logprobs, float("nan"))
            )
            memory = ray.get(policy.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
            result = {
                "world_size": args.world_size,
                "context_parallel_size": args.world_size,
                "sample_packing": bool(cfg.trainer.use_sample_packing),
                "cp_comm_type": args.cp_comm_type if args.world_size > 1 else "not_applicable",
                "shape": args.shape,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "steps": args.steps,
                "torch": str(torch.__version__),
                "nvte_flash_attn_v4": os.environ["NVTE_FLASH_ATTN_V4"],
                "forward_seconds": forward_seconds,
                "training_seconds": training_seconds,
                "statuses": statuses,
                "weight_health": weight_health,
                "captured_gradients": bool(gradients),
                "post_forward_skipped_due_to_nonfinite_weights": not weights_finite,
                "parity_passed": parity_passed,
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
                    "gradients": gradients,
                },
                args.output,
            )
            print(json.dumps(result, indent=2))
            assert parity_passed, "HF parity gate failed; diagnostic data are not a pass"
            assert len(statuses) == args.steps
            assert all(math.isfinite(status["policy_loss"]) for status in statuses)
            # Grug disables clipping, so a zero norm may be a missing metric;
            # nonfinite norms remain a failed numeric gate.
            assert all(0 <= status["raw_grad_norm"] < float("inf") for status in statuses)
            assert all(status["policy_update_steps"] == 1 for status in statuses)
            assert weights_finite and torch.isfinite(before_logprobs).all() and torch.isfinite(after_logprobs).all()
            assert any(health["changed"] > 0 for health in weight_health.values())
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
