"""Two-rank native remainder and moment-memory qualification on a synthetic matrix.

No model or dataset is loaded. Real MCore DDP/distributed Adam performs ten
small-learning-rate updates after a separate zero-gradient decay diagnostic.
Full reconstructed FP32 masters and BF16 model bytes are compared at update ten.
Checkpoint continuation is checked only at the same two-rank DP geometry.
"""

import argparse
from datetime import timedelta
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import time

import fsspec
import torch
import torch.distributed as dist

from skyrl_train.entrypoints.startup_capture import phase
from skyrl_train.distributed.megatron.optimizer import init_megatron_optim_config, megatron_optimizer_kwargs
from skyrl_train.entrypoints.probe_megatron_optimizer_precision import (
    MATRIX_WIDTH,
    build_arm,
    checkpoint_bytes,
    declared_arm_kwargs,
    live_tensors,
    main_parameter_tensors,
    restore_checkpoint,
    tensor_inventory,
    update,
    validate_moment_inventory,
)


ARMS = (
    ("native_fp32", "native_fp32", False),
    ("aware_fp32", "precision_fp32", False),
    ("bf16_both", "precision_bf16_both", False),
    ("fp32_remainders", "precision_fp32", True),
    ("bf16_remainders", "precision_bf16_both", True),
)


def optimizer_inventory(optimizer) -> dict:
    """Measure owned masters and states, excluding aliases of the BF16 model."""
    tensors = []
    elements = 0
    for index, part in enumerate(optimizer.chained_optimizers):
        inner = part.optimizer
        for group_index, group in enumerate(inner.param_groups):
            for param_index, parameter in enumerate(group["params"]):
                elements += parameter.numel()
                prefix = f"optimizer.{index}.{group_index}.{param_index}"
                if not inner.master_weights:
                    tensors.append((prefix + ".parameter_master", parameter))
                for name, value in sorted(inner.state[parameter].items()):
                    if isinstance(value, torch.Tensor):
                        tensors.append((prefix + "." + name, value))
                for name, value in sorted(inner._scales.get(parameter, {}).items()):
                    tensors.append((prefix + ".scale." + name, value))
        tensors.append((f"optimizer.{index}.overflow", inner._dummy_overflow_buf))
    if elements != MATRIX_WIDTH**2 // dist.get_world_size():
        raise AssertionError("Unexpected owned parameter count")
    inventory = tensor_inventory(tensors)
    inventory["owned_parameters"] = elements
    inventory["bytes_per_owned_parameter"] = inventory["unique_retained_storage_bytes"] / elements
    return inventory


def tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Check every stored bit, including the sign of floating zero."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8), right.detach().cpu().contiguous().view(torch.uint8)
        )
    )


def run_arm(base: str, remainders: bool, lr: float, weight_decay: float) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model, optimizer, declared = build_arm(base, lr, weight_decay, store_param_remainders=remainders)
    for part in optimizer.chained_optimizers:
        if part.optimizer.store_param_remainders is not remainders:
            raise AssertionError("Native optimizer did not enable the declared remainder setting")
    rows = []
    for step in range(11):
        rows.append(update(model, optimizer, step, lr, weight_decay))
        if step == 5:
            saved = checkpoint_bytes(model, optimizer)
    all_tensors = live_tensors(model, optimizer)
    full_inventory = tensor_inventory(all_tensors)
    moment_dtype = "bfloat16" if base == "precision_bf16_both" else "float32"
    validate_moment_inventory(full_inventory, len(optimizer.chained_optimizers), moment_dtype, moment_dtype)
    inventory = optimizer_inventory(optimizer)
    masters = torch.cat([p.detach().flatten().cpu() for p in main_parameter_tensors(optimizer)])
    weights = model.module.weight.detach().cpu().contiguous().view(torch.int16).clone()
    expected = {name: value.detach().cpu().clone() for name, value in all_tensors}
    if remainders:
        master_rows = [r for r in inventory["tensors"] if r["name"].endswith(".master_param")]
        if not master_rows or any(r["dtype"] != "torch.int16" for r in master_rows):
            raise AssertionError("Remainder arm retained a whole master or omitted its state")
    del all_tensors, model, optimizer
    gc.collect()
    model, optimizer, _ = build_arm(base, lr, weight_decay, store_param_remainders=remainders)
    restore_checkpoint(model, optimizer, saved)
    for step in range(6, 11):
        update(model, optimizer, step, lr, weight_decay)
    actual = dict(live_tensors(model, optimizer))
    checkpoint_exact = actual.keys() == expected.keys() and all(
        tensor_bytes_equal(p, expected[name]) for name, p in actual.items()
    )
    result = {
        "declared": declared,
        "updates": rows,
        "optimizer_inventory": inventory,
        "full_inventory": full_inventory,
        "checkpoint_bytes": len(saved),
        "checkpoint_continuation_exact": checkpoint_exact,
        "master_sha256": hashlib.sha256(masters.numpy().tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
    }
    return result, masters, weights


def compare_arm_states(results: dict, snapshots: dict) -> dict:
    """Apply separate memory and every-byte update gates to observed arm states."""
    baseline = results["native_fp32"]["optimizer_inventory"]["bytes_per_owned_parameter"]
    savings = {
        name: baseline - result["optimizer_inventory"]["bytes_per_owned_parameter"] for name, result in results.items()
    }
    control_masters, control_weights = snapshots["aware_fp32"]
    remainder_masters, remainder_weights = snapshots["fp32_remainders"]
    masters_equal = torch.equal(control_masters.view(torch.int32), remainder_masters.view(torch.int32))
    weights_equal = torch.equal(control_weights, remainder_weights)
    return {
        "savings_bytes_per_owned_parameter": savings,
        "bf16_moments_memory_pass": savings["bf16_both"] >= 3.5,
        "combined_remainders_memory_pass": savings["bf16_remainders"] >= 5.5,
        "fp32_remainders_master_bits_equal_at_update10": masters_equal,
        "fp32_remainders_model_bits_equal_at_update10": weights_equal,
        "fp32_remainders_master_mismatched_elements": int(
            (control_masters.view(torch.int32) != remainder_masters.view(torch.int32)).sum()
        ),
        "fp32_remainders_model_mismatched_elements": int((control_weights != remainder_weights).sum()),
        "all_same_geometry_checkpoints_exact": all(r["checkpoint_continuation_exact"] for r in results.values()),
    }


def write_receipt(prefix: str, identity: str, name: str, value: dict) -> None:
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > 1024**2:
        raise ValueError("Native diagnostic receipt exceeds one MiB")
    path = f"{prefix}/{identity}/{name}.json"
    filesystem = fsspec.filesystem("s3", config_kwargs={"connect_timeout": 5, "read_timeout": 10})
    filesystem.pipe(path, data)
    print("REMAINDERS_RECEIPT " + json.dumps({"uri": path, "sha256": hashlib.sha256(data).hexdigest()}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--durable-prefix", required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if not 0 < args.lr <= 1e-4 or not args.durable_prefix.startswith("s3://marin-us-east-02a/marin/"):
        raise ValueError("Use a positive small learning rate and an east diagnostic prefix")
    if args.preview:
        composed = []
        for name, base, remainders in ARMS:
            for decay in (0.0, 0.01):
                declared = declared_arm_kwargs(base, remainders)
                normalized = megatron_optimizer_kwargs(
                    {"lr": args.lr, "weight_decay": decay, "max_grad_norm": 0.0}, declared
                )
                composed.append(
                    {"arm": name, "weight_decay": decay, "declared": declared, "native_optimizer_kwargs": normalized}
                )
        print(
            json.dumps(
                {
                    "arms": composed,
                    "updates": 10,
                    "decay_diagnostic_steps": 1,
                    "world_size": 2,
                    "arguments": vars(args),
                    "matrix_width": MATRIX_WIDTH,
                    "memory_gates": {"bf16_both": 3.5, "bf16_remainders": 5.5},
                    "equality": "every FP32 master and BF16 model bit at update ten; checkpoint continuation",
                },
                default=str,
            )
        )
        return
    from megatron.core import parallel_state
    from transformer_engine.pytorch.optimizers import FusedAdam

    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("This gate requires exactly two GPU ranks")
    versions = {n: importlib.metadata.version(n) for n in ("megatron-core", "transformer-engine", "torch")}
    if versions["megatron-core"] != "0.18.0" or versions["transformer-engine"] != "2.11.0":
        raise ValueError("Use the inspected MCore 0.18.0 and TE 2.11.0 pin")
    identity = os.environ["IRIS_ATTEMPT_UID"]
    phase("cuda_device_started")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    phase("nccl_init_started")
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    phase("nccl_init_finished")
    phase("gloo_init_started")
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    phase("gloo_init_finished")
    phase("model_parallel_init_started")
    parallel_state.initialize_model_parallel()
    phase("model_parallel_init_finished")
    rank = dist.get_rank()
    provenance = {
        "versions": versions,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "arguments": vars(args),
        "source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
            for name, value in {
                "entrypoint": main,
                "adapter": init_megatron_optim_config,
                "shared_probe": update,
                "te_fused_adam": FusedAdam,
            }.items()
        },
        "scope": "Synthetic DP2 actual MCore/TE optimizer; no EP/Snowball learning or checkpoint resharding claim",
        "peak_scope": "Update peaks include diagnostic snapshots; retained optimizer bytes measured separately",
        "started_ns": time.time_ns(),
    }
    reports = []
    try:
        write_receipt(args.durable_prefix, identity, f"rank{rank}-started", provenance)
        for decay in (0.0, 0.01):
            results, snapshots = {}, {}
            for name, base, remainders in ARMS:
                result, master, weight = run_arm(base, remainders, args.lr, decay)
                results[name] = result
                snapshots[name] = (master, weight)
                write_receipt(args.durable_prefix, identity, f"rank{rank}-decay{decay}-{name}", result)
            checks = compare_arm_states(results, snapshots)
            report = {"weight_decay": decay, "rank": rank, "checks": checks, "arms": results}
            reports.append(report)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {"provenance": provenance, "reports": reports}, group=control)
        keys = (
            "bf16_moments_memory_pass",
            "combined_remainders_memory_pass",
            "fp32_remainders_master_bits_equal_at_update10",
            "fp32_remainders_model_bits_equal_at_update10",
            "all_same_geometry_checkpoints_exact",
        )
        passed = all(report["checks"][key] for item in gathered for report in item["reports"] for key in keys)
        if rank == 0:
            write_receipt(args.durable_prefix, identity, "complete", {"passed": passed, "ranks": gathered})
            print(
                "MEGATRON_REMAINDERS_CAPABILITY_PASS ranks=2 updates=10"
                if passed
                else "MEGATRON_REMAINDERS_CAPABILITY_FAIL see=complete_receipt",
                flush=True,
            )
        if not passed:
            raise AssertionError("Observed remainder capability did not meet all declared gates")
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
