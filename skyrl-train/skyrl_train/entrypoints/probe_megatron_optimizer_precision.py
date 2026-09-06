"""Bounded MCore/TE capability gate, separate from any Qwen learning experiment.

torchrun --standalone --nproc-per-node=2 -m \
    skyrl_train.entrypoints.probe_megatron_optimizer_precision

Uses one synthetic BF16 matrix, actual MCore distributed Adam/DDP, fixed FP32
gradient reduction and no optimizer graphs/offload/remainders. Four arms compare
native FP32 state, precision-aware FP32 state, BF16 first moment, and BF16 both
moments. An explicitly separate zero-gradient decay diagnostic uses lr=1e-3;
subsequent ordinary/tiny-gradient updates use --lr (default 1e-6).

The checkpoint check serializes native model/wrapper/inner-optimizer state plus
the native parameter shards, then verifies exact continuation at the same DP
geometry. It does not qualify distributed checkpoint resharding or model loaders.
No dataset/model downloads, training callbacks, or learning-equivalence claims.
"""

import argparse
from copy import deepcopy
from datetime import timedelta
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist

from skyrl_train.distributed.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config


MATRIX_WIDTH = 256
MAX_CHECKPOINT_BYTES = 16 * 1024**2
ARMS = {
    "native_fp32": (False, "float32", "float32"),
    "precision_fp32": (True, "float32", "float32"),
    "precision_bf16_first": (True, "bfloat16", "float32"),
    "precision_bf16_both": (True, "bfloat16", "bfloat16"),
}


class GradientMatrix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        values = torch.linspace(0.01, 0.1, MATRIX_WIDTH**2, device="cuda", dtype=torch.float32)
        self.weight = torch.nn.Parameter(values.reshape(MATRIX_WIDTH, MATRIX_WIDTH).to(torch.bfloat16))

    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        return (self.weight.float() * gradient).sum()


def tensor_inventory(named_tensors: list[tuple[str, torch.Tensor]]) -> dict:
    """Report logical tensor bytes separately from deduplicated retained storage."""
    storages, rows = {}, []
    for name, tensor in named_tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in storages:
            storages[key] = (len(storages), storage.nbytes())
        storage_id, storage_bytes = storages[key]
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "logical_bytes": tensor.numel() * tensor.element_size(),
                "storage_id": storage_id,
                "storage_bytes": storage_bytes,
                "storage_offset": tensor.storage_offset(),
            }
        )
    return {
        "tensors": rows,
        "logical_bytes": sum(row["logical_bytes"] for row in rows),
        "unique_retained_storage_bytes": sum(size for _, size in storages.values()),
    }


def validate_moment_inventory(inventory: dict, part_count: int, first: str, second: str) -> None:
    """Require both actual moment tensors for every local optimizer parameter."""
    rows = {row["name"]: row for row in inventory["tensors"]}
    if len(rows) != len(inventory["tensors"]) or part_count < 1:
        raise AssertionError("Duplicate tensor names or empty optimizer inventory")
    for part_index in range(part_count):
        parameters = [
            row
            for name, row in rows.items()
            if name.startswith(f"optimizer.{part_index}.") and name.endswith(".parameter")
        ]
        if not parameters:
            raise AssertionError(f"Missing optimizer parameters for part {part_index}")
        for parameter in parameters:
            prefix = parameter["name"].removesuffix(".parameter")
            for suffix, dtype in (("exp_avg", first), ("exp_avg_sq", second)):
                moment = rows.get(prefix + "." + suffix)
                if moment is None or moment["logical_bytes"] == 0:
                    raise AssertionError(f"Missing persistent moment: {prefix}.{suffix}")
                if moment["dtype"] != "torch." + dtype or moment["shape"] != parameter["shape"]:
                    raise AssertionError(f"Persistent moment dtype/shape differs: {prefix}.{suffix}")


def live_tensors(model, optimizer) -> list[tuple[str, torch.Tensor]]:
    """Read actual persistent tensors, never the FP32-converting TE state_dict."""
    tensors = []
    for name, parameter in model.module.named_parameters():
        tensors.extend((("model." + name, parameter), ("main_grad." + name, parameter.main_grad)))
    for part_index, part in enumerate(optimizer.chained_optimizers):
        inner = part.optimizer
        for group_index, group in enumerate(inner.param_groups):
            for param_index, parameter in enumerate(group["params"]):
                prefix = f"optimizer.{part_index}.{group_index}.{param_index}"
                tensors.append((prefix + ".parameter", parameter))
                tensors.extend(
                    (prefix + "." + name, value)
                    for name, value in sorted(inner.state[parameter].items())
                    if isinstance(value, torch.Tensor)
                )
                # TE 2.11 keeps scale tensors outside optimizer.state.
                tensors.extend(
                    (prefix + ".scale." + name, value)
                    for name, value in sorted(inner._scales.get(parameter, {}).items())
                )
        tensors.append((f"optimizer.{part_index}.overflow", inner._dummy_overflow_buf))
    return tensors


def build_arm(name: str, lr: float, weight_decay: float):
    # Native dependencies belong to the optional, pinned Megatron CUDA profile.
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformer_engine.pytorch.optimizers import FusedAdam

    aware, first, second = ARMS[name]
    declared = {
        "use_precision_aware_optimizer": aware,
        "optimizer_cuda_graph": False,
        "store_param_remainders": False,
        "optimizer_cpu_offload": False,
        "main_params_dtype": "float32",
        "main_grads_dtype": "float32",
        "exp_avg_dtype": first,
        "exp_avg_sq_dtype": second,
    }
    config = init_megatron_optim_config({"lr": lr, "weight_decay": weight_decay, "max_grad_norm": 0.0}, declared)
    module = GradientMatrix()
    for parameter in module.parameters():
        set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
    model = DistributedDataParallel(
        config=TransformerConfig(
            num_layers=1,
            hidden_size=MATRIX_WIDTH,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
        ),
        ddp_config=DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
        ),
        module=module,
    )
    optimizer = get_megatron_optimizer([model], config)
    for part in optimizer.chained_optimizers:
        if not isinstance(part.optimizer, FusedAdam):
            raise AssertionError("Capability gate requires the actual TransformerEngine FusedAdam")
        if any(group["weight_decay"] != weight_decay for group in part.optimizer.param_groups):
            raise AssertionError("Synthetic matrix unexpectedly excluded from weight decay")
    return model, optimizer, declared


def gradient_for_step(step: int, device: torch.device, rank: int | None = None) -> torch.Tensor:
    if step == 0:
        return torch.zeros((MATRIX_WIDTH, MATRIX_WIDTH), dtype=torch.float32, device=device)
    scales = torch.tensor([0, 1e-12, -1e-12, 1e-7, -1e-7, 1e-3, -1e-3, 1e-2], dtype=torch.float32, device=device)
    # Distinct local gradients force actual DP reduction; every arm gets the same inputs.
    gradient = scales.repeat(MATRIX_WIDTH**2 // len(scales)).reshape(MATRIX_WIDTH, MATRIX_WIDTH)
    return gradient * ((dist.get_rank() if rank is None else rank) + 1) * (-1 if step % 2 == 0 else 1)


def expected_reduced_gradient(step: int, device: torch.device, world_size: int) -> torch.Tensor:
    """Analytical derivative: BF16 leaf cast, then FP32 DP average, with no collective."""
    if world_size not in (1, 2):
        raise ValueError("The independent gradient oracle supports only 1-2 ranks")
    local_gradients = [gradient_for_step(step, device, rank).to(torch.bfloat16).float() for rank in range(world_size)]
    return torch.stack(local_gradients).mean(dim=0)


def main_parameter_tensors(optimizer) -> list[torch.Tensor]:
    tensors = []
    for part in optimizer.chained_optimizers:
        inner = part.optimizer
        for group in inner.param_groups:
            for parameter in group["params"]:
                tensors.append(inner.state[parameter]["master_param"] if inner.master_weights else parameter)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise AssertionError("This gate requires full FP32 master weights and excludes remainders")
    return tensors


def update(model, optimizer, step: int, lr: float, weight_decay: float) -> dict:
    optimizer.zero_grad()
    model.zero_grad_buffer()
    gradient = gradient_for_step(step, model.module.weight.device)
    model(gradient).backward()
    model.finish_grad_sync()
    for parameter in model.module.parameters():
        if parameter.main_grad.dtype != torch.float32:
            raise AssertionError("Actual main gradients must be FP32")
    # Reduce-scatter updates only the owned slice. Other slices can still contain
    # unreduced local gradients, so checking the full main_grad would be incorrect.
    if len(optimizer.chained_optimizers) != 1:
        raise AssertionError("The single-matrix probe requires one optimizer component")
    owned_range = optimizer.chained_optimizers[0]._get_model_param_range_map(model.module.weight)["param"]
    expected_gradient = expected_reduced_gradient(step, gradient.device, dist.get_world_size()).flatten()
    actual_gradient = model.module.weight.main_grad.flatten()[owned_range.start : owned_range.end]
    if actual_gradient.numel() != MATRIX_WIDTH**2 // dist.get_world_size():
        raise AssertionError("Unexpected owned gradient shard size for this matrix/DP geometry")
    torch.testing.assert_close(actual_gradient, expected_gradient[owned_range.start : owned_range.end], rtol=0, atol=0)
    expected_gradient_norm = float(expected_gradient.double().norm())
    del expected_gradient, actual_gradient
    # The first zero-gradient diagnostic uses a visible decay step separately.
    actual_lr = 1e-3 if step == 0 else lr
    for part in optimizer.chained_optimizers:
        for group in part.optimizer.param_groups:
            group["lr"] = actual_lr
    before = model.module.weight.detach().clone()
    # Masters are initialized by TE on the first step; the initial model is exact BF16.
    initial_shards = (
        [
            parameter.detach().float().clone()
            for part in optimizer.chained_optimizers
            for group in part.optimizer.param_groups
            for parameter in group["params"]
        ]
        if step == 0
        else [value.detach().clone() for value in main_parameter_tensors(optimizer)]
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    successful, grad_norm, _ = optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    if not successful or grad_norm is None or not torch.isfinite(torch.as_tensor(grad_norm)):
        raise AssertionError("Optimizer update was skipped or nonfinite")
    # ChainedOptimizer computes this norm even with clip_grad=0 in pinned MCore.
    torch.testing.assert_close(float(grad_norm), expected_gradient_norm, rtol=2e-6, atol=1e-15)
    if any(not torch.isfinite(value).all() for _, value in live_tensors(model, optimizer)):
        raise AssertionError("Nonfinite parameter, gradient or optimizer state")
    masters = main_parameter_tensors(optimizer)
    if step == 0:
        for previous, actual in zip(initial_shards, masters, strict=True):
            expected = previous * (1 - actual_lr * weight_decay)
            torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-9)
            if weight_decay and torch.equal(actual, previous):
                raise AssertionError("Visible zero-gradient weight decay did not update FP32 masters")
    deltas = torch.cat(
        [(actual - previous).flatten() for previous, actual in zip(initial_shards, masters, strict=True)]
    )
    return {
        "step": step,
        "phase": "zero_gradient_decay_diagnostic" if step == 0 else "fixed_small_lr_gradients",
        "lr": actual_lr,
        "gradient_norm": float(grad_norm),
        "gradient_norm_source": "MCore ChainedOptimizer.step; clipping disabled",
        "expected_reduced_gradient_l2": expected_gradient_norm,
        "reduced_gradient_shard_exact_before_update": True,
        "reduced_gradient_shard_range": [owned_range.start, owned_range.end],
        "optimizer_including_param_gather_seconds": elapsed,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "master_update_max_abs": float(deltas.abs().max()),
        "master_update_l2": float(deltas.norm()),
        "master_unchanged_elements": int((deltas == 0).sum()),
        "model_changed_bit_elements": int((before.view(torch.int16) != model.module.weight.view(torch.int16)).sum()),
    }


def checkpoint_bytes(model, optimizer) -> bytes:
    value = {
        "model": deepcopy(model.module.state_dict()),
        "wrapper": deepcopy(optimizer.state_dict()),
        "inner": [deepcopy(part.optimizer.state_dict()) for part in optimizer.chained_optimizers],
        "parameters": [
            [parameter.detach().clone() for group in part.optimizer.param_groups for parameter in group["params"]]
            for part in optimizer.chained_optimizers
        ],
    }
    buffer = io.BytesIO()
    torch.save(value, buffer)
    if buffer.tell() > MAX_CHECKPOINT_BYTES:
        raise ValueError("Capability checkpoint exceeds the 16 MiB bound")
    return buffer.getvalue()


def restore_checkpoint(model, optimizer, data: bytes) -> None:
    value = torch.load(io.BytesIO(data), weights_only=True, map_location=model.module.weight.device)
    model.module.load_state_dict(value["model"])
    optimizer.load_state_dict(value["wrapper"])
    with torch.no_grad():
        for part, state, parameters in zip(
            optimizer.chained_optimizers, value["inner"], value["parameters"], strict=True
        ):
            part.optimizer.load_state_dict(state)
            actual = [parameter for group in part.optimizer.param_groups for parameter in group["params"]]
            for parameter, saved in zip(actual, parameters, strict=True):
                parameter.copy_(saved)


def run_arm(name: str, lr: float, weight_decay: float) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model, optimizer, declared = build_arm(name, lr, weight_decay)
    rows = []
    for step in range(7):
        rows.append(update(model, optimizer, step, lr, weight_decay))
        if step == 3:
            saved = checkpoint_bytes(model, optimizer)
    inventory = tensor_inventory(live_tensors(model, optimizer))
    _, first, second = ARMS[name]
    validate_moment_inventory(inventory, len(optimizer.chained_optimizers), first, second)
    expected = {name: value.detach().cpu().clone() for name, value in live_tensors(model, optimizer)}
    final_weights = model.module.weight.detach().float().cpu()
    final_masters = torch.cat([value.detach().flatten().cpu() for value in main_parameter_tensors(optimizer)])
    ddp = {
        key: getattr(model.ddp_config, key)
        for key in (
            "use_distributed_optimizer",
            "grad_reduce_in_fp32",
            "average_in_collective",
            "overlap_grad_reduce",
            "overlap_param_gather",
        )
    }
    types = [
        f"{type(part.optimizer).__module__}.{type(part.optimizer).__qualname__}"
        for part in optimizer.chained_optimizers
    ]
    del model, optimizer
    gc.collect()
    restored_model, restored_optimizer, _ = build_arm(name, lr, weight_decay)
    restore_checkpoint(restored_model, restored_optimizer, saved)
    for step in range(4, 7):
        update(restored_model, restored_optimizer, step, lr, weight_decay)
    restored_tensors = dict(live_tensors(restored_model, restored_optimizer))
    if restored_tensors.keys() != expected.keys():
        raise AssertionError("Checkpoint continuation has missing or unexpected live tensors")
    for tensor_name, value in restored_tensors.items():
        reference = expected[tensor_name]
        if value.dtype != reference.dtype or not torch.equal(value.detach().cpu(), reference):
            raise AssertionError(f"Checkpoint continuation differs: {tensor_name}")
    return (
        {
            "arm": name,
            "weight_decay": weight_decay,
            "declared_optimizer_kwargs": declared,
            "actual_optimizer_types": types,
            "actual_ddp_config": ddp,
            "actual_state": inventory,
            "updates": rows,
            "checkpoint_bytes": len(saved),
            "same_geometry_checkpoint_continuation_exact": True,
        },
        final_weights,
        final_masters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.lr <= 1e-4 or int(os.environ.get("WORLD_SIZE", 0)) not in (1, 2):
        raise ValueError("Use 1-2 GPU ranks and learning rate in (0, 1e-4]")
    from megatron.core import parallel_state
    from megatron.core.optimizer import OptimizerConfig
    from transformer_engine.pytorch.optimizers import FusedAdam

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    try:
        parallel_state.initialize_model_parallel()
        rank = dist.get_rank()
        versions = {name: importlib.metadata.version(name) for name in ("megatron-core", "transformer-engine", "torch")}
        if versions["megatron-core"] != "0.18.0" or versions["transformer-engine"] != "2.11.0":
            raise ValueError("Gate requires the inspected MCore0.18.0/TE2.11.0 runtime")
        sources = {
            "entrypoint": Path(__file__),
            "adapter": Path(inspect.getfile(init_megatron_optim_config)),
            "mcore_config": Path(inspect.getfile(OptimizerConfig)),
            "te_fused_adam": Path(inspect.getfile(FusedAdam)),
        }
        provenance = {
            "versions": versions,
            "cuda": torch.version.cuda,
            "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
            "scope": "tiny actual distributed Adam/DDP capability; no Qwen learning or production checkpoint/resharding equivalence",
            "memory_scope": "step peak includes model/DDP and diagnostic before/master snapshots; persistent inventory is separate; serialization/restoration excluded from timing",
            "gradient_validation": "before each optimizer step, compare owned FP32 reduce-scatter slice against analytic rank-average of BF16 leaf gradients; oracle computation/synchronization excluded from optimizer timing",
            "precision_contract": "BF16 model; FP32 main gradients/reduction and masters; graphs/offload/remainders disabled; TE BF16 moments cast deterministically, no FSDP stochastic update flag",
            "world_size": dist.get_world_size(),
            "rank": rank,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "arguments": vars(args),
        }
        print("OPTIMIZER_PRECISION_PROVENANCE " + json.dumps(provenance), flush=True)
        for weight_decay in (0.0, 0.01):
            native_weights, native_masters, control_weights, control_masters = None, None, None, None
            for name in ARMS:
                result, weights, masters = run_arm(name, args.lr, weight_decay)
                if name == "native_fp32":
                    native_weights, native_masters = weights, masters
                if name == "precision_fp32":
                    control_weights, control_masters = weights, masters
                    result["model_max_abs_difference_from_native"] = float((weights - native_weights).abs().max())
                    result["master_max_abs_difference_from_native"] = float((masters - native_masters).abs().max())
                if name.startswith("precision_bf16"):
                    result["model_max_abs_difference_from_precision_fp32"] = float(
                        (weights - control_weights).abs().max()
                    )
                    result["master_max_abs_difference_from_precision_fp32"] = float(
                        (masters - control_masters).abs().max()
                    )
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, {"rank": rank, **result}, group=control)
                if rank == 0:
                    print(
                        "OPTIMIZER_PRECISION_RESULT "
                        + json.dumps({"arm": name, "weight_decay": weight_decay, "ranks": gathered}),
                        flush=True,
                    )
        if rank == 0:
            print("OPTIMIZER_PRECISION_CAPABILITY_PASS", flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
