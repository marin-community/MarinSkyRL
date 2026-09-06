"""One worker-local inventory after its first successful Megatron Adam update.

Read tensor metadata before zero_grad; never call state_dict (TE can upcast it),
copy tensors, synchronize CUDA, or reset allocator peaks. Category byte totals
can overlap through aliases. Only the all-category storage union is additive
within one worker. Neither quantity includes CUDA/NCCL allocations outside the
observed tensor storages or Python optimizer bookkeeping.

master_state is the raw TE master_param state (int16 residuals when remainders
are enabled), or the native optimizer's FP32 parameter shard. It is never a
reconstructed FP32 master copy. Model buffers and unreferenced padding are not
enumerated, though a tensor view counts its entire retained backing storage.

Join cuda_memory_observation PPO exit events by rank and target-update step for
phase peaks. This inventory adds host work to its first optimizer-step timing;
collection and observation durations are reported separately. Observation ends
before emitting the final summary, so neither duration is the full instrumentation
overhead or a GPU kernel duration.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import time

from loguru import logger
import torch

from skyrl_train.telemetry import WORKER_ROLE, record_event


@dataclass(frozen=True)
class OptimizerStatePart:
    parameters: Sequence[torch.Tensor]
    state: Mapping[torch.Tensor, Mapping[str, object]]
    scales: Mapping[torch.Tensor, Mapping[str, torch.Tensor]]
    overflow_buffer: torch.Tensor | None
    master_weights: bool
    decoupled_grad: bool
    settings: Mapping[str, str | bool]


class StorageInventory:
    """Aggregate actual tensors without retaining them or publishing parameter IDs."""

    def __init__(self):
        self.rows = {}
        self.storages = {}
        self.category_storages = defaultdict(set)
        self.seen_tensors = set()

    def add(self, category: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or (category, id(tensor)) in self.seen_tensors:
            return
        self.seen_tensors.add((category, id(tensor)))
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr())
        self.storages[storage_key] = storage.nbytes()
        key = (category, str(tensor.dtype), str(tensor.device))
        row = self.rows.setdefault(key, {"tensor_count": 0, "elements": 0, "logical_bytes": 0})
        row["tensor_count"] += 1
        row["elements"] += tensor.numel()
        row["logical_bytes"] += tensor.numel() * tensor.element_size()
        self.category_storages[key].add(storage_key)

    def summary(self) -> dict:
        rows = [
            {
                "category": key[0],
                "dtype": key[1],
                "device": key[2],
                **row,
                "unique_retained_storage_bytes": sum(self.storages[item] for item in self.category_storages[key]),
            }
            for key, row in sorted(self.rows.items())
        ]
        if len(rows) > 64:
            raise ValueError("Optimizer inventory exceeds 64 category/dtype/device rows")
        return {
            "rows": rows,
            "logical_bytes_including_role_aliases": sum(row["logical_bytes"] for row in rows),
            "unique_retained_storage_bytes": sum(self.storages.values()),
            "unique_cuda_storage_bytes": sum(
                size for (device, _), size in self.storages.items() if device.startswith("cuda:")
            ),
            "unique_cpu_storage_bytes": sum(size for (device, _), size in self.storages.items() if device == "cpu"),
            "unique_storage_count": len(self.storages),
        }


def collect_optimizer_inventory(model_parameters: Sequence[torch.Tensor], parts: Sequence[OptimizerStatePart]) -> dict:
    """Inspect raw persistent states; coverage failures remain explicit in output."""
    inventory = StorageInventory()
    coverage = defaultdict(int)
    for parameter in model_parameters:
        inventory.add("model_parameter", parameter)
        main_grad = getattr(parameter, "main_grad", None)
        inventory.add("model_main_grad", main_grad)
        coverage["model_parameters"] += 1
        coverage["model_trainable_parameters"] += int(parameter.requires_grad)
        coverage["model_main_grad_present"] += int(main_grad is not None)
    for part in parts:
        coverage["optimizer_components"] += 1
        coverage["empty_optimizer_components"] += int(not part.parameters)
        for parameter in part.parameters:
            coverage["optimizer_parameters"] += 1
            coverage["empty_optimizer_parameters"] += int(parameter.numel() == 0)
            state = part.state.get(parameter, {})
            inventory.add("optimizer_parameter", parameter)
            gradient = getattr(parameter, "decoupled_grad", None) if part.decoupled_grad else parameter.grad
            inventory.add("optimizer_grad", gradient)
            coverage["optimizer_grad_present"] += int(gradient is not None)
            master = state.get("master_param") if part.master_weights else parameter
            inventory.add("master_state", master)
            coverage["master_state_present"] += int(master is not None)
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state.get(name)
                inventory.add(name, moment)
                coverage[name + "_present"] += int(moment is not None)
                coverage[name + "_dtype_mismatch"] += int(
                    moment is not None and str(moment.dtype) != part.settings[name + "_dtype"]
                )
                coverage[name + "_shape_mismatch"] += int(moment is not None and moment.shape != parameter.shape)
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and name not in ("master_param", "exp_avg", "exp_avg_sq"):
                    inventory.add("other_optimizer_state", value)
            for scale in part.scales.get(parameter, {}).values():
                inventory.add("optimizer_scale", scale)
        inventory.add("optimizer_auxiliary", part.overflow_buffer)
    complete = coverage["optimizer_parameters"] > 0 and all(
        coverage[name + "_present"] == coverage["optimizer_parameters"]
        for name in ("optimizer_grad", "master_state", "exp_avg", "exp_avg_sq")
    )
    complete &= all(
        coverage[name + suffix] == 0
        for name in ("exp_avg", "exp_avg_sq")
        for suffix in ("_dtype_mismatch", "_shape_mismatch")
    )
    complete &= coverage["empty_optimizer_components"] == coverage["empty_optimizer_parameters"] == 0
    return {**inventory.summary(), "coverage": dict(coverage), "complete": complete}


def megatron_inventory(model_chunks, optimizer) -> tuple[dict, list[dict]]:
    """Adapter for the pinned ordinary-DDP MCore ChainedOptimizer/TE path."""
    parameters = list(dict.fromkeys(parameter for chunk in model_chunks for parameter in chunk.module.parameters()))
    parts, settings = [], []
    for index, component in enumerate(optimizer.chained_optimizers):
        inner = component.optimizer
        declared = {
            name: str(getattr(component.config, name))
            for name in ("params_dtype", "main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype")
        }
        actual = {
            "component": index,
            "optimizer_class": f"{type(inner).__module__}.{type(inner).__qualname__}",
            "use_precision_aware_optimizer": bool(component.config.use_precision_aware_optimizer),
            "master_weights": bool(inner.master_weights),
            "use_decoupled_grad": bool(inner.use_decoupled_grad),
            "store_param_remainders": bool(inner.store_param_remainders),
            "optimizer_cuda_graph": bool(component.config.optimizer_cuda_graph),
            "grad_reduce_in_fp32": bool(component.ddp_config.grad_reduce_in_fp32),
            "average_in_collective": bool(component.ddp_config.average_in_collective),
            **declared,
        }
        settings.append(actual)
        parts.append(
            OptimizerStatePart(
                parameters=[parameter for group in inner.param_groups for parameter in group["params"]],
                state=inner.state,
                scales=inner._scales,
                overflow_buffer=inner._dummy_overflow_buf,
                master_weights=inner.master_weights,
                decoupled_grad=inner.use_decoupled_grad,
                settings=declared,
            )
        )
    return collect_optimizer_inventory(parameters, parts), settings


class OptimizerStateObserver:
    """Collect once after a successful native update, before clearing gradients.

    Diagnostic failure logs an error and disables this observer. It never
    substitutes success for missing evidence or changes the optimizer outcome.
    """

    def __init__(self, *, enabled: bool, rank: int):
        self.enabled = enabled
        self.rank = rank
        self.skipped_update_attempts = 0

    def after_step(self, successful: bool, *, model_chunks, optimizer, step: int, minibatch: int) -> None:
        if not self.enabled:
            return
        if not successful:
            self.skipped_update_attempts += 1
            return
        self.enabled = False
        started = time.perf_counter()
        attributes = {
            "backend": "megatron",
            "role": WORKER_ROLE,
            "worker_role": "policy",
            "rank": str(self.rank),
            "step": str(step),
            "step_kind": "target_update",
            "boundary": "after_optimizer_step_before_zero_grad",
            "inventory_version": "1",
        }
        try:
            inventory, settings = megatron_inventory(model_chunks, optimizer)
            device = torch.cuda.current_device()
            attributes["cuda_device"] = str(device)
            attributes["gpu_uuid"] = str(torch.cuda.get_device_properties(device).uuid)
            memory = torch.cuda.memory_stats(device)
            fields = {
                **inventory["coverage"],
                "complete": inventory["complete"],
                "logical_bytes_including_role_aliases": inventory["logical_bytes_including_role_aliases"],
                "unique_retained_storage_bytes": inventory["unique_retained_storage_bytes"],
                "unique_cuda_storage_bytes": inventory["unique_cuda_storage_bytes"],
                "unique_cpu_storage_bytes": inventory["unique_cpu_storage_bytes"],
                "unique_storage_count": inventory["unique_storage_count"],
                "storage_row_count": len(inventory["rows"]),
                "allocated_bytes": memory["allocated_bytes.all.current"],
                "reserved_bytes": memory["reserved_bytes.all.current"],
                "optimizer_minibatch_in_target_update": minibatch,
                "skipped_update_attempts_before_inventory": self.skipped_update_attempts,
                "host_collection_seconds": time.perf_counter() - started,
            }
            for row in inventory["rows"]:
                record_event("optimizer_state_storage", row, attributes=attributes)
                logger.info("OPTIMIZER_STATE_STORAGE {}", json.dumps({**attributes, **row}, sort_keys=True))
            for component in settings:
                record_event("optimizer_state_settings", component, attributes=attributes)
                logger.info("OPTIMIZER_STATE_SETTINGS {}", json.dumps({**attributes, **component}, sort_keys=True))
            # Emit the summary last: a partial export has no complete manifest.
            fields["host_observer_seconds_before_summary"] = time.perf_counter() - started
            record_event("optimizer_state_inventory", fields, attributes=attributes)
            logger.info("OPTIMIZER_STATE_INVENTORY {}", json.dumps({**attributes, **fields}, sort_keys=True))
        except Exception as error:
            logger.warning("Optimizer state inventory failed: {}", error)
            logger.error(
                "OPTIMIZER_STATE_INVENTORY_ERROR {}",
                json.dumps({**attributes, "error": str(error)[:512]}, sort_keys=True),
            )
