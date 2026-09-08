"""Read native receiver prerequisites without allocating weight-sync buffers."""

import os
import socket
import time

import torch


def read_publication_receiver_state(worker):
    """Report post-initialization device memory and each instantiated MoE layout.

    The caller must invoke this after engine initialization/warmup. This method
    reports observed state; it does not select a backend or change model storage.
    """
    device = worker.device
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    hf = worker.vllm_config.model_config.hf_config
    parallel = worker.vllm_config.parallel_config
    experts = int(getattr(hf, "num_local_experts", getattr(hf, "num_experts", 0)))
    layers = []
    for name, module in worker.model_runner.model.named_modules():
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        backend = getattr(getattr(module, "quant_method", None), "unquantized_backend", None)
        parameters = {}
        for key in ("w13_weight", "w2_weight"):
            tensor = getattr(module, key)
            parameters[key] = {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "contiguous": tensor.is_contiguous(),
                "data_ptr": tensor.data_ptr(),
            }
        mapper = getattr(module, "_map_global_expert_id_to_local_expert_id", None)
        mapping = [int(mapper(expert)) for expert in range(experts)] if mapper is not None else None
        layers.append(
            {
                "name": name,
                "backend": getattr(backend, "name", None),
                "backend_repr": str(backend),
                "parameters": parameters,
                "expert_map": mapping,
            }
        )
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "observed_monotonic": time.monotonic(),
        "device": str(device),
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "weight_reload_active": bool(getattr(worker, "_skyrl_weight_update_active", False)),
        "rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(),
        "model": worker.vllm_config.model_config.model,
        "parallel": {
            key: getattr(parallel, key, None)
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "data_parallel_rank",
                "enable_expert_parallel",
            )
        },
        "hf_dimensions": {
            key: getattr(hf, key, None)
            for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_local_experts")
        },
        "model_type": getattr(hf, "model_type", None),
        "num_experts": experts,
        "layers": layers,
    }
