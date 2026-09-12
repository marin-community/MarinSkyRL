"""Prove original-loader coverage before binding new storage for byte replay."""

import math

import torch

from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.wire_inventory import WireInventory
from skyrl_train.weight_sync.worker_bucket_protocol import (
    MAX_REPLAY_EXTRA_BYTES,
    begin_worker_bucket_sync,
    parameter_storage,
    validate_sync_identity,
)


def manifest_wire_inventory(manifest):
    entries = {}
    for part in manifest.entries:
        if part.hf_name in entries:
            continue
        numel = math.prod(part.full_shape)
        entries[part.hf_name] = {
            "name": part.hf_name,
            "shape": list(part.full_shape),
            "dtype": part.wire_dtype,
            "numel": numel,
            "bytes": numel * getattr(torch, part.wire_dtype).itemsize,
        }
    return list(entries.values())


def begin_reference_sync(worker, manifest_id, publication_id):
    receipt = begin_worker_bucket_sync(worker, manifest_id, publication_id)
    state = worker._diagnostic_bucket_state
    state["reference_layout"] = state["receiver"].release_for_reference_reload()
    state["reference_inventory"] = WireInventory()
    state["reference_active"] = True
    return {**receipt, "installation_mode": "original_loader", "old_parameter_references_released": True}


def observe_reference_load(worker, weights):
    state = getattr(worker, "_diagnostic_bucket_state", None)
    if state is None or not state.get("reference_active", False):
        return
    if not getattr(worker, "_skyrl_weight_update_active", False) or hasattr(worker, "_accumulated_weights"):
        raise ValueError("Reference inventory requires the original unfused layerwise reload")
    for name, tensor in weights:
        state["reference_inventory"].observe(name, tensor)


def finish_reference_sync(worker, manifest_id, publication_id):
    state = worker._diagnostic_bucket_state
    validate_sync_identity(state, manifest_id, publication_id)
    if not state.get("reference_active", False):
        raise ValueError("No original reference reload is active")
    if getattr(worker, "_skyrl_weight_update_active", False) or hasattr(worker, "_accumulated_weights"):
        raise ValueError("The original reload must finish before reference completion")
    receiver = state["receiver"]
    inventory = state["reference_inventory"].finish(completed_update=publication_id)
    if inventory["entries"] != manifest_wire_inventory(receiver.manifest):
        raise ValueError("Original loader inventory does not exactly cover the wire manifest")
    if inventory["wire_bytes"] != sum(part.nbytes for part in receiver.manifest.entries):
        raise ValueError("Original loader wire-byte total differs from the complete manifest")
    torch.cuda.synchronize(worker.device)
    install_peak = torch.cuda.max_memory_allocated(worker.device)
    allocated = torch.cuda.memory_allocated(worker.device)
    torch.cuda.reset_peak_memory_stats(worker.device)
    model = worker.model_runner.model
    maps = {}
    for name, module in model.named_modules():
        if name not in receiver.expert_maps:
            continue
        if getattr(getattr(module.quant_method, "unquantized_backend", None), "name", None) != "TRITON":
            raise ValueError("Original reload changed the expert backend")
        maps[name] = tuple(
            int(module._map_global_expert_id_to_local_expert_id(expert))
            for expert in range(worker.vllm_config.model_config.hf_config.num_experts)
        )
    if maps != receiver.expert_maps:
        raise ValueError("Original reload changed the expert placement")
    parameters = dict(model.named_parameters())
    state["receiver"] = receiver.bind_reference_install(parameters, expected_layout=state["reference_layout"])
    state["parameter_storage"] = parameter_storage(parameters)
    torch.cuda.synchronize(worker.device)
    peak = torch.cuda.max_memory_allocated(worker.device)
    state["reference_active"] = False
    state["install_complete"] = True
    return {
        "identity": bucket_identity(worker.device),
        "manifest_id": manifest_id,
        "publication_id": publication_id,
        "installation_mode": "original_loader",
        "install_complete": True,
        "original_reload_complete": True,
        "wire_inventory": inventory,
        "installed_parameter_bytes": state["receiver"].expected_bytes,
        "allocated_before": state["install_allocated_before"],
        "allocated_after": allocated,
        "peak_allocated_bytes": install_peak,
        "free_device_bytes_before": state["install_free_before"],
        "free_device_bytes_after": torch.cuda.mem_get_info(worker.device)[0],
        "reference_bind_allocated_before": allocated,
        "reference_bind_peak_allocated_bytes": peak,
        "reference_bind_peak_extra_bytes": peak - allocated,
        "reference_bind_memory_within_limit": peak - allocated <= MAX_REPLAY_EXTRA_BYTES,
        "reference_bind_scratch_limit_bytes": MAX_REPLAY_EXTRA_BYTES,
    }
