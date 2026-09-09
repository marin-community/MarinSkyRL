"""Explicit diagnostic RPC protocol for native Grug bucket install and replay.

Ordinary weight sync never calls this protocol. The driver must keep inference
paused and the sender frozen through install and the subsequent untimed replay.
"""

import time

import torch

from skyrl_train.weight_sync.bucket_receiver import GrugBucketReceiver
from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.manifest import parse_manifest


BUCKET_BYTES = 2**30
# Pinned Torch count_nonzero promotes this entire bool slice to int64.
# 64 KiB keeps that temporary plus the router conversion below the 1 MiB gate.
# Native phase peaks remain authoritative, including allocator/kernel overhead.
REPLAY_SCRATCH_BYTES = 64 * 1024
MAX_REPLAY_EXTRA_BYTES = 2**20


def parameter_storage(parameters):
    return {
        name: (
            id(tensor),
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
        for name, tensor in parameters.items()
    }


def prepare_worker_buckets(worker, payload, manifest_id):
    if getattr(worker, "_model_update_group", None) is None:
        raise ValueError("The native weight-update communicator must already exist")
    if hasattr(worker, "_diagnostic_bucket_state"):
        raise ValueError("Receiver bucket state must be closed before preparing again")
    if getattr(worker, "_skyrl_weight_update_active", False) or hasattr(worker, "_accumulated_weights"):
        raise ValueError("Bucket protocol cannot overlap the layerwise reload protocol")
    config = worker.vllm_config
    hf = config.model_config.hf_config
    parallel = config.parallel_config
    if (
        hf.model_type != "grug_moe"
        or config.model_config.quantization is not None
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_eplb
    ):
        raise ValueError("Receiver bucket protocol requires unquantized TP1 PP1 Grug without expert rebalancing")
    manifest = parse_manifest(payload, manifest_id)
    if manifest.bucket_bytes != BUCKET_BYTES:
        raise ValueError("Native receiver requires the approved 1 GiB bucket capacity")
    model = worker.model_runner.model
    maps = {}
    for name, module in model.named_modules():
        if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
            continue
        backend = getattr(getattr(module.quant_method, "unquantized_backend", None), "name", None)
        if backend != "TRITON":
            raise ValueError("Every instantiated expert module must report the native TRITON backend")
        mapper = module._map_global_expert_id_to_local_expert_id
        maps[name] = tuple(int(mapper(expert)) for expert in range(hf.num_experts))
    if len(maps) != hf.num_hidden_layers:
        raise ValueError("Native expert module coverage differs from the configured layer count")
    parameters = dict(model.named_parameters())
    torch.cuda.synchronize(worker.device)
    free, total = torch.cuda.mem_get_info(worker.device)
    if free < 2 * BUCKET_BYTES + MAX_REPLAY_EXTRA_BYTES:
        raise ValueError("Receiver lacks free space for both buffers and bounded replay scratch")
    allocated_before = torch.cuda.memory_allocated(worker.device)
    buffers = tuple(torch.empty(BUCKET_BYTES, dtype=torch.uint8, device=worker.device) for _ in range(2))
    receiver = GrugBucketReceiver(
        manifest, parameters, maps, buffers, backend="TRITON", tensor_parallel_size=parallel.tensor_parallel_size
    )
    worker._diagnostic_bucket_state = {
        "receiver": receiver,
        "parameter_storage": parameter_storage(parameters),
        "scratch": None,
        "load_stream": torch.cuda.Stream(device=worker.device),
        "receive_events": [torch.cuda.Event(), torch.cuda.Event()],
        "load_events": [torch.cuda.Event(), torch.cuda.Event()],
        "slot_used": [False, False],
        "install_complete": False,
        "publication_id": None,
        "replay_verified": False,
        "install_allocated_before": torch.cuda.memory_allocated(worker.device),
        "install_free_before": torch.cuda.mem_get_info(worker.device)[0],
    }
    torch.cuda.reset_peak_memory_stats(worker.device)
    return {
        "identity": bucket_identity(worker.device),
        "manifest_id": manifest_id,
        "bucket_count": manifest.bucket_count,
        "bucket_bytes": BUCKET_BYTES,
        "installed_parameter_bytes": receiver.expected_bytes,
        "expert_modules": len(maps),
        "free_bytes_before": free,
        "total_bytes": total,
        "buffer_allocated_delta": torch.cuda.memory_allocated(worker.device) - allocated_before,
        "rank": torch.distributed.get_rank(),
    }


def begin_worker_bucket_sync(worker, manifest_id: str, publication_id: int):
    """Reuse prepared buffers while binding all subsequent RPCs to one sync."""
    state = worker._diagnostic_bucket_state
    receiver = state["receiver"]
    if type(publication_id) is not int or publication_id < 0 or manifest_id != receiver.manifest.manifest_id:
        raise ValueError("Invalid cached-manifest weight-sync identity")
    prior = state["publication_id"]
    if prior is not None and publication_id <= prior:
        raise ValueError("Weight-sync versions must increase; delayed reset RPC rejected")
    if getattr(worker, "_skyrl_weight_update_active", False) or hasattr(worker, "_accumulated_weights"):
        raise ValueError("Bucket reset cannot overlap layerwise reload")
    current = dict(worker.model_runner.model.named_parameters())
    if parameter_storage(current) != state["parameter_storage"]:
        raise ValueError("Installed parameter identity changed since manifest preparation")
    if prior is not None or any(state["slot_used"]):
        if not state["replay_verified"]:
            raise ValueError("The preceding sync needs a complete successful replay")
        receiver.reset_after_verified_replay()
    torch.cuda.synchronize(worker.device)
    state["scratch"] = None
    state.update(
        publication_id=publication_id,
        replay_verified=False,
        slot_used=[False, False],
        install_complete=False,
        install_allocated_before=torch.cuda.memory_allocated(worker.device),
        install_free_before=torch.cuda.mem_get_info(worker.device)[0],
    )
    torch.cuda.reset_peak_memory_stats(worker.device)
    return {
        "identity": bucket_identity(worker.device),
        "manifest_id": manifest_id,
        "publication_id": publication_id,
        "buffers_reused": True,
        "buffer_pointers": [buffer.data_ptr() for buffer in receiver.buffers],
    }


def receive_worker_bucket(
    worker, bucket_id: int, *, replay: bool = False, manifest_id: str | None = None, publication_id: int | None = None
):
    state = worker._diagnostic_bucket_state
    receiver = state["receiver"]
    if (manifest_id, publication_id) != (
        receiver.manifest.manifest_id if state["publication_id"] is not None else None,
        state["publication_id"],
    ):
        raise ValueError("Bucket RPC does not match the active manifest and weight-sync version")
    receiver.validate_next_bucket(bucket_id, replay=replay)
    if replay and not state["install_complete"]:
        raise ValueError("Replay requires the explicit installed-weight completion join")
    entries = receiver.manifest.bucket(bucket_id)
    nbytes = entries[-1].offset + entries[-1].nbytes
    slot = bucket_id % 2
    buffer = receiver.buffers[slot]
    if replay and state["scratch"] is None:
        # Peak is measured across allocation, every comparison and reduction.
        # This resets process allocator statistics only in the explicit diagnostic.
        torch.cuda.synchronize(worker.device)
        state["replay_started"] = time.monotonic()
        state["replay_allocated_before"] = torch.cuda.memory_allocated(worker.device)
        state["replay_reserved_before"] = torch.cuda.memory_reserved(worker.device)
        state["replay_free_before"] = torch.cuda.mem_get_info(worker.device)[0]
        torch.cuda.reset_peak_memory_stats(worker.device)
        state["scratch"] = torch.empty(REPLAY_SCRATCH_BYTES, dtype=torch.bool, device=worker.device)
    receive_stream = torch.cuda.current_stream(worker.device)
    if state["slot_used"][slot]:
        receive_stream.wait_event(state["load_events"][slot])
    torch.distributed.broadcast(buffer.narrow(0, 0, nbytes), src=0, group=worker._model_update_group)
    state["receive_events"][slot].record(receive_stream)
    with torch.cuda.stream(state["load_stream"]):
        state["load_stream"].wait_event(state["receive_events"][slot])
        if replay:
            result = receiver.replay_bucket(bucket_id, state["scratch"])
            receipt = {"compared_bytes": result.compared_bytes, "mismatches": result.mismatches}
        else:
            receipt = {"installed_bytes": receiver.install_bucket(bucket_id)}
        state["load_events"][slot].record(state["load_stream"])
    state["slot_used"][slot] = True
    return {
        "identity": bucket_identity(worker.device),
        "bucket_id": bucket_id,
        "manifest_id": receiver.manifest.manifest_id,
        "publication_id": state["publication_id"],
        "slot": slot,
        "load_completion_event_recorded": True,
        **receipt,
    }


def finish_worker_install(worker):
    state = worker._diagnostic_bucket_state
    state["receiver"].validate_install_complete()
    completed_slots = []
    for slot, (used, event) in enumerate(zip(state["slot_used"], state["load_events"], strict=True)):
        if used:
            event.synchronize()
            if not event.query():
                raise RuntimeError("A receiver load event remains incomplete after its completion join")
            completed_slots.append(slot)
    state["install_complete"] = True
    return {
        "identity": bucket_identity(worker.device),
        "install_complete": True,
        "publication_id": state["publication_id"],
        "allocated_before": state["install_allocated_before"],
        "allocated_after": torch.cuda.memory_allocated(worker.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
        "free_device_bytes_before": state["install_free_before"],
        "free_device_bytes_after": torch.cuda.mem_get_info(worker.device)[0],
        "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
        "completed_slots": completed_slots,
        "manifest_id": state["receiver"].manifest.manifest_id,
    }


def finish_worker_replay(worker):
    state = worker._diagnostic_bucket_state
    if state["scratch"] is None:
        raise ValueError("No frozen replay was started")
    result = state["receiver"].finish_replay()
    torch.cuda.synchronize(worker.device)
    peak_extra = torch.cuda.max_memory_allocated(worker.device) - state["replay_allocated_before"]
    state["replay_verified"] = (
        result.mismatches == 0
        and result.compared_bytes == state["receiver"].expected_bytes
        and peak_extra <= MAX_REPLAY_EXTRA_BYTES
    )
    # Return the measured failure evidence before the aggregate gate rejects it.
    # Rank zero persists this receipt before applying the unchanged 1 MiB bound.
    return {
        "identity": bucket_identity(worker.device),
        "manifest_id": state["receiver"].manifest.manifest_id,
        "compared_bytes": result.compared_bytes,
        "expected_bytes": state["receiver"].expected_bytes,
        "mismatches": result.mismatches,
        "coverage": 1.0,
        "replay_seconds": time.monotonic() - state["replay_started"],
        "publication_id": state["publication_id"],
        "replay_peak_extra_bytes": peak_extra,
        "replay_scratch_limit_bytes": MAX_REPLAY_EXTRA_BYTES,
        "replay_memory_within_limit": peak_extra <= MAX_REPLAY_EXTRA_BYTES,
        "reserved_before": state["replay_reserved_before"],
        "reserved_after": torch.cuda.memory_reserved(worker.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(worker.device),
        "allocated_before": state["replay_allocated_before"],
        "allocated_after": torch.cuda.memory_allocated(worker.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
        "free_device_bytes_before": state["replay_free_before"],
        "free_device_bytes_after": torch.cuda.mem_get_info(worker.device)[0],
        "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
        "rank": torch.distributed.get_rank(),
    }


def close_worker_buckets(worker):
    present = hasattr(worker, "_diagnostic_bucket_state")
    if present:
        torch.cuda.synchronize(worker.device)
        del worker._diagnostic_bucket_state
    return {
        "identity": bucket_identity(worker.device),
        "rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(),
        "closed": True,
        "state_was_present": present,
    }
