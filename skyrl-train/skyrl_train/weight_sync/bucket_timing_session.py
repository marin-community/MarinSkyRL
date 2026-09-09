"""Policy ownership across measured install and untimed full-byte replay.

This gate path keeps two buffers on every policy rank and requires successful
replay before reuse. It does not qualify a proof-free production path.
"""

from dataclasses import dataclass
from enum import StrEnum
import time

import torch

from skyrl_train.weight_sync.bucket_qualification import mark_measurement_once
from skyrl_train.weight_sync.frozen_source_views import local_source_slices
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.reference_bucket_protocol import manifest_wire_inventory
from skyrl_train.weight_sync.megatron_bucket_protocol import (
    PreparedBucketTransfer,
    close_preserving_failure,
    parameter_versions,
    persist_bucket_phase,
    prepare_bucket_transfer,
    receiver_rows,
    run_bucket_phase,
    validate_bucket_phase,
)
from skyrl_train.weight_sync.worker_bucket_protocol import MAX_REPLAY_EXTRA_BYTES


class Phase(StrEnum):
    READY = "ready"
    BEGINNING = "beginning"
    BEGUN = "begun"
    INSTALLING = "installing"
    INSTALLED = "installed"
    REPLAYING = "replaying"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class BucketTimingSession:
    prepared: PreparedBucketTransfer
    source_slices: tuple
    source_storage: dict
    mode: str = "bucket"
    phase: Phase = Phase.READY
    publication_id: int | None = None
    receiver_publication_id: int | None = None
    receiver_begin_pending: bool = False
    token: str | None = None
    versions: tuple = ()
    completed_update: int | None = None
    install: dict | None = None
    measurement_marker: dict | None = None


def source_storage(sources):
    # Resolved tasks may return fresh views of the same original storage.
    return {
        key: (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
        for key, value in sources.items()
    }


def resolved_sources(worker):
    return local_source_slices(worker.bridge.get_conversion_tasks(worker.actor_module), worker.provider)


def session_for(worker, publication_id, expected):
    state = worker._bucket_timing_session
    if not (state.phase is Phase.READY and publication_id is None) and (
        type(publication_id) is not int or publication_id != state.publication_id
    ):
        raise ValueError("Timing RPC does not match the active weight-sync version")
    if state.phase not in expected:
        raise RuntimeError(f"Timing RPC is invalid during phase {state.phase}")
    return state


async def prepare_timing(worker, client, *, source_owners, mode="bucket"):
    if mode not in ("bucket", "reference"):
        raise ValueError("Unknown timing transfer mode")
    if mode == "reference" and not getattr(worker.cfg.generator, "weight_sync_wire_inventory", False):
        raise ValueError("Reference timing requires the actual native wire inventory")
    if hasattr(worker, "_bucket_timing_session"):
        raise RuntimeError("Timing buffers are already prepared")
    with worker._policy_weight_access.hold("bucket-timing-preparation"):
        try:
            prepared = await prepare_bucket_transfer(worker, client, source_owners=source_owners)
            worker._bucket_timing_session = BucketTimingSession(
                prepared, prepared.local_slices, source_storage(prepared.local_sources), mode=mode
            )
            persist_readback(
                worker.cfg.trainer.weight_sync_readback_output, "bucket-timing-prepared", prepared.metadata
            )
        except BaseException as error:
            await close_preserving_failure(client, torch.cuda.current_device(), torch.distributed.get_rank(), error)
            if hasattr(worker, "_bucket_timing_session"):
                del worker._bucket_timing_session
            raise
    return prepared.metadata


async def abort_timing(worker, client, state, primary_error):
    state.phase = Phase.FAILED
    try:
        if torch.distributed.get_rank() == 0:
            versions = [state.receiver_publication_id]
            if state.receiver_begin_pending:
                versions.insert(0, state.publication_id)
            for index, version in enumerate(versions):
                try:
                    await client.close_diagnostic_weight_sync_buckets(
                        manifest_id=state.prepared.manifest.manifest_id if version is not None else None,
                        publication_id=version,
                    )
                    break
                except BaseException:
                    if index == len(versions) - 1:
                        raise
        torch.cuda.synchronize(torch.cuda.current_device())
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"Timing cleanup also failed: {type(cleanup_error).__name__}: {str(cleanup_error)[:4096]}"
        )
    finally:
        if state.token is not None:
            worker._policy_weight_access.release(state.token)
            state.token = None
        del worker._bucket_timing_session


async def begin_timing(worker, client, publication_id):
    state = worker._bucket_timing_session
    if type(publication_id) is not int or publication_id < 0:
        raise ValueError("Timing version must be a nonnegative integer")
    if state.phase not in (Phase.READY, Phase.VERIFIED):
        raise RuntimeError("The preceding timing sync has not completed its proof")
    if state.publication_id is not None and publication_id <= state.publication_id:
        raise ValueError("Timing versions must increase")
    state.token = worker._policy_weight_access.acquire("bucket-timing")
    state.phase = Phase.BEGINNING
    state.publication_id = publication_id
    try:
        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        slices, sources = resolved_sources(worker)
        if slices != state.source_slices or source_storage(sources) != state.source_storage:
            raise ValueError("Resolved source mapping/storage changed since timing preparation")
        # Read current task views, never cached export materializations.
        state.prepared.local_sources = sources
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        if peak - allocated > MAX_REPLAY_EXTRA_BYTES:
            raise ValueError("Replay source-view refresh exceeds the proof scratch limit")
        state.versions = parameter_versions(worker)
        state.completed_update = worker._completed_update
        state.install = None
        rows = None
        if torch.distributed.get_rank() == 0:
            if state.measurement_marker is None:
                state.measurement_marker = mark_measurement_once(worker.cfg.trainer.weight_sync_readback_output)
            cfg = worker.cfg.generator
            state.receiver_begin_pending = True
            begin = (
                client.begin_reference_bucket_sync if state.mode == "reference" else client.begin_diagnostic_weight_sync
            )
            rows, _ = receiver_rows(
                await begin(state.prepared.manifest.manifest_id, publication_id),
                engine_count=cfg.num_inference_engines,
                ranks_per_engine=(
                    cfg.inference_engine_tensor_parallel_size
                    * cfg.inference_engine_data_parallel_size
                    * cfg.inference_engine_pipeline_parallel_size
                ),
                data_parallel_size=cfg.inference_engine_data_parallel_size,
                expected=state.prepared.receiver_identities,
            )
            if any(
                row["publication_id"] != publication_id or row["manifest_id"] != state.prepared.manifest.manifest_id
                for row in rows
            ):
                raise ValueError("Receiver reset returned a different sync identity")
        torch.distributed.barrier()
        state.receiver_publication_id = publication_id
        state.receiver_begin_pending = False
        state.phase = Phase.BEGUN
        receipt = {
            "publication_id": publication_id,
            "measurement_marker": state.measurement_marker,
            "source_refresh_allocated_before": allocated,
            "source_refresh_peak_allocated_bytes": peak,
            "source_refresh_peak_extra_bytes": peak - allocated,
            "receivers": rows,
        }
        persist_readback(
            worker.cfg.trainer.weight_sync_readback_output,
            f"bucket-timing-begin-sync-{publication_id}",
            {"identity": state.prepared.metadata["identity"], **receipt},
        )
        return receipt
    except BaseException as error:
        await abort_timing(worker, client, state, error)
        raise


async def install_timing(worker, client, publication_id):
    state = session_for(worker, publication_id, (Phase.BEGUN,))
    state.phase = Phase.INSTALLING
    try:
        if state.mode == "reference":
            state.install = await run_reference_phase(worker, client, state)
        else:
            state.install = await run_bucket_phase(
                worker,
                client,
                state.prepared,
                replay=False,
                start_versions=state.versions,
                start_update=state.completed_update,
                publication_id=publication_id,
            )
            validate_bucket_phase(state.prepared, state.install, replay=False, rank=torch.distributed.get_rank())
        state.phase = Phase.INSTALLED
        return {"publication_id": publication_id, "install": state.install}
    except BaseException as error:
        await abort_timing(worker, client, state, error)
        raise


async def replay_timing(worker, client, publication_id):
    state = session_for(worker, publication_id, (Phase.INSTALLED,))
    state.phase = Phase.REPLAYING
    try:
        if state.mode == "reference":
            await complete_reference_install(worker, client, state)
        persist_bucket_phase(
            worker,
            state.prepared,
            state.install,
            replay=False,
            start_update=state.completed_update,
            publication_id=publication_id,
        )
        if state.mode == "reference" and torch.distributed.get_rank() == 0:
            if any(not row["reference_bind_memory_within_limit"] for row in state.install["receivers"]):
                raise ValueError("Reference receiver binding exceeded the proof scratch limit")
        receipt = await run_bucket_phase(
            worker,
            client,
            state.prepared,
            replay=True,
            start_versions=state.versions,
            start_update=state.completed_update,
            publication_id=publication_id,
        )
        persist_bucket_phase(
            worker,
            state.prepared,
            receipt,
            replay=True,
            start_update=state.completed_update,
            publication_id=publication_id,
        )
        validate_bucket_phase(state.prepared, receipt, replay=True, rank=torch.distributed.get_rank())
        state.phase = Phase.VERIFIED
        worker._policy_weight_access.release(state.token)
        state.token = None
        return {"publication_id": publication_id, "replay": receipt}
    except BaseException as error:
        await abort_timing(worker, client, state, error)
        raise


async def close_timing(worker, client, publication_id):
    state = session_for(worker, publication_id, (Phase.READY, Phase.BEGUN, Phase.INSTALLED, Phase.VERIFIED))
    await abort_timing(worker, client, state, None)
    return {"publication_id": publication_id, "closed": True}


async def run_reference_phase(worker, client, state):
    if parameter_versions(worker) != state.versions or worker._completed_update != state.completed_update:
        raise ValueError("Policy weights changed before original reference installation")
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    free = torch.cuda.mem_get_info(device)[0]
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    # Execute the original per-tensor export/reload path without substituting its load implementation.
    await worker.broadcast_to_inference_engines(client)
    inventory = worker.read_weight_sync_wire_inventory() if rank == 0 else None
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    elapsed = time.monotonic() - started
    if parameter_versions(worker) != state.versions or worker._completed_update != state.completed_update:
        raise ValueError("Policy weights changed during original reference installation")
    return {
        "installation_mode": "original_loader",
        "seconds": elapsed,
        "peak_extra_bytes": torch.cuda.max_memory_allocated(device) - allocated,
        "allocated_before": allocated,
        "allocated_after": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "free_device_bytes_before": free,
        "free_device_bytes_after": torch.cuda.mem_get_info(device)[0],
        "sender": inventory,
        "receivers": None,
        "buckets": None,
        "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
    }


async def complete_reference_install(worker, client, state):
    """Rebind proof views after the original timed native load has returned."""
    rank = torch.distributed.get_rank()
    started = time.monotonic()
    receivers = None
    inventory = state.install["sender"]
    if rank == 0:
        cfg = worker.cfg.generator
        receivers, _ = receiver_rows(
            await client.finish_reference_bucket_sync(state.prepared.manifest.manifest_id, state.publication_id),
            engine_count=cfg.num_inference_engines,
            ranks_per_engine=(
                cfg.inference_engine_tensor_parallel_size
                * cfg.inference_engine_data_parallel_size
                * cfg.inference_engine_pipeline_parallel_size
            ),
            data_parallel_size=cfg.inference_engine_data_parallel_size,
            expected=state.prepared.receiver_identities,
        )
        expected = manifest_wire_inventory(state.prepared.manifest)
        if inventory["entries"] != expected:
            raise ValueError("Native reference broadcasts differ from the frozen wire manifest")
        for index, row in enumerate(receivers):
            if (
                row["manifest_id"] != state.prepared.manifest.manifest_id
                or row["publication_id"] != state.publication_id
                or not row["original_reload_complete"]
                or row["wire_inventory"]["entries"] != inventory["entries"]
                or row["wire_inventory"]["wire_bytes"] != inventory["wire_bytes"]
                or row["installed_parameter_bytes"]
                != state.prepared.prepared_receivers[index]["installed_parameter_bytes"]
            ):
                raise ValueError("Reference completion lacks exact native sender/receiver coverage")
    state.install["receivers"] = receivers
    state.install["reference_binding_seconds"] = time.monotonic() - started
