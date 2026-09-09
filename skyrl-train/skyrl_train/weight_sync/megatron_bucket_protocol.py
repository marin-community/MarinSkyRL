"""One frozen Megatron interval: timed bucket installation, then untimed replay."""

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import chain
import time

import torch

from skyrl_train.weight_sync.bucket_identity import bucket_identity
from skyrl_train.weight_sync.cpu_source_catalogue import gather_source_catalogue
from skyrl_train.weight_sync.bucket_sender import StreamingBucketSender
from skyrl_train.weight_sync.frozen_source_views import local_source_slices
from skyrl_train.weight_sync.frozen_source_plan import frozen_source_plan
from skyrl_train.weight_sync.frozen_view_sender import FrozenViewBucketSender
from skyrl_train.weight_sync.manifest import PublicationManifest, TensorSpec, build_manifest
from skyrl_train.weight_sync.receiver_readback_rpc import group_external_dp_workers
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.worker_bucket_protocol import BUCKET_BYTES, MAX_REPLAY_EXTRA_BYTES


def complete_exports(worker):
    for chunk in worker.weight_extractor.extract_weights(torch.bfloat16):
        if len(chunk) != 1:
            raise ValueError("Diagnostic bucket packing requires complete unbucketed HF exports")
        yield chunk.names[0], chunk.tensors[0]


def parameter_versions(worker):
    # Cheap mutation tripwire only. The common exclusive lock prevents PPO from
    # entering; full byte replay, not this metadata, proves installed equality.
    return tuple(
        (index, name, id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape), str(tensor.dtype))
        for index, module in enumerate(worker.actor_module)
        for name, tensor in chain(module.named_parameters(), module.named_buffers())
    )


async def close_preserving_failure(client, device, rank, primary_error):
    """Retain the initiating native failure if cleanup also fails."""
    try:
        if rank == 0:
            await client.close_diagnostic_weight_sync_buckets()
        torch.cuda.synchronize(device)
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"Bucket cleanup also failed: {type(cleanup_error).__name__}: {str(cleanup_error)[:4096]}"
        )


def receiver_rows(nested, *, engine_count: int, ranks_per_engine: int, data_parallel_size: int = 1, expected=None):
    native = [[{"rank": row["identity"]["rank"], "receipt": row} for row in actor] for actor in nested]
    grouped = group_external_dp_workers(
        native,
        {
            "receiver_engines": engine_count,
            "receiver_ranks_per_engine": ranks_per_engine,
            "receiver_parallel": {"data_parallel_size": data_parallel_size},
        },
    )
    nested = [[row["receipt"] for row in engine] for engine in grouped]
    if len(nested) != engine_count or any(len(rows) != ranks_per_engine for rows in nested):
        raise ValueError("Bucket receipt is missing configured receiver workers")
    identities = []
    rows = []
    for engine, values in enumerate(nested):
        by_rank = {row["identity"]["rank"]: row for row in values}
        if set(by_rank) != set(range(ranks_per_engine)):
            raise ValueError("Bucket receipt contains missing or duplicate receiver ranks")
        for rank in range(ranks_per_engine):
            row = by_rank[rank]
            identity = row["identity"]
            if identity["world_size"] != ranks_per_engine:
                raise ValueError("Receiver world size differs from requested geometry")
            identities.append((engine, identity))
            rows.append(row)
    physical = {(identity["host"], identity["gpu_uuid"]) for _, identity in identities}
    if len(physical) != engine_count * ranks_per_engine:
        raise ValueError("Receiver identities share a physical device")
    if expected is not None and identities != expected:
        raise ValueError("Receiver identity changed during the frozen interval")
    return rows, identities


@dataclass
class PreparedBucketTransfer:
    manifest: PublicationManifest
    source_plan: tuple
    local_slices: tuple
    local_sources: dict
    buffers: tuple
    prepared_receivers: object
    receiver_identities: object
    metadata: dict


async def prepare_bucket_transfer(worker, client, *, source_owners):
    if worker.use_cuda_ipc or worker.weight_extractor.enable_bucketing:
        raise ValueError("Bucket diagnostic requires unbucketed non-colocated NCCL export")
    if worker.weight_extractor.model_type != "grug_moe":
        raise ValueError("This diagnostic requires the qualified Grug export geometry")
    generator = worker.cfg.generator
    if generator.model_dtype != "bfloat16":
        raise ValueError("Bucket diagnostic preserves the BF16 export and FP32 router biases")
    rank = torch.distributed.get_rank()
    device = torch.cuda.current_device()
    identity = bucket_identity(device)
    prepared_at = time.monotonic()
    specs = []
    for name, tensor in complete_exports(worker):
        expert = any(
            name.endswith(f".experts.{projection}.weight") for projection in ("gate_proj", "up_proj", "down_proj")
        )
        specs.append(TensorSpec(name, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."), expert))
    if not specs:
        raise ValueError("The frozen source exporter returned no tensors")
    # The catalogue retains metadata only, not the last converted tensor.
    del tensor
    manifest = build_manifest(specs, bucket_bytes=BUCKET_BYTES)
    hashes = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(hashes, manifest.manifest_id)
    if any(value != manifest.manifest_id for value in hashes):
        raise ValueError("Policy ranks disagree on the canonical export manifest")
    torch.cuda.synchronize(device)
    catalogue_allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    tasks = worker.bridge.get_conversion_tasks(worker.actor_module)
    local_slices, local_sources = local_source_slices(tasks, worker.provider)
    source_catalogue = gather_source_catalogue(
        {"rank": rank, **source_owners, "slices": [asdict(item) for item in local_slices]}
    )
    source_plan = frozen_source_plan(manifest, source_catalogue)
    source_bytes = Counter()
    for bucket in source_plan:
        for part in bucket:
            source_bytes[part.owner_rank] += part.source.numel * getattr(torch, part.source.wire_dtype).itemsize
    source_catalogue_sha256 = hashlib.sha256(
        json.dumps(source_catalogue, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.cuda.synchronize(device)
    catalogue_peak_extra = torch.cuda.max_memory_allocated(device) - catalogue_allocated_before
    if catalogue_peak_extra > MAX_REPLAY_EXTRA_BYTES:
        raise ValueError("Replay-only source catalogue introduced more than 1 MiB of GPU allocation")
    catalogue_memory = {
        "backend": "gloo",
        "allocated_before": catalogue_allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_extra_bytes": catalogue_peak_extra,
        "scope": "replay-only source task/catalogue construction and CPU group creation/gather/destruction",
    }
    free, total = torch.cuda.mem_get_info(device)
    if free < 2 * BUCKET_BYTES + MAX_REPLAY_EXTRA_BYTES:
        raise ValueError("A policy rank lacks headroom for both transfer buffers")
    allocated_before = torch.cuda.memory_allocated(device)
    buffers = tuple(torch.empty(BUCKET_BYTES, device=device, dtype=torch.uint8) for _ in range(2))
    buffer_delta = torch.cuda.memory_allocated(device) - allocated_before
    engine_count = generator.num_inference_engines
    ranks_per_engine = (
        generator.inference_engine_tensor_parallel_size
        * generator.inference_engine_data_parallel_size
        * generator.inference_engine_pipeline_parallel_size
    )
    prepared_receivers = None
    receiver_identities = None
    if rank == 0:
        prepared_receivers, receiver_identities = receiver_rows(
            await client.prepare_diagnostic_weight_sync_buckets(asdict(manifest), manifest.manifest_id),
            engine_count=engine_count,
            ranks_per_engine=ranks_per_engine,
            data_parallel_size=generator.inference_engine_data_parallel_size,
        )
        if any(row["manifest_id"] != manifest.manifest_id for row in prepared_receivers):
            raise ValueError("A receiver prepared a different manifest")
    torch.distributed.barrier()
    preparation_seconds = time.monotonic() - prepared_at
    return PreparedBucketTransfer(
        manifest,
        source_plan,
        local_slices,
        local_sources,
        buffers,
        prepared_receivers,
        receiver_identities,
        {
            "schema": "megatron_bucket_install_replay_v1",
            "identity": identity,
            "manifest": asdict(manifest) if rank == 0 else None,
            "manifest_id": manifest.manifest_id,
            "policy_manifest_agreement": len(hashes),
            "frozen_source_owners": sorted({part.owner_rank for bucket in source_plan for part in bucket}),
            "frozen_source_segments": sum(len(bucket) for bucket in source_plan),
            "frozen_source_bytes_by_owner": dict(sorted(source_bytes.items())),
            "source_catalogue_sha256": source_catalogue_sha256,
            "source_catalogue_memory": catalogue_memory,
            "source_byte_coverage": 1.0,
            "local_source_parameters": len(local_sources),
            "free_bytes_before_buffers": free,
            "total_device_bytes": total,
            "buffer_allocated_delta": buffer_delta,
            "preparation_seconds": preparation_seconds,
            "prepared_receivers": prepared_receivers,
            "timing_scope": "install includes complete export, pack, transfer, load joins and final barrier; receipt persistence and replay excluded",
        },
    )


async def run_bucket_phase(worker, client, prepared, *, replay, start_versions, start_update, publication_id=None):
    manifest = prepared.manifest
    source_plan = prepared.source_plan
    local_sources = prepared.local_sources
    buffers = prepared.buffers
    receiver_identities = prepared.receiver_identities
    generator = worker.cfg.generator
    rank = torch.distributed.get_rank()
    device = torch.cuda.current_device()
    engine_count = generator.num_inference_engines
    ranks_per_engine = (
        generator.inference_engine_tensor_parallel_size
        * generator.inference_engine_data_parallel_size
        * generator.inference_engine_pipeline_parallel_size
    )
    sync_identity = (
        {} if publication_id is None else {"manifest_id": manifest.manifest_id, "publication_id": publication_id}
    )
    if parameter_versions(worker) != start_versions or worker._completed_update != start_update:
        raise ValueError("Policy weights changed inside the frozen diagnostic interval")
    torch.cuda.synchronize(device)
    allocated_at_phase = torch.cuda.memory_allocated(device)
    free_at_phase = torch.cuda.mem_get_info(device)[0]
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    sender = (
        FrozenViewBucketSender(manifest, source_plan, local_sources, buffers)
        if replay
        else StreamingBucketSender(manifest, complete_exports(worker), buffers)
    )
    per_bucket = []
    for bucket in range(manifest.bucket_count):
        buffer = sender.pack_next_bucket()
        if rank == 0:
            receive_task = asyncio.create_task(
                client.receive_diagnostic_weight_sync_bucket(bucket, replay=replay, **sync_identity)
            )
            send_stream = torch.cuda.current_stream(device)

            def broadcast():
                # Preserve the same stream even when NCCL dispatch occurs
                # on a helper thread, while the actor loop services RPCs.
                with torch.cuda.device(device), torch.cuda.stream(send_stream):
                    torch.distributed.broadcast(buffer, 0, group=worker._model_update_group)

            try:
                await asyncio.to_thread(broadcast)
            except BaseException:
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)
                raise
            sender.mark_bucket_sent(bucket)
            rows, _ = receiver_rows(
                await receive_task,
                engine_count=engine_count,
                ranks_per_engine=ranks_per_engine,
                data_parallel_size=generator.inference_engine_data_parallel_size,
                expected=receiver_identities,
            )
            if any(row["bucket_id"] != bucket or not row["load_completion_event_recorded"] for row in rows):
                raise ValueError("Receiver bucket sequence or load event receipt is incomplete")
            per_bucket.append(
                {
                    "bucket_id": bucket,
                    "wire_bytes": buffer.numel(),
                    "receiver_bytes": [row["compared_bytes" if replay else "installed_bytes"] for row in rows],
                    "mismatches": [row["mismatches"] for row in rows] if replay else None,
                }
            )
        else:
            # All policy ranks currently perform the complete bridge
            # export/pack; only rank zero uses the update communicator.
            sender.mark_bucket_sent(bucket)
        torch.distributed.barrier()
    sender_receipt = sender.finish()
    receivers = None
    if rank == 0:
        method = client.finish_diagnostic_weight_sync_replay if replay else client.finish_diagnostic_weight_sync_install
        receivers, _ = receiver_rows(
            await method(**sync_identity),
            engine_count=engine_count,
            ranks_per_engine=ranks_per_engine,
            data_parallel_size=generator.inference_engine_data_parallel_size,
            expected=receiver_identities,
        )
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    elapsed = time.monotonic() - started
    peak_extra = torch.cuda.max_memory_allocated(device) - allocated_at_phase
    receipt = {
        "seconds": elapsed,
        "peak_extra_bytes": peak_extra,
        "allocated_before": allocated_at_phase,
        "allocated_after": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "free_device_bytes_before": free_at_phase,
        "free_device_bytes_after": torch.cuda.mem_get_info(device)[0],
        "memory_scope": "Torch allocator peak plus device-free endpoints; external allocator peak unmeasured",
        "sender": sender_receipt,
        "receivers": receivers,
        "buckets": per_bucket if rank == 0 else None,
    }
    # Release the completed exporter before taking the next phase baseline.
    # Otherwise a freed conversion tensor could mask new replay allocation.
    del sender
    if parameter_versions(worker) != start_versions or worker._completed_update != start_update:
        raise ValueError("Policy weights changed before the transfer phase completed")
    return receipt


def persist_bucket_phase(worker, prepared, receipt, *, replay, start_update, publication_id=None):
    rank = torch.distributed.get_rank()
    if rank != 0:
        if publication_id is not None:
            phase = "replay" if replay else "install"
            persist_readback(
                worker.cfg.trainer.weight_sync_readback_output,
                f"bucket-{phase}-sender-sync-{publication_id}",
                {
                    "identity": prepared.metadata["identity"],
                    "manifest_id": prepared.manifest.manifest_id,
                    "publication_id": publication_id,
                    "completed_update": start_update,
                    **receipt,
                },
            )
        return
    receivers = receipt["receivers"]
    per_bucket = receipt["buckets"]
    identity = prepared.metadata["identity"]
    manifest = prepared.manifest
    prepared_receivers = prepared.prepared_receivers
    phase_name = "bucket-replay-receivers" if replay else "bucket-install-receivers"
    if publication_id is not None:
        phase_name += f"-sync-{publication_id}"
    persist_readback(
        worker.cfg.trainer.weight_sync_readback_output,
        phase_name,
        {
            "sender_identity": identity,
            "manifest_id": manifest.manifest_id,
            "completed_update": start_update,
            "receivers": receivers,
            "prepared_receivers": prepared_receivers,
            "buckets": per_bucket,
            "validation_status": "not_yet_validated",
            "sender_phase": {key: value for key, value in receipt.items() if key not in ("receivers", "buckets")},
        },
    )


def validate_bucket_phase(prepared, receipt, *, replay, rank):
    if rank == 0:
        receivers = receipt["receivers"]
        per_bucket = receipt["buckets"]
        manifest = prepared.manifest
        prepared_receivers = prepared.prepared_receivers
        if any(row["manifest_id"] != manifest.manifest_id for row in receivers):
            raise ValueError("Receiver completion changed manifest identity")
        for index, row in enumerate(receivers):
            expected_bytes = prepared_receivers[index]["installed_parameter_bytes"]
            if sum(bucket["receiver_bytes"][index] for bucket in per_bucket) != expected_bytes:
                raise ValueError("Per-bucket receipt coverage differs from installed parameter bytes")
            if replay:
                if (
                    row["mismatches"] != 0
                    or row["coverage"] != 1.0
                    or row["compared_bytes"] != expected_bytes
                    or row["expected_bytes"] != expected_bytes
                    or sum(bucket["mismatches"][index] for bucket in per_bucket) != 0
                    or row["replay_peak_extra_bytes"] > MAX_REPLAY_EXTRA_BYTES
                ):
                    raise ValueError("Full-byte replay or its allocation gate failed")
            elif not row["install_complete"] or row["completed_slots"] != list(range(min(2, manifest.bucket_count))):
                raise ValueError("Receiver installation did not join every used load slot")
    if replay and receipt["peak_extra_bytes"] > MAX_REPLAY_EXTRA_BYTES:
        raise ValueError("Sender replay introduced more than 1 MiB of additional allocation")


async def install_and_replay(worker, client, *, source_owners):
    """Caller holds PolicyWeightAccess across install and original-view replay."""
    if worker._policy_weight_access.owner != "bucket-install-and-replay":
        raise RuntimeError("Bucket diagnostic requires exclusive policy-weight ownership")
    start_versions = parameter_versions(worker)
    start_update = worker._completed_update
    rank = torch.distributed.get_rank()
    device = torch.cuda.current_device()
    primary_error = None
    phases = {}
    try:
        prepared = await prepare_bucket_transfer(worker, client, source_owners=source_owners)
        for replay in (False, True):
            receipt = await run_bucket_phase(
                worker, client, prepared, replay=replay, start_versions=start_versions, start_update=start_update
            )
            persist_bucket_phase(worker, prepared, receipt, replay=replay, start_update=start_update)
            validate_bucket_phase(prepared, receipt, replay=replay, rank=rank)
            phases["replay" if replay else "install"] = receipt
        return {
            **prepared.metadata,
            "completed_update_before": start_update,
            "completed_update_after": worker._completed_update,
            "exclusive_weight_owner": worker._policy_weight_access.owner,
            "parameter_version_tripwire_unchanged": True,
            "phases": phases,
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        await close_preserving_failure(client, device, rank, primary_error)
