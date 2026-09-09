"""Zero-training native byte/memory qualification with durable attempt evidence."""

import asyncio
import hashlib
import json
import math
import os
import time

from skyrl_train.io.io import exists, read_bytes, write_bytes_atomic
from skyrl_train.weight_sync.initial_readback import run_initial_readback
from skyrl_train.weight_sync.manifest import parse_manifest
from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.worker_bucket_protocol import BUCKET_BYTES, MAX_REPLAY_EXTRA_BYTES


def mark_measurement_once(output_uri: str) -> dict:
    """Startup retries may proceed; a recorded measurement is never repeated."""
    uri = f"{output_uri.rstrip('/')}/bucket-measurement-started.json"
    if exists(uri):
        raise ValueError("A previous attempt already entered bucket measurement")
    attempt = os.environ["IRIS_ATTEMPT_UID"]
    if not attempt:
        raise ValueError("Measurement requires native attempt identity")
    payload = json.dumps({"attempt_uid": attempt, "unix_ns": time.time_ns()}, sort_keys=True).encode()
    write_bytes_atomic(uri, payload)
    if read_bytes(uri) != payload:
        raise ValueError("Measurement marker readback differs from the written bytes")
    return {"uri": uri, "sha256": hashlib.sha256(payload).hexdigest(), "attempt_uid": attempt}


def validate_bucket_prerequisites(reference: dict) -> None:
    if not reference["precursor_coverage"] or not all(reference["precursor_coverage"].values()):
        raise ValueError("Native source or receiver precursor coverage is incomplete")
    for engine in reference["receivers"]:
        for row in engine:
            if row["free_bytes"] < 2 * BUCKET_BYTES + MAX_REPLAY_EXTRA_BYTES:
                raise ValueError("Native receiver lacks two-buffer and replay headroom")
            if not row["layers"] or any(layer["backend"] != "TRITON" for layer in row["layers"]):
                raise ValueError("Native receiver backend is not qualified for direct expert installation")


def validate_bucket_results(rows: list[dict], geometry: dict) -> None:
    if sorted(row["identity"]["rank"] for row in rows) != list(range(geometry["policy_ranks"])):
        raise ValueError("Bucket diagnostic is missing policy rank receipts")
    if any(row["identity"]["world_size"] != geometry["policy_ranks"] for row in rows):
        raise ValueError("Policy world size changed during the diagnostic")
    if len({(row["identity"]["host"], row["identity"]["gpu_uuid"]) for row in rows}) != len(rows):
        raise ValueError("Policy receipt identities share a physical GPU")
    if len({row["manifest_id"] for row in rows}) != 1 or len({row["source_catalogue_sha256"] for row in rows}) != 1:
        raise ValueError("Policy ranks disagree on source or destination coverage")
    root = next(row for row in rows if row["identity"]["rank"] == 0)
    manifest = parse_manifest(root["manifest"], root["manifest_id"])
    wire_bytes = sum(entry.nbytes for entry in manifest.entries)
    for row in rows:
        if (
            row["completed_update_before"] != row["completed_update_after"]
            or row["completed_update_after"] not in (None, 0)
            or row["exclusive_weight_owner"] != "bucket-install-and-replay"
            or not row["parameter_version_tripwire_unchanged"]
            or row["source_byte_coverage"] != 1.0
            or row["source_catalogue_memory"]["backend"] != "gloo"
            or row["source_catalogue_memory"]["peak_extra_bytes"] > MAX_REPLAY_EXTRA_BYTES
            or sum(row["frozen_source_bytes_by_owner"].values()) != wire_bytes
            or row["policy_manifest_agreement"] != geometry["policy_ranks"]
        ):
            raise ValueError("Policy freeze or complete source-byte coverage failed")
        if set(row["phases"]) != {"install", "replay"}:
            raise ValueError("A policy rank omitted a diagnostic phase")
        for phase, result in row["phases"].items():
            if not math.isfinite(result["seconds"]) or result["seconds"] <= 0:
                raise ValueError("A policy phase lacks finite elapsed timing")
            if result["sender"]["wire_bytes"] != wire_bytes or not result["sender"]["send_completion_joined"]:
                raise ValueError("Sender byte coverage or completion join failed")
            if phase == "replay" and result["peak_extra_bytes"] > MAX_REPLAY_EXTRA_BYTES:
                raise ValueError("Sender replay scratch exceeds one MiB")
    expected_receivers = geometry["receiver_engines"] * geometry["receiver_ranks_per_engine"]
    if len(root["prepared_receivers"]) != expected_receivers:
        raise ValueError("Prepared receiver coverage is incomplete")
    for phase in ("install", "replay"):
        if len(root["phases"][phase]["receivers"]) != expected_receivers:
            raise ValueError("Completed receiver coverage is incomplete")
    for prepared, replay in zip(root["prepared_receivers"], root["phases"]["replay"]["receivers"], strict=True):
        if (
            replay["compared_bytes"] != prepared["installed_parameter_bytes"]
            or replay["coverage"] != 1.0
            or replay["mismatches"] != 0
            or replay["replay_peak_extra_bytes"] > MAX_REPLAY_EXTRA_BYTES
        ):
            raise ValueError("Receiver full-byte comparison or replay scratch failed")


async def run_bucket_qualification(trainer, output_uri: str, geometry: dict) -> dict:
    reference = await run_initial_readback(trainer, output_uri, geometry)
    reference_durable = persist_readback(output_uri, "reference", reference)
    validate_bucket_prerequisites(reference)
    await trainer.inference_engine_client.pause_generation()
    marker = mark_measurement_once(output_uri)
    rows = await asyncio.gather(
        *trainer.policy_model.async_run_ray_method(
            "pass_through", "diagnostic_bucket_install_and_replay", trainer.inference_engine_client
        )
    )
    # Retain native evidence before interpreting the aggregate gate.
    raw_durable = persist_readback(output_uri, "bucket-native", {"policy": rows})
    validate_bucket_results(rows, geometry)
    if trainer.global_step != 0:
        raise ValueError("Zero-training diagnostic advanced the trainer step")
    await trainer.inference_engine_client.resume_generation()
    return {
        "schema": "snowball_bucket_byte_memory_v1",
        "updates": 0,
        "initial_syncs": 1,
        "packed_installs": 1,
        "full_replays": 1,
        "reference": reference,
        "reference_durable": reference_durable,
        "measurement_marker": marker,
        "bucket_native_durable": raw_durable,
        "policy": rows,
        "requested_geometry": geometry,
        "timing_scope": "one diagnostic sample; no latency gate or speedup claim",
    }
