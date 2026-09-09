"""Audit retained native timing receipts without executing a weight transfer."""

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
from types import SimpleNamespace

from skyrl_train.weight_sync.initial_readback import validate_communicator_environment
from skyrl_train.weight_sync.manifest import parse_manifest
from skyrl_train.weight_sync.megatron_bucket_protocol import validate_bucket_phase
from skyrl_train.weight_sync.reference_bucket_protocol import manifest_wire_inventory

SCRATCH_LIMIT = 1024 * 1024
IDENTITY_FIELDS = ("host", "pid", "rank", "world_size", "device", "gpu_uuid", "attempt_uid", "task_id")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    result = tuple(row[key] for key in IDENTITY_FIELDS)
    require(all(value is not None and value != "" for value in result), "Missing native rank identity")
    return result


def indexed(rows, count):
    require(len(rows) == count, "Missing or duplicate native rank receipt")
    result = {row["identity"]["rank"]: row for row in rows}
    require(set(result) == set(range(count)), "Native ranks are incomplete")
    require(len({identity(row["identity"]) for row in rows}) == count, "Duplicate native identity")
    return result


def phase_rows(documents, phase, version):
    prefix = f"bucket-{phase}-"
    suffix = f"-sync-{version}-"
    result = []
    for name, row in documents.items():
        if not name.startswith(prefix) or suffix not in name:
            continue
        if name.startswith(f"bucket-{phase}-receivers-"):
            result.append(
                {
                    "identity": row["sender_identity"],
                    "manifest_id": row["manifest_id"],
                    "completed_update": row["completed_update"],
                    **row["sender_phase"],
                    "receivers": row["receivers"],
                    "buckets": row["buckets"],
                }
            )
        elif name.startswith(f"bucket-{phase}-sender-"):
            require(row["publication_id"] == version, "Sender filename/version mismatch")
            result.append(row)
    return result


def memory(row, *, before, peak, extra):
    values = [row[before], row[peak], row[extra]]
    require(all(type(value) is int and value >= 0 for value in values), "Invalid native allocation value")
    require(row[peak] - row[before] == row[extra], "Native allocation arithmetic differs")
    require(row[extra] <= SCRATCH_LIMIT, "Proof exceeds one MiB additional allocation")


def audit_timing_receipts(documents, metrics, *, mode, attempt_uids, policy_ranks=32, receiver_ranks=8, syncs=20):
    """Join every selected measured rank/version to native byte, memory and timing evidence.

    The caller must bind these exact artifact bytes and selected attempt UIDs to
    the submitted native request and physical task history. This audit does not
    infer server provenance or terminal task cost from diagnostic JSON.
    """
    require(mode in ("reference", "bucket"), "Unknown transfer mode")
    prepared = indexed(
        [row for name, row in documents.items() if name.startswith("bucket-timing-prepared-")], policy_ranks
    )
    root = prepared[0]
    manifest = parse_manifest(root["manifest"], root["manifest_id"])
    wire_bytes = sum(part.nbytes for part in manifest.entries)
    receivers = root["prepared_receivers"]
    require(len(receivers) == receiver_ranks, "Missing receiver preparation")
    receiver_ids = [identity(row["identity"]) for row in receivers]
    require(len(set(receiver_ids)) == receiver_ranks, "Duplicate receiver identity")
    all_rows = [*prepared.values(), *receivers]
    require({row["identity"]["attempt_uid"] for row in all_rows} <= set(attempt_uids), "Unbound native attempt")
    validate_communicator_environment(all_rows)
    proof_peak = 0
    for rank, row in prepared.items():
        require(row["identity"]["world_size"] == policy_ranks, "Policy world size changed")
        require(row["manifest_id"] == manifest.manifest_id, "Policy manifest disagreement")
        require(row["policy_manifest_agreement"] == policy_ranks, "Incomplete policy manifest agreement")
        require(row["source_byte_coverage"] == 1.0, "Incomplete frozen source coverage")
        require(sum(row["frozen_source_bytes_by_owner"].values()) == wire_bytes, "Source-owner byte coverage differs")
        catalogue = row["source_catalogue_memory"]
        require(catalogue["backend"] == "gloo", "Replay catalogue did not use CPU collectives")
        memory(catalogue, before="allocated_before", peak="peak_allocated_bytes", extra="peak_extra_bytes")
        proof_peak = max(proof_peak, catalogue["peak_extra_bytes"])
        require(row["buffer_allocated_delta"] >= 2 * manifest.bucket_bytes, "Transfer buffers omitted from storage")
    versions = set(range(1, syncs + 1))
    observed_versions = {
        int(match[1]) for name in documents if (match := re.match(r"bucket-timing-begin-sync-(\d+)-", name))
    }
    require(observed_versions == versions, "Measured sync versions differ from the frozen protocol")
    for phase in ("install", "replay"):
        phase_versions = {
            int(match[1])
            for name in documents
            if (match := re.match(rf"bucket-{phase}-(?:receivers|sender)-sync-(\d+)-", name))
        }
        require(phase_versions == versions, "Orphan or missing measured phase version")
    marker = documents["bucket-measurement-started.json"]
    require(marker["attempt_uid"] in attempt_uids, "Measurement marker is not a selected native attempt")
    metadata_inventory = manifest_wire_inventory(manifest)
    installed_bytes = [row["installed_parameter_bytes"] for row in receivers]
    for version in sorted(versions):
        begins = indexed(
            [row for name, row in documents.items() if name.startswith(f"bucket-timing-begin-sync-{version}-")],
            policy_ranks,
        )
        for rank, begin in begins.items():
            require(begin["publication_id"] == version, "Begin filename/version mismatch")
            require(identity(begin["identity"]) == identity(prepared[rank]["identity"]), "Policy restarted before sync")
            memory(
                begin,
                before="source_refresh_allocated_before",
                peak="source_refresh_peak_allocated_bytes",
                extra="source_refresh_peak_extra_bytes",
            )
            proof_peak = max(proof_peak, begin["source_refresh_peak_extra_bytes"])
        require(begins[0]["measurement_marker"]["attempt_uid"] == marker["attempt_uid"], "Measurement marker changed")
        begin_receivers = begins[0]["receivers"]
        require(
            [identity(row["identity"]) for row in begin_receivers] == receiver_ids, "Receiver begin identities changed"
        )
        require(
            all(
                type(row["publication_id"]) is int
                and row["publication_id"] == version
                and row["manifest_id"] == manifest.manifest_id
                for row in begin_receivers
            ),
            "Receiver begin version changed",
        )
        for phase in ("install", "replay"):
            rows = indexed(phase_rows(documents, phase, version), policy_ranks)
            for rank, row in rows.items():
                require(
                    identity(row["identity"]) == identity(prepared[rank]["identity"]), "Policy restarted during sync"
                )
                require(row["manifest_id"] == manifest.manifest_id, "Phase manifest changed")
                require(row["completed_update"] == version, "Frozen policy update differs from measured version")
                require(math.isfinite(row["seconds"]) and row["seconds"] > 0, "Missing native phase duration")
                if phase == "replay" or mode == "bucket":
                    require(row["sender"]["wire_bytes"] == wire_bytes, "Native sender byte count changed")
                    require(row["sender"]["send_completion_joined"], "Sender did not join transfer completion")
                if phase == "replay":
                    memory(row, before="allocated_before", peak="peak_allocated_bytes", extra="peak_extra_bytes")
                    proof_peak = max(proof_peak, row["peak_extra_bytes"])
            root_phase = rows[0]
            require(
                [identity(row["identity"]) for row in root_phase["receivers"]] == receiver_ids,
                "Receiver identities changed or are incomplete",
            )
            for receiver in root_phase["receivers"]:
                require(
                    type(receiver["publication_id"]) is int and receiver["publication_id"] == version,
                    "Receiver completed another sync version",
                )
                require(receiver["manifest_id"] == manifest.manifest_id, "Receiver manifest changed")
            if phase == "install" and mode == "reference":
                inventory = root_phase["sender"]
                require(
                    inventory["entries"] == metadata_inventory and inventory["wire_bytes"] == wire_bytes,
                    "Original broadcaster has different tensor order or wire bytes",
                )
                for i, receiver in enumerate(root_phase["receivers"]):
                    require(receiver["original_reload_complete"], "Original receiver reload did not complete")
                    require(
                        receiver["wire_inventory"]["entries"] == inventory["entries"], "Original load order differs"
                    )
                    require(receiver["wire_inventory"]["wire_bytes"] == wire_bytes, "Original load bytes differ")
                    require(receiver["installed_parameter_bytes"] == installed_bytes[i], "Installed coverage changed")
                    memory(
                        receiver,
                        before="reference_bind_allocated_before",
                        peak="reference_bind_peak_allocated_bytes",
                        extra="reference_bind_peak_extra_bytes",
                    )
                    proof_peak = max(proof_peak, receiver["reference_bind_peak_extra_bytes"])
            else:
                validate_bucket_phase(
                    SimpleNamespace(manifest=manifest, prepared_receivers=receivers),
                    root_phase,
                    replay=phase == "replay",
                    rank=0,
                )
                require(len(root_phase["buckets"]) == manifest.bucket_count, "Missing native bucket receipt")
                require(
                    [row["bucket_id"] for row in root_phase["buckets"]] == list(range(manifest.bucket_count)),
                    "Native bucket order differs",
                )
                require(
                    sum(row["wire_bytes"] for row in root_phase["buckets"]) == wire_bytes, "Bucket wire sum differs"
                )
            if phase == "replay":
                for receiver in root_phase["receivers"]:
                    memory(
                        receiver,
                        before="allocated_before",
                        peak="peak_allocated_bytes",
                        extra="replay_peak_extra_bytes",
                    )
                    proof_peak = max(proof_peak, receiver["replay_peak_extra_bytes"])
    require(len(metrics) == syncs and {row["step"] for row in metrics} == versions, "Driver timing coverage differs")
    timings = [row["weight_broadcast"] for row in metrics]
    require(all(math.isfinite(value) and value > 0 for value in timings), "Invalid driver broadcast time")
    return {
        "mode": mode,
        "syncs": syncs,
        "policy_ranks": policy_ranks,
        "receiver_ranks": receiver_ranks,
        "wire_bytes_per_sync": wire_bytes,
        "ordered_wire_inventory": metadata_inventory,
        "installed_bytes_per_receiver": installed_bytes,
        "mismatches": 0,
        "coverage": 1.0,
        "maximum_replay_extra_bytes": proof_peak,
        "weight_broadcast_seconds": timings,
        "weight_broadcast_p50_seconds": median(timings),
        "scope": "Selected measured native intervals; physical task provenance/cost audited separately",
    }


def audit_timing_pair(reference, candidate, *, expected_syncs=20):
    require(reference["mode"] == "reference" and candidate["mode"] == "bucket", "Wrong paired transfer modes")
    for key in (
        "syncs",
        "policy_ranks",
        "receiver_ranks",
        "wire_bytes_per_sync",
        "ordered_wire_inventory",
        "installed_bytes_per_receiver",
    ):
        require(reference[key] == candidate[key], f"Matched gate differs: {key}")
    for arm in (reference, candidate):
        require(
            arm["syncs"] == expected_syncs and arm["mismatches"] == 0 and arm["coverage"] == 1.0,
            "Pair lacks the frozen complete proof",
        )
        require(arm["maximum_replay_extra_bytes"] <= SCRATCH_LIMIT, "Pair exceeds replay memory")
        values = arm["weight_broadcast_seconds"]
        require(
            len(values) == expected_syncs and all(math.isfinite(value) and value > 0 for value in values),
            "Pair timing observations are incomplete",
        )
        require(median(values) == arm["weight_broadcast_p50_seconds"], "Pair median differs from observations")
    historical_threshold = 7.477672202046961
    current_ratio = candidate["weight_broadcast_p50_seconds"] / reference["weight_broadcast_p50_seconds"]
    return {
        "primary_historical_threshold_seconds": historical_threshold,
        "historical_timing_pass": candidate["weight_broadcast_p50_seconds"] <= historical_threshold,
        "matched_reference_ratio": current_ratio,
        "matched_half_reference_pass": current_ratio <= 0.5,
        "equal_wire_bytes": True,
        "full_byte_proof": True,
        "scope": "Proof-instrumented timing gate; no production throughput or quality claim",
    }


def main():
    """Read a retained JSON packet; native request/attempt binding remains external."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_bytes())
    results = {}
    for mode in ("reference", "bucket"):
        arm = packet[mode]
        results[mode] = audit_timing_receipts(
            arm["documents"], arm["metrics"], mode=mode, attempt_uids=arm["attempt_uids"]
        )
    results["pair"] = audit_timing_pair(results["reference"], results["bucket"])
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    passed = results["pair"]["historical_timing_pass"]
    print("SNOWBALL_BUCKET_TIMING_AUDIT_PASS" if passed else "SNOWBALL_BUCKET_TIMING_AUDIT_FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
