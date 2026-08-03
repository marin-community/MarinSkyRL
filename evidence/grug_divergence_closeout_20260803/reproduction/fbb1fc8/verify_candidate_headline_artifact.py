#!/usr/bin/env python3
"""Independently verify the pinned candidate's one authorized 32-H100 pair."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


EXPECTED_URI = (
    "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/"
    "divergence-closeout-fbb1fc8/headline-paired-s1.json"
)
EXPECTED_PINS = {
    "source_revision": "fbb1fc8378601e0346d00d186809f10d1ad0360d",
    "image": (
        "ghcr.io/marin-community/marinskyrl@"
        "sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770"
    ),
    "model": "marin-community/grug-67b-a2b-sft-s2-thinking-step630",
    "model_revision": "a822321c2c21af099189e7116104b3cf5142c119",
    "manifest_sha256": "5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74",
    "logical_batch_sha256": "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d",
    "runtime_benchmark_sha256": "b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404",
}
EXPECTED_MOE_SHA256 = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"
EXPECTED_WORKER_SHA256 = "c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1"
OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-3
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4
MINIMUM_SPEEDUP = 1.10


def s3_client():
    kwargs = {"config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"})}
    endpoint = os.environ.get("CW_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
    access_key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def topology(rows):
    return [(row["rank"], row["host"], row["phys_uuid"]) for row in rows]


def topology_valid(rows):
    normalized = topology(rows)
    if [row[0] for row in normalized] != list(range(32)):
        return False
    if len({row[2] for row in normalized}) != 32 or len({row[1] for row in normalized}) != 4:
        return False
    return all(len({row[1] for row in normalized[start : start + 8]}) == 1 for start in range(0, 32, 8))


def empty_comparison():
    return {
        "checked": 0,
        "nonfinite": 0,
        "violations": 0,
        "max_abs_difference": 0.0,
        "max_allowance_ratio": 0.0,
    }


def record_close(summary, actual, expected, *, rtol, atol):
    summary["checked"] += 1
    if not math.isfinite(actual) or not math.isfinite(expected):
        summary["nonfinite"] += 1
        summary["violations"] += 1
        summary["max_allowance_ratio"] = math.inf
        return
    difference = abs(actual - expected)
    allowance = atol + rtol * abs(expected)
    summary["max_abs_difference"] = max(summary["max_abs_difference"], difference)
    summary["max_allowance_ratio"] = max(summary["max_allowance_ratio"], difference / allowance)
    summary["violations"] += int(difference > allowance)


def compare_paired_numeric(candidate, reference):
    output = empty_comparison()
    gradient = empty_comparison()
    if len(candidate["timed_workers"]) != len(reference["timed_workers"]):
        raise RuntimeError("paired numeric worker counts differ")
    for candidate_worker, reference_worker in zip(
        candidate["timed_workers"], reference["timed_workers"], strict=True
    ):
        if candidate_worker["rank"] != reference_worker["rank"]:
            raise RuntimeError("paired numeric worker ranks differ")
        for actual, expected in zip(
            candidate_worker["representative_action_log_probs"],
            reference_worker["representative_action_log_probs"],
            strict=True,
        ):
            record_close(output, actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        candidate_gradients = candidate_worker["representative_gradients"]
        reference_gradients = reference_worker["representative_gradients"]
        if sorted(candidate_gradients) != sorted(reference_gradients):
            raise RuntimeError("paired representative gradient names differ")
        for name, candidate_gradient in candidate_gradients.items():
            reference_gradient = reference_gradients[name]
            if candidate_gradient["local_numel"] != reference_gradient["local_numel"]:
                raise RuntimeError(f"paired representative gradient shard differs: {name}")
            for field in ("l2_norm", "max_abs"):
                record_close(
                    gradient,
                    candidate_gradient[field],
                    reference_gradient[field],
                    rtol=GRADIENT_RTOL,
                    atol=GRADIENT_ATOL,
                )
            for actual, expected in zip(
                candidate_gradient["samples"], reference_gradient["samples"], strict=True
            ):
                record_close(gradient, actual, expected, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
    loss = empty_comparison()
    record_close(
        loss,
        candidate["metrics"]["matched_global_ce_loss"],
        reference["metrics"]["matched_global_ce_loss"],
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )
    for comparison in (output, loss, gradient):
        comparison["passed"] = comparison["violations"] == 0 and comparison["nonfinite"] == 0
    return {
        "representative_action_log_probs": output,
        "matched_global_ce": loss,
        "representative_gradients": gradient,
    }


def arm_check(arm, initial_topology, initial_identities, baseline_hashes, staging):
    implementation = arm["arm"]
    identities = arm["worker_identities"]
    workers = arm["timed_workers"]
    identity_contract = all(
        identity["rank"] == rank
        and identity["expert_implementation"] == implementation
        and identity["runtime_grug_module_sha256"] == EXPECTED_MOE_SHA256
        and identity["runtime_worker_sha256"] == EXPECTED_WORKER_SHA256
        for rank, identity in enumerate(identities)
    )
    identity_equal = all(
        {key: value for key, value in current.items() if key not in {"expert_implementation", "native_grouped_mm"}}
        == {key: value for key, value in baseline.items() if key not in {"expert_implementation", "native_grouped_mm"}}
        for current, baseline in zip(identities, initial_identities, strict=True)
    )
    selection_restore = all(
        row["rank"] == rank
        and row["implementation"] == implementation
        and row["selected_blocks"] == row["sparse_blocks"]
        and row["gradient_tensors"] == 0
        and row["cpu_rng_restored"]
        and row["cuda_rng_restored"]
        for rank, row in enumerate(arm["selection_restore"])
    )
    warmup_restore = all(
        row["rank"] == rank
        and row["state_hash_before"] == baseline_hashes[rank]
        and row["state_hash_after"] == baseline_hashes[rank]
        for rank, row in enumerate(arm["warmup_restore"])
    )
    timed_contract = True
    for rank, (worker, identity, staged) in enumerate(zip(workers, identities, staging, strict=True)):
        expected_blocks = identity["num_hidden_layers"]
        expected_calls = expected_blocks * 128
        expected_layer_load = staged["allocated_tokens"] * identity["num_experts_per_tok"]
        evidence = worker["expert_attribution"]
        timed_contract &= (
            worker["rank"] == rank
            and worker["microbatches"] == 128
            and worker["gradient_tensors"] > 0
            and worker["gradient_numel"] > 0
            and worker["nonfinite_gradient_tensors"] == 0
            and evidence is not None
            and evidence["module_count"] == expected_blocks
            and evidence["call_counts"] == {"forward": expected_calls, "backward": expected_calls}
            and evidence["route_calls_per_layer"] == [128] * expected_blocks
            and evidence["paired_route_mode"] is None
            and not evidence["paired_route_comparison"]
            and len(evidence["route_loads_per_layer"]) == expected_blocks
            and all(
                len(loads) == identity["num_local_experts"] and sum(loads) == expected_layer_load
                for loads in evidence["route_loads_per_layer"]
            )
        )
    check = {
        "topology_valid": topology_valid(arm["topology"]),
        "topology_equal_initial": topology(arm["topology"]) == initial_topology,
        "worker_identity_equal_initial_except_intervention": identity_equal,
        "identity_contract": identity_contract,
        "selection_restore_ok": selection_restore,
        "warmup_state_restored": warmup_restore,
        "timed_route_and_allocation_contract": timed_contract,
    }
    check["passed"] = all(check.values())
    return check


uri, raw_path, summary_path = sys.argv[1:]
if uri != EXPECTED_URI:
    raise RuntimeError(f"unexpected artifact URI: {uri}")
parsed = urlparse(uri)
payload = s3_client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
Path(raw_path).write_bytes(payload)
payload_sha256 = hashlib.sha256(payload).hexdigest()
result = json.loads(payload)
claimed = result.pop("result_sha256")
actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if actual != claimed:
    raise RuntimeError(f"result digest mismatch: claimed={claimed} actual={actual}")
result["result_sha256"] = claimed
for field, expected in EXPECTED_PINS.items():
    if result[field] != expected:
        raise RuntimeError(f"unexpected {field}: expected={expected} actual={result[field]}")
if result["objective"] != "paired_matched_ce" or result["mode"] != "headline":
    raise RuntimeError("artifact is not the paired headline")
if result["world_size"] != 32 or result["sample"] != 1:
    raise RuntimeError("unexpected headline world size or sample")
if result["attention_backend"] != "flash_attention_2":
    raise RuntimeError("unexpected attention backend")

initial_topology = topology(result["initial_topology"])
initial_identities = result["initial_worker_identities"]
staging = result["staging"]
baseline = result["paired_baseline"]
if [row["rank"] for row in staging] != list(range(32)):
    raise RuntimeError("staging returned wrong rank order")
if [row["rank"] for row in baseline] != list(range(32)):
    raise RuntimeError("paired baseline returned wrong rank order")
staging_contract = all(row["batch_size"] == 128 for row in staging) and (
    sum(row["allocated_tokens"] for row in staging)
    == result["manifest_batch"]["counts"]["allocated_positions"]
)
baseline_hashes = [row["state_hash"] for row in baseline]
baseline_contract = all(
    row["gradient_tensors"] == 0
    and row["sparse_blocks"] == initial_identities[rank]["num_hidden_layers"]
    for rank, row in enumerate(baseline)
)
raw_by_name = {arm["arm"]: arm for arm in result["arms"]}
if set(raw_by_name) != {"eager", "grouped"} or len(result["arms"]) != 2:
    raise RuntimeError(f"unexpected headline arms: {sorted(raw_by_name)}")
if any(raw_by_name[name]["expert_implementation"] != name for name in raw_by_name):
    raise RuntimeError("headline arm selected the wrong expert implementation")
arm_checks = {
    name: arm_check(arm, initial_topology, initial_identities, baseline_hashes, staging)
    for name, arm in raw_by_name.items()
}

pair = compare_paired_numeric(raw_by_name["grouped"], raw_by_name["eager"])
semantic_pass = pair["representative_action_log_probs"]["passed"] and pair["matched_global_ce"]["passed"]
embedded = result["semantic_check"]
if embedded["grouped_versus_eager"] != pair:
    raise RuntimeError("independent grouped/eager recomputation differs from embedded result")
if semantic_pass != (embedded["verdict"] == "pass"):
    raise RuntimeError("independent semantic verdict differs from embedded verdict")

final_topology_equal_initial = topology(result["final_topology"]) == initial_topology
final_worker_identities_equal_initial = result["final_worker_identities"] == initial_identities
finish = result["paired_finish"]
finish_contract = all(
    row["rank"] == rank
    and row["gradient_tensors"] == 0
    and row["eager_blocks"] == row["sparse_blocks"]
    and row["state_hash_before"] == baseline_hashes[rank]
    and row["state_hash_after"] == baseline_hashes[rank]
    for rank, row in enumerate(finish)
)
correctness_pass = all(
    (
        topology_valid(result["initial_topology"]),
        final_topology_equal_initial,
        final_worker_identities_equal_initial,
        staging_contract,
        baseline_contract,
        finish_contract,
        semantic_pass,
        *(check["passed"] for check in arm_checks.values()),
    )
)

eager_metrics = raw_by_name["eager"]["metrics"]
grouped_metrics = raw_by_name["grouped"]["metrics"]
wall_speedup = eager_metrics["synchronized_wall_seconds"] / grouped_metrics["synchronized_wall_seconds"]
throughput_ratio = (
    grouped_metrics["nonpadding_tokens_per_second"] / eager_metrics["nonpadding_tokens_per_second"]
)
hbm_fit = all(
    arm["metrics"]["peak_allocated_bytes_max"]
    < min(identity["cuda_total_memory_bytes"] for identity in arm["worker_identities"])
    for arm in raw_by_name.values()
)
performance_pass = wall_speedup >= MINIMUM_SPEEDUP and throughput_ratio >= MINIMUM_SPEEDUP and hbm_fit
headline_valid = correctness_pass and performance_pass
summary = {
    "uri": uri,
    "payload_bytes": len(payload),
    "payload_sha256": payload_sha256,
    "result_sha256": claimed,
    "created_utc": result["created_utc"],
    "source_revision": result["source_revision"],
    "image": result["image"],
    "model_revision": result["model_revision"],
    "manifest_sha256": result["manifest_sha256"],
    "logical_batch_sha256": result["logical_batch_sha256"],
    "runtime_benchmark_sha256": result["runtime_benchmark_sha256"],
    "world_size": result["world_size"],
    "initial_topology": initial_topology,
    "final_topology_equal_initial": final_topology_equal_initial,
    "final_worker_identities_equal_initial": final_worker_identities_equal_initial,
    "staging_contract": staging_contract,
    "baseline_contract": baseline_contract,
    "finish_contract": finish_contract,
    "arm_checks": arm_checks,
    "grouped_versus_eager_recomputed": pair,
    "embedded_semantic_check_matches_recomputed": True,
    "semantic_pass": semantic_pass,
    "minimum_speedup": MINIMUM_SPEEDUP,
    "synchronized_wall_speedup": wall_speedup,
    "nonpadding_throughput_ratio": throughput_ratio,
    "hbm_fit": hbm_fit,
    "performance_pass": performance_pass,
    "correctness_pass": correctness_pass,
    "headline_valid": headline_valid,
    "eager_metrics": eager_metrics,
    "grouped_metrics": grouped_metrics,
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True), flush=True)
if not headline_valid:
    raise RuntimeError("headline semantic/performance gate failed; summary printed above")
