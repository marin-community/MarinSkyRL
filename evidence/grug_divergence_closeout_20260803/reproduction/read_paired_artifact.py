#!/usr/bin/env python3
"""Independently validate and summarize one paired Grug benchmark artifact."""

import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


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


uri, raw_path, summary_path = sys.argv[1:]
parsed = urlparse(uri)
payload = s3_client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
Path(raw_path).write_bytes(payload)
result = json.loads(payload)
claimed = result.pop("result_sha256")
actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if actual != claimed:
    raise RuntimeError(f"result digest mismatch: claimed={claimed} actual={actual}")
result["result_sha256"] = claimed
if result["objective"] != "paired_matched_ce" or result["mode"] != "headline":
    raise RuntimeError(
        f"expected paired headline artifact, got objective={result['objective']} mode={result['mode']}"
    )
if result["world_size"] != 32:
    raise RuntimeError(f"expected 32 workers, got {result['world_size']}")
semantic_pass = result["semantic_check"]["verdict"] == "pass"

initial_topology = topology(result["initial_topology"])
initial_identities = result["initial_worker_identities"]
arms = []
for arm in result["arms"]:
    workers = arm["timed_workers"]
    gradient_tensors = [worker["gradient_tensors"] for worker in workers]
    gradient_numel = [worker["gradient_numel"] for worker in workers]
    nonfinite = [worker["nonfinite_gradient_tensors"] for worker in workers]
    route_comparison = []
    for worker in workers:
        evidence = worker.get("expert_attribution")
        if evidence and evidence.get("paired_route_comparison"):
            route_comparison.extend(evidence["paired_route_comparison"])
    arms.append(
        {
            "arm": arm["arm"],
            "expert_implementation": arm["expert_implementation"],
            "topology": topology(arm["topology"]),
            "topology_equal_initial": topology(arm["topology"]) == initial_topology,
            "worker_identity_equal_initial_except_intervention": all(
                {
                    key: value
                    for key, value in current.items()
                    if key not in {"expert_implementation", "native_grouped_mm"}
                }
                == {
                    key: value
                    for key, value in baseline.items()
                    if key not in {"expert_implementation", "native_grouped_mm"}
                }
                for current, baseline in zip(arm["worker_identities"], initial_identities)
            ),
            "selection_restore_ok": all(
                row["gradient_tensors"] == 0 and row["cpu_rng_restored"] and row["cuda_rng_restored"]
                for row in arm["selection_restore"]
            ),
            "warmup_state_restored": all(
                row["state_hash_before"] == row["state_hash_after"] for row in arm["warmup_restore"]
            ),
            "gradient_tensors_min": min(gradient_tensors),
            "gradient_numel_min": min(gradient_numel),
            "nonfinite_gradient_tensors_sum": sum(nonfinite),
            "route_comparison_rows": len(route_comparison),
            "route_unexplained_changed_tokens_sum": sum(
                row["unexplained_changed_tokens"] for row in route_comparison
            ),
            "metrics": arm["metrics"],
        }
    )

by_name = {arm["arm"]: arm for arm in arms}
if set(by_name) != {"eager", "grouped"} or len(arms) != 2:
    raise RuntimeError(f"expected eager and grouped arms, got {[arm['arm'] for arm in arms]}")
if by_name["eager"]["expert_implementation"] != "eager":
    raise RuntimeError("eager arm did not select eager experts")
if by_name["grouped"]["expert_implementation"] != "grouped":
    raise RuntimeError("grouped arm did not select grouped experts")

final_topology_equal_initial = topology(result["final_topology"]) == initial_topology
final_worker_identities_equal_initial = result["final_worker_identities"] == initial_identities
baseline_empty_gradients = all(row["gradient_tensors"] == 0 for row in result["paired_baseline"])
finish_empty_gradients = all(row["gradient_tensors"] == 0 for row in result["paired_finish"])
finish_state_restored = all(
    row["state_hash_before"] == row["state_hash_after"] for row in result["paired_finish"]
)
arm_checks = {
    arm["arm"]: all(
        (
            arm["topology_equal_initial"],
            arm["worker_identity_equal_initial_except_intervention"],
            arm["selection_restore_ok"],
            arm["warmup_state_restored"],
            arm["gradient_tensors_min"] > 0,
            arm["gradient_numel_min"] > 0,
            arm["nonfinite_gradient_tensors_sum"] == 0,
        )
    )
    for arm in arms
}
correctness_pass = all(
    (
        final_topology_equal_initial,
        final_worker_identities_equal_initial,
        baseline_empty_gradients,
        finish_empty_gradients,
        finish_state_restored,
        *arm_checks.values(),
    )
)
if not correctness_pass:
    raise RuntimeError(
        "headline identity/state/gradient correctness failed: "
        f"final_topology={final_topology_equal_initial} "
        f"final_identities={final_worker_identities_equal_initial} "
        f"baseline_empty={baseline_empty_gradients} finish_empty={finish_empty_gradients} "
        f"finish_restored={finish_state_restored} arms={arm_checks}"
    )

eager_metrics = by_name["eager"]["metrics"]
grouped_metrics = by_name["grouped"]["metrics"]
eager_wall = eager_metrics["synchronized_wall_seconds"]
grouped_wall = grouped_metrics["synchronized_wall_seconds"]
world_size = result["world_size"]
comparison = {
    "eager_total_gpu_seconds": eager_wall * world_size,
    "grouped_total_gpu_seconds": grouped_wall * world_size,
    "total_gpu_seconds_saved": (eager_wall - grouped_wall) * world_size,
    "synchronized_wall_speedup": eager_wall / grouped_wall,
    "nonpadding_throughput_ratio": (
        grouped_metrics["nonpadding_tokens_per_second"]
        / eager_metrics["nonpadding_tokens_per_second"]
    ),
    "routed_expert_seconds_reduction": (
        eager_metrics["critical_rank_expert_seconds"]
        - grouped_metrics["critical_rank_expert_seconds"]
    ),
    "nonrouted_seconds_change": (
        grouped_metrics["critical_rank_nonexpert_seconds"]
        - eager_metrics["critical_rank_nonexpert_seconds"]
    ),
    "matched_global_ce_absolute_difference": abs(
        grouped_metrics["matched_global_ce_loss"] - eager_metrics["matched_global_ce_loss"]
    ),
}

summary = {
    "uri": uri,
    "payload_bytes": len(payload),
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
    "result_sha256": claimed,
    "schema_version": result["schema_version"],
    "created_utc": result["created_utc"],
    "benchmark": result["benchmark"],
    "objective": result["objective"],
    "mode": result["mode"],
    "source_revision": result["source_revision"],
    "image": result["image"],
    "model": result["model"],
    "model_revision": result["model_revision"],
    "manifest_sha256": result["manifest_sha256"],
    "logical_batch_sha256": result["logical_batch_sha256"],
    "world_size": result["world_size"],
    "runtime_benchmark_sha256": result["runtime_benchmark_sha256"],
    "initial_topology": initial_topology,
    "final_topology": topology(result["final_topology"]),
    "final_topology_equal_initial": final_topology_equal_initial,
    "final_worker_identities_equal_initial": final_worker_identities_equal_initial,
    "baseline_empty_gradients": baseline_empty_gradients,
    "finish_empty_gradients": finish_empty_gradients,
    "finish_state_restored": finish_state_restored,
    "arm_checks": arm_checks,
    "correctness_pass": correctness_pass,
    "semantic_pass": semantic_pass,
    "headline_valid": correctness_pass and semantic_pass,
    "semantic_check": result["semantic_check"],
    "timing_boundary": result["timing_boundary"],
    "arms": arms,
    "comparison": comparison,
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True), flush=True)
