#!/usr/bin/env python3
"""Independently validate and summarize one paired Grug benchmark artifact."""

import hashlib
import json
import math
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


EXPECTED_URI = (
    "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/"
    "divergence-closeout-2dd905e/headline-paired-s1.json"
)
EXPECTED_PAYLOAD_SHA256 = "0f62a8123a280edaf6692ab1db0c01d82b644c8c46906316421f63cc22f2fa8a"
EXPECTED_RESULT_SHA256 = "2c1ef16927846e2ea031077064fd61b84e63bd707b9ec63904169096cb3fbe0c"
EXPECTED_PINS = {
    "source_revision": "2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2",
    "image": (
        "ghcr.io/marin-community/marinskyrl@"
        "sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2"
    ),
    "model": "marin-community/grug-67b-a2b-sft-s2-thinking-step630",
    "model_revision": "a822321c2c21af099189e7116104b3cf5142c119",
    "manifest_sha256": "5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74",
    "logical_batch_sha256": "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d",
    "runtime_benchmark_sha256": "b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404",
}
OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-3
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4


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


uri, raw_path, summary_path = sys.argv[1:]
if uri != EXPECTED_URI:
    raise RuntimeError(f"unexpected artifact URI: {uri}")
parsed = urlparse(uri)
payload = s3_client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
payload_sha256 = hashlib.sha256(payload).hexdigest()
if payload_sha256 != EXPECTED_PAYLOAD_SHA256:
    raise RuntimeError(
        f"payload digest mismatch: expected={EXPECTED_PAYLOAD_SHA256} actual={payload_sha256}"
    )
Path(raw_path).write_bytes(payload)
result = json.loads(payload)
claimed = result.pop("result_sha256")
actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if actual != claimed:
    raise RuntimeError(f"result digest mismatch: claimed={claimed} actual={actual}")
if claimed != EXPECTED_RESULT_SHA256:
    raise RuntimeError(f"unexpected result digest: expected={EXPECTED_RESULT_SHA256} actual={claimed}")
result["result_sha256"] = claimed
for field, expected in EXPECTED_PINS.items():
    if result[field] != expected:
        raise RuntimeError(f"unexpected {field}: expected={expected} actual={result[field]}")
if result["objective"] != "paired_matched_ce" or result["mode"] != "headline":
    raise RuntimeError(
        f"expected paired headline artifact, got objective={result['objective']} mode={result['mode']}"
    )
if result["world_size"] != 32:
    raise RuntimeError(f"expected 32 workers, got {result['world_size']}")
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
raw_by_name = {arm["arm"]: arm for arm in result["arms"]}
if set(by_name) != {"eager", "grouped"} or len(arms) != 2:
    raise RuntimeError(f"expected eager and grouped arms, got {[arm['arm'] for arm in arms]}")
if by_name["eager"]["expert_implementation"] != "eager":
    raise RuntimeError("eager arm did not select eager experts")
if by_name["grouped"]["expert_implementation"] != "grouped":
    raise RuntimeError("grouped arm did not select grouped experts")

semantic_recomputed = compare_paired_numeric(raw_by_name["grouped"], raw_by_name["eager"])
embedded_numeric = result["semantic_check"]["grouped_versus_eager"]
if semantic_recomputed != embedded_numeric:
    raise RuntimeError("independent numeric semantic recomputation differs from embedded verdict")
semantic_pass = (
    semantic_recomputed["representative_action_log_probs"]["passed"]
    and semantic_recomputed["matched_global_ce"]["passed"]
)
if semantic_pass != (result["semantic_check"]["verdict"] == "pass"):
    raise RuntimeError("independent semantic verdict differs from embedded verdict")

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
    "matched_global_ce_absolute_difference": abs(
        grouped_metrics["matched_global_ce_loss"] - eager_metrics["matched_global_ce_loss"]
    ),
}

summary = {
    "uri": uri,
    "payload_bytes": len(payload),
    "payload_sha256": payload_sha256,
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
    "semantic_recomputed": semantic_recomputed,
    "embedded_semantic_check_matches_recomputed": True,
    "timing_boundary": result["timing_boundary"],
    "arms": arms,
    "comparison": comparison,
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True), flush=True)
