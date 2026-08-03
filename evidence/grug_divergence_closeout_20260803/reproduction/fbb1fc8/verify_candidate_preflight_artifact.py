#!/usr/bin/env python3
"""Independently verify the pinned candidate's eight-H100 preflight artifact."""

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
    "divergence-closeout-fbb1fc8/preflight-paired-s1.json"
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
EXPECTED_LOCALIZATION = {
    "result_s3_uri": (
        "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/"
        "divergence-closeout-53d420e/localization-s3.json"
    ),
    "payload_sha256": "976566dc1aa8a882db12326fa84be3d30dbeb2d20565933fbeb7265530e25a3f",
    "result_sha256": "d890dc2cb8938b6607b4a98a60cfc9476e187695dbc48f6dba599c8a15cb9782",
    "route_dependent_gradient_check": (
        "all eight layer-zero selected-expert probes are bitwise exact for output, input gradient, "
        "and gate/up/down weight gradients"
    ),
}
EXPECTED_MOE_SHA256 = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"
EXPECTED_WORKER_SHA256 = "c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1"
EXPECTED_ARMS = {
    "eager_oracle": ("eager", False, None),
    "eager_instrumented": ("eager", True, "capture"),
    "grouped_instrumented": ("grouped", True, "compare"),
}
OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-3
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4


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


def summarize_route_contract(grouped_arm):
    summary = {
        "rank_layers": 0,
        "tokens": 0,
        "routed_allocations": 0,
        "changed_tokens": 0,
        "changed_routed_allocations": 0,
        "ordered_slot_mismatches": 0,
        "unexplained_changed_tokens": 0,
        "max_token_logit_delta": 0.0,
        "max_changed_reference_margin": 0.0,
        "max_changed_current_margin": 0.0,
    }
    for worker in grouped_arm["timed_workers"]:
        comparisons = worker["expert_attribution"]["paired_route_comparison"]
        if not comparisons:
            raise RuntimeError(f"rank {worker['rank']} returned no paired route comparisons")
        for comparison in comparisons:
            summary["rank_layers"] += 1
            for key in (
                "tokens",
                "routed_allocations",
                "changed_tokens",
                "changed_routed_allocations",
                "ordered_slot_mismatches",
                "unexplained_changed_tokens",
            ):
                summary[key] += comparison[key]
            for key in (
                "max_token_logit_delta",
                "max_changed_reference_margin",
                "max_changed_current_margin",
            ):
                summary[key] = max(summary[key], comparison[key])
    summary["changed_token_fraction"] = summary["changed_tokens"] / summary["tokens"]
    summary["changed_routed_allocation_fraction"] = (
        summary["changed_routed_allocations"] / summary["routed_allocations"]
    )
    summary["passed"] = summary["unexplained_changed_tokens"] == 0
    summary["acceptance_rule"] = (
        "for every token/call/layer, changed selected-expert membership is explained only when the eager "
        "kth-vs-(k+1)th adjusted-logit margin is <= 2 * the maximum per-token adjusted-logit delta"
    )
    return summary


def arm_structural_check(arm, initial_topology, initial_identities, baseline_hashes, staging):
    workers = arm["timed_workers"]
    implementation, attributed, route_mode = EXPECTED_ARMS[arm["arm"]]
    rank_order = [worker["rank"] for worker in workers] == list(range(8))
    identities = arm["worker_identities"]
    selection = arm["selection_restore"]
    warmup = arm["warmup_restore"]
    identity_contract = all(
        identity["rank"] == rank
        and identity["expert_implementation"] == implementation
        and identity["runtime_grug_module_sha256"] == EXPECTED_MOE_SHA256
        and identity["runtime_worker_sha256"] == EXPECTED_WORKER_SHA256
        for rank, identity in enumerate(identities)
    )
    selection_restore_ok = all(
        row["rank"] == rank
        and row["implementation"] == implementation
        and row["selected_blocks"] == row["sparse_blocks"]
        and row["gradient_tensors"] == 0
        and row["cpu_rng_restored"]
        and row["cuda_rng_restored"]
        for rank, row in enumerate(selection)
    )
    warmup_state_restored = all(
        row["rank"] == rank
        and row["state_hash_before"] == baseline_hashes[rank]
        and row["state_hash_after"] == baseline_hashes[rank]
        for rank, row in enumerate(warmup)
    )
    timed_contract = True
    for rank, (worker, identity, staged) in enumerate(zip(workers, identities, staging, strict=True)):
        timed_contract &= worker["rank"] == rank and worker["microbatches"] == 1
        evidence = worker["expert_attribution"]
        if not attributed:
            timed_contract &= evidence is None
            continue
        expected_blocks = identity["num_hidden_layers"]
        expected_calls = expected_blocks
        expected_layer_load = staged["allocated_tokens"] * identity["num_experts_per_tok"]
        timed_contract &= (
            evidence is not None
            and evidence["module_count"] == expected_blocks
            and evidence["call_counts"] == {"forward": expected_calls, "backward": expected_calls}
            and evidence["route_calls_per_layer"] == [1] * expected_blocks
            and evidence["paired_route_mode"] == route_mode
            and evidence["paired_route_reference_calls"] == [1] * expected_blocks
            and len(evidence["route_loads_per_layer"]) == expected_blocks
            and all(
                len(loads) == identity["num_local_experts"] and sum(loads) == expected_layer_load
                for loads in evidence["route_loads_per_layer"]
            )
        )
        if route_mode == "capture":
            timed_contract &= evidence["paired_route_comparison"] is None
        else:
            timed_contract &= len(evidence["paired_route_comparison"] or []) == expected_blocks
    return {
        "topology_equal_initial": topology(arm["topology"]) == initial_topology,
        "worker_identity_equal_initial_except_intervention": all(
            {k: v for k, v in current.items() if k not in {"expert_implementation", "native_grouped_mm"}}
            == {k: v for k, v in baseline.items() if k not in {"expert_implementation", "native_grouped_mm"}}
            for current, baseline in zip(arm["worker_identities"], initial_identities, strict=True)
        ),
        "identity_contract": identity_contract,
        "selection_restore_ok": selection_restore_ok,
        "warmup_state_restored": warmup_state_restored,
        "timed_route_and_allocation_contract": timed_contract,
        "rank_order": rank_order,
        "gradient_tensors_min": min(worker["gradient_tensors"] for worker in workers),
        "gradient_numel_min": min(worker["gradient_numel"] for worker in workers),
        "nonfinite_gradient_tensors_sum": sum(
            worker["nonfinite_gradient_tensors"] for worker in workers
        ),
    }


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
if result["objective"] != "paired_matched_ce" or result["mode"] != "preflight":
    raise RuntimeError("artifact is not the paired preflight")
if result["world_size"] != 8 or result["sample"] != 1:
    raise RuntimeError("unexpected preflight world size or sample")
if result["attention_backend"] != "flash_attention_2":
    raise RuntimeError("unexpected attention backend")

initial_topology = topology(result["initial_topology"])
initial_identities = result["initial_worker_identities"]
if [row["rank"] for row in result["staging"]] != list(range(8)):
    raise RuntimeError("staging returned wrong rank order")
if [row["rank"] for row in result["paired_baseline"]] != list(range(8)):
    raise RuntimeError("paired baseline returned wrong rank order")
baseline_hashes = [row["state_hash"] for row in result["paired_baseline"]]
raw_by_name = {arm["arm"]: arm for arm in result["arms"]}
if set(raw_by_name) != set(EXPECTED_ARMS) or len(result["arms"]) != 3:
    raise RuntimeError(f"unexpected preflight arms: {sorted(raw_by_name)}")
if raw_by_name["eager_oracle"]["expert_implementation"] != "eager":
    raise RuntimeError("eager oracle did not use eager experts")
if raw_by_name["eager_instrumented"]["expert_implementation"] != "eager":
    raise RuntimeError("instrumented oracle did not use eager experts")
if raw_by_name["grouped_instrumented"]["expert_implementation"] != "grouped":
    raise RuntimeError("grouped arm did not use grouped experts")

arm_checks = {
    name: arm_structural_check(
        arm,
        initial_topology,
        initial_identities,
        baseline_hashes,
        result["staging"],
    )
    for name, arm in raw_by_name.items()
}
arm_passes = {
    name: all(
        (
            check["topology_equal_initial"],
            check["worker_identity_equal_initial_except_intervention"],
            check["identity_contract"],
            check["selection_restore_ok"],
            check["warmup_state_restored"],
            check["timed_route_and_allocation_contract"],
            check["rank_order"],
            check["gradient_tensors_min"] > 0,
            check["gradient_numel_min"] > 0,
            check["nonfinite_gradient_tensors_sum"] == 0,
        )
    )
    for name, check in arm_checks.items()
}

oracle = compare_paired_numeric(raw_by_name["eager_instrumented"], raw_by_name["eager_oracle"])
grouped = compare_paired_numeric(
    raw_by_name["grouped_instrumented"], raw_by_name["eager_instrumented"]
)
routes = summarize_route_contract(raw_by_name["grouped_instrumented"])
embedded = result["semantic_check"]
if embedded["eager_instrumentation_oracle"] != oracle:
    raise RuntimeError("independent eager-oracle recomputation differs from embedded result")
if embedded["grouped_versus_eager"] != grouped:
    raise RuntimeError("independent grouped/eager recomputation differs from embedded result")
if embedded["route_contract"] != routes:
    raise RuntimeError("independent route-contract recomputation differs from embedded result")
if embedded["causal_localization_prerequisite"] != EXPECTED_LOCALIZATION:
    raise RuntimeError("unexpected causal-localization prerequisite")
semantic_pass = all(
    (
        oracle["representative_action_log_probs"]["passed"],
        oracle["matched_global_ce"]["passed"],
        oracle["representative_gradients"]["passed"],
        grouped["representative_action_log_probs"]["passed"],
        grouped["matched_global_ce"]["passed"],
        routes["passed"],
    )
)
if semantic_pass != (embedded["verdict"] == "pass"):
    raise RuntimeError("independent semantic verdict differs from embedded verdict")

final_topology_equal_initial = topology(result["final_topology"]) == initial_topology
final_worker_identities_equal_initial = result["final_worker_identities"] == initial_identities
baseline_empty_gradients = all(
    row["rank"] == rank
    and row["gradient_tensors"] == 0
    and row["sparse_blocks"] == initial_identities[rank]["num_hidden_layers"]
    for rank, row in enumerate(result["paired_baseline"])
)
finish_empty_gradients = all(
    row["rank"] == rank
    and row["gradient_tensors"] == 0
    and row["eager_blocks"] == row["sparse_blocks"]
    for rank, row in enumerate(result["paired_finish"])
)
finish_state_restored = all(
    row["state_hash_before"] == baseline_hashes[rank]
    and row["state_hash_after"] == baseline_hashes[rank]
    for rank, row in enumerate(result["paired_finish"])
)
correctness_pass = all(
    (
        final_topology_equal_initial,
        final_worker_identities_equal_initial,
        baseline_empty_gradients,
        finish_empty_gradients,
        finish_state_restored,
        semantic_pass,
        *arm_passes.values(),
    )
)

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
    "baseline_empty_gradients": baseline_empty_gradients,
    "finish_empty_gradients": finish_empty_gradients,
    "finish_state_restored": finish_state_restored,
    "arm_checks": arm_checks,
    "arm_passes": arm_passes,
    "eager_instrumentation_oracle_recomputed": oracle,
    "grouped_versus_eager_recomputed": grouped,
    "route_contract_recomputed": routes,
    "embedded_semantic_check_matches_recomputed": True,
    "semantic_pass": semantic_pass,
    "preflight_valid": correctness_pass,
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True), flush=True)
if not correctness_pass:
    raise RuntimeError("preflight final identity/state/semantic check failed; summary printed above")
