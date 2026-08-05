#!/usr/bin/env python3
"""Independently verify the frozen 32-H100 route-discriminator artifact."""

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
    "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/"
    "route-residual-fbb1fc8/headline-discriminator-s1-rno-cpu8-mem768.json"
)
PREFLIGHT_URI = (
    "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/"
    "route-residual-fbb1fc8/preflight-discriminator-s1-rno-cpu8-mem768.json"
)
EXPECTED_PREFLIGHT_PAYLOAD_SHA256 = "4abc88af6961e84830f91f5156581b41ffa4f052058ed14702922beb59f89ec8"
EXPECTED_PREFLIGHT_RESULT_SHA256 = "ca157100c0efe469f979c7979e95b8580495e73418f1a290aa033d11f7148087"
EXPECTED = {
    "schema_version": 2,
    "objective": "paired_matched_ce",
    "mode": "headline",
    "sample": 1,
    "source_revision": "fbb1fc8378601e0346d00d186809f10d1ad0360d",
    "image": (
        "ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770"
    ),
    "model": "marin-community/grug-67b-a2b-sft-s2-thinking-step630",
    "model_revision": "a822321c2c21af099189e7116104b3cf5142c119",
    "attention_backend": "flash_attention_2",
    "world_size": 32,
    "manifest_s3_uri": (
        "s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/"
        "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json"
    ),
    "manifest_sha256": "5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74",
    "logical_batch_sha256": "e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d",
    "runtime_benchmark_sha256": "bbdc711b3d26b5127a71b3b8e24f7f3dfb5e00ba94e1a51f28f9bf83111dd084",
    "performance_claim": False,
}
EXPECTED_MODEL_SHA256 = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"
EXPECTED_WORKER_SHA256 = "d66c1c3ee148a8aef0007d1d3e17af4ef522381c0107f836d3c0817805fc0de4"
EXPECTED_ARMS = ["eager", "grouped"]
EXPECTED_ARM_MODES = {"eager": ("eager", "capture"), "grouped": ("grouped", "compare")}
ARCHIVED_TARGETS = [
    {"rank": 1, "microbatch": 28, "representative_position": "middle"},
    {"rank": 8, "microbatch": 114, "representative_position": "first"},
    {"rank": 16, "microbatch": 33, "representative_position": "middle"},
]
POSITION_INDEX = {"first": 0, "middle": 1, "last": 2}
POSITION_COORDINATES = {
    "first": (0, 1355),
    "middle": (3328, 4683),
    "last": (6655, 8010),
}
BOUNDARY_NAMES = (
    "decoder_input",
    "post_attention_hidden",
    "pre_router_hidden",
    "sparse_moe_output",
    "shared_expert_output",
    "decoder_output",
)
OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-3
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


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


def fetch(client, uri):
    parsed = urlparse(uri)
    return client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()


def verify_result_digest(payload):
    result = json.loads(payload)
    claimed = result.pop("result_sha256")
    actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(claimed == actual, f"result digest mismatch: claimed={claimed} actual={actual}")
    result["result_sha256"] = claimed
    return result, claimed


def topology_tuple(rows):
    return [(row["rank"], row["host"], row["phys_uuid"]) for row in rows]


def verify_topology(rows):
    normalized = topology_tuple(rows)
    require([row[0] for row in normalized] == list(range(32)), "topology rank order changed")
    require(len({row[2] for row in normalized}) == 32, "topology reused a GPU UUID")
    require(len({row[1] for row in normalized}) == 4, "headline did not use four hosts")
    require(
        all(len({row[1] for row in normalized[start : start + 8]}) == 1 for start in range(0, 32, 8)),
        "rank blocks are not eight GPUs per host",
    )


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
    output_violation_records = []
    gradient = empty_comparison()
    require(
        len(candidate["timed_workers"]) == len(reference["timed_workers"]),
        "paired numeric worker counts differ",
    )
    for candidate_worker, reference_worker in zip(candidate["timed_workers"], reference["timed_workers"], strict=True):
        require(candidate_worker["rank"] == reference_worker["rank"], "paired numeric worker ranks differ")
        candidate_output = candidate_worker["representative_action_log_probs"]
        reference_output = reference_worker["representative_action_log_probs"]
        candidate_coordinates = candidate_worker["representative_action_log_prob_coordinates"]
        reference_coordinates = reference_worker["representative_action_log_prob_coordinates"]
        require(len(candidate_output) == len(reference_output), "paired representative output counts differ")
        require(
            candidate_coordinates == reference_coordinates and len(reference_coordinates) == len(reference_output),
            "paired representative output coordinates differ",
        )
        for actual, expected, coordinate in zip(candidate_output, reference_output, reference_coordinates, strict=True):
            record_close(output, actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
            finite = math.isfinite(actual) and math.isfinite(expected)
            difference = abs(actual - expected) if finite else math.inf
            allowance = OUTPUT_ATOL + OUTPUT_RTOL * abs(expected)
            if not finite or difference > allowance:
                output_violation_records.append(
                    {
                        "rank": reference_worker["rank"],
                        **coordinate,
                        "eager": expected,
                        "grouped": actual,
                        "difference": difference,
                        "allowance": allowance,
                        "allowance_ratio": difference / allowance if finite else math.inf,
                    }
                )

        candidate_gradients = candidate_worker["representative_gradients"]
        reference_gradients = reference_worker["representative_gradients"]
        require(sorted(candidate_gradients) == sorted(reference_gradients), "gradient names differ")
        for name, candidate_gradient in candidate_gradients.items():
            reference_gradient = reference_gradients[name]
            require(
                candidate_gradient["local_numel"] == reference_gradient["local_numel"],
                f"gradient shard differs: {name}",
            )
            for field in ("l2_norm", "max_abs"):
                record_close(
                    gradient,
                    candidate_gradient[field],
                    reference_gradient[field],
                    rtol=GRADIENT_RTOL,
                    atol=GRADIENT_ATOL,
                )
            require(
                len(candidate_gradient["samples"]) == len(reference_gradient["samples"]),
                f"gradient sample count differs: {name}",
            )
            for actual, expected in zip(candidate_gradient["samples"], reference_gradient["samples"], strict=True):
                record_close(gradient, actual, expected, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)

    loss = empty_comparison()
    record_close(
        loss,
        candidate["metrics"]["matched_global_ce_loss"],
        reference["metrics"]["matched_global_ce_loss"],
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )
    for check in (output, loss, gradient):
        check["passed"] = check["violations"] == 0 and check["nonfinite"] == 0
    output["violation_records"] = output_violation_records
    return {
        "representative_action_log_probs": output,
        "matched_global_ce": loss,
        "representative_gradients": gradient,
    }


def normalized_identity(identity):
    return {key: value for key, value in identity.items() if key not in {"expert_implementation", "native_grouped_mm"}}


def verify_arm(arm, initial_topology, initial_identities, baseline_hashes, staging):
    arm_name = arm["arm"]
    implementation, discriminator_mode = EXPECTED_ARM_MODES[arm_name]
    require(arm["expert_implementation"] == implementation, f"{arm_name} selected wrong implementation")
    require(arm["expert_attribution_in_timed_sample"] is True, f"{arm_name} omitted attribution")
    require(arm["topology"] == initial_topology, f"{arm_name} changed topology")
    require([row["rank"] for row in arm["worker_identities"]] == list(range(32)), "identity order changed")
    for rank, identity in enumerate(arm["worker_identities"]):
        require(identity["expert_implementation"] == implementation, f"{arm_name} rank {rank} path changed")
        require(identity["runtime_grug_module_sha256"] == EXPECTED_MODEL_SHA256, "runtime model changed")
        require(identity["runtime_worker_sha256"] == EXPECTED_WORKER_SHA256, "runtime worker changed")
        require(
            normalized_identity(identity) == normalized_identity(initial_identities[rank]),
            f"{arm_name} rank {rank} identity changed",
        )
    require(
        all(
            row["rank"] == rank
            and row["implementation"] == implementation
            and row["selected_blocks"] == row["sparse_blocks"]
            and row["gradient_tensors"] == 0
            and row["cpu_rng_restored"]
            and row["cuda_rng_restored"]
            for rank, row in enumerate(arm["selection_restore"])
        ),
        f"{arm_name} selection/RNG restore failed",
    )
    require(
        all(
            row["rank"] == rank
            and row["state_hash_before"] == baseline_hashes[rank]
            and row["state_hash_after"] == baseline_hashes[rank]
            for rank, row in enumerate(arm["warmup_restore"])
        ),
        f"{arm_name} warmup changed state",
    )
    require([row["rank"] for row in arm["timed_workers"]] == list(range(32)), "timed rank order changed")
    for rank, (worker, identity, staged) in enumerate(
        zip(arm["timed_workers"], arm["worker_identities"], staging, strict=True)
    ):
        layers = identity["num_hidden_layers"]
        top_k = identity["num_experts_per_tok"]
        require(
            worker["microbatches"] == 128
            and worker["gradient_tensors"] > 0
            and worker["gradient_numel"] > 0
            and worker["nonfinite_gradient_tensors"] == 0,
            f"{arm_name} rank {rank} timed result failed",
        )
        evidence = worker["expert_attribution"]
        require(evidence is not None and evidence["module_count"] == layers, f"{arm_name} missed blocks")
        require(
            evidence["call_counts"] == {"forward": layers * 128, "backward": layers * 128}
            and evidence["route_calls_per_layer"] == [128] * layers
            and evidence["paired_route_mode"] is None
            and evidence["paired_route_reference_calls"] is None
            and evidence["paired_route_comparison"] is None,
            f"{arm_name} rank {rank} route-call contract failed",
        )
        expected_layer_load = staged["allocated_tokens"] * top_k
        require(
            len(evidence["route_loads_per_layer"]) == layers
            and all(
                len(loads) == identity["num_local_experts"] and sum(loads) == expected_layer_load
                for loads in evidence["route_loads_per_layer"]
            ),
            f"{arm_name} rank {rank} routed work changed",
        )
        capture = worker["route_discriminator"]
        expected_capture = {
            "mode": discriminator_mode,
            "layers": layers,
            "microbatches": 128,
            "samples": 384,
            "boundaries_per_sample": layers * len(BOUNDARY_NAMES) + 1,
            "selected_expert_ids_per_layer_sample": top_k,
            "adjusted_boundary_values_per_layer_sample": top_k + 1,
            "storage": "sampled rows hashed on CPU; exact selected IDs and top-k+1 adjusted values retained on host",
            "full_router_logits_retained": False,
        }
        require(capture == expected_capture, f"{arm_name} rank {rank} capture contract changed")


def target_key(item):
    return (int(item["rank"]), int(item["microbatch"]), str(item["representative_position"]))


def ordered_target_rows(rows):
    by_key = {target_key(row): row for row in rows}
    return [by_key[key] for key in sorted(by_key, key=lambda item: (item[0], item[1], POSITION_INDEX[item[2]]))]


def expected_coordinate(row):
    position = row["representative_position"]
    action_index, model_token_index = POSITION_COORDINATES[position]
    return {
        "rank": int(row["rank"]),
        "microbatch": int(row["microbatch"]),
        "representative_position": position,
        "worker_sample_index": int(row["microbatch"]) * 3 + POSITION_INDEX[position],
        "action_index": action_index,
        "model_token_index": model_token_index,
    }


def expected_boundary_order(layer_count):
    return [
        *(f"layer_{layer:02d}.{name}" for layer in range(layer_count) for name in BOUNDARY_NAMES),
        "model.final_hidden",
    ]


def verify_target(target, layer_count, top_k, semantic_violation):
    coordinate = target["coordinate"]
    require(coordinate == expected_coordinate(coordinate), f"bad target coordinate: {coordinate}")
    path = target["boundary_path"]
    require([row["boundary"] for row in path] == expected_boundary_order(layer_count), "boundary order changed")
    first_difference = None
    last_match = None
    for row in path:
        require(len(row["eager_sha256"]) == 64 and len(row["grouped_sha256"]) == 64, "bad boundary digest")
        exact = row["eager_sha256"] == row["grouped_sha256"]
        require(row["exact"] is exact, f"bad exact flag at {row['boundary']}")
        if first_difference is None:
            if exact:
                last_match = row["boundary"]
            else:
                first_difference = row["boundary"]
    require(target["first_differing_boundary"] == first_difference, "first differing boundary is wrong")
    require(target["last_matching_before_first_difference"] == last_match, "last matching boundary is wrong")

    routes = target["route_path"]
    require([row["layer"] for row in routes] == list(range(layer_count)), "route layer order changed")
    first_route_mismatch = None
    for route in routes:
        exact = route["eager_selected_experts"] == route["grouped_selected_experts"]
        require(route["selected_experts_exact"] is exact, "selected-expert exact flag is wrong")
        if not exact and first_route_mismatch is None:
            first_route_mismatch = route["layer"]
        for prefix in ("eager", "grouped"):
            selected = route[f"{prefix}_selected_experts"]
            boundary_ids = route[f"{prefix}_boundary_expert_ids"]
            boundary_values = route[f"{prefix}_adjusted_topk_plus_one"]
            require(len(selected) == top_k and len(set(selected)) == top_k, "selected expert IDs are incomplete")
            require(len(boundary_ids) == top_k + 1 and len(set(boundary_ids)) == top_k + 1, "boundary IDs incomplete")
            require(len(boundary_values) == top_k + 1, "boundary values incomplete")
            require(all(math.isfinite(value) for value in boundary_values), "nonfinite boundary value")
            require(
                all(left >= right for left, right in zip(boundary_values, boundary_values[1:])),
                "adjusted boundary values are not descending",
            )
    require(target["first_route_mismatch_layer"] == first_route_mismatch, "first route mismatch is wrong")
    require(target["semantic_violation"] == semantic_violation, "target semantic violation is wrong")
    return {
        "coordinate": coordinate,
        "last_matching_before_first_difference": last_match,
        "first_differing_boundary": first_difference,
        "first_route_mismatch_layer": first_route_mismatch,
        "semantic_violation": semantic_violation,
    }


def find_forbidden_router_logits(value, path="result"):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            require(key != "router_logits", f"full router logits retained at {child_path}")
            find_forbidden_router_logits(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            find_forbidden_router_logits(child, f"{path}[{index}]")


uri = sys.argv[1] if len(sys.argv) == 2 else EXPECTED_URI
require(uri == EXPECTED_URI, f"unexpected artifact URI: {uri}")
client = s3_client()

preflight_payload = fetch(client, PREFLIGHT_URI)
require(
    hashlib.sha256(preflight_payload).hexdigest() == EXPECTED_PREFLIGHT_PAYLOAD_SHA256,
    "preflight prerequisite payload changed",
)
preflight, preflight_result_sha256 = verify_result_digest(preflight_payload)
require(preflight_result_sha256 == EXPECTED_PREFLIGHT_RESULT_SHA256, "preflight result changed")
require(preflight["semantic_check"]["verdict"] == "pass", "preflight prerequisite did not pass")
require(preflight["route_discriminator"]["semantic_violation_count"] == 0, "preflight had action violations")
require(preflight["performance_claim"] is False, "preflight made a performance claim")

payload = fetch(client, uri)
payload_sha256 = hashlib.sha256(payload).hexdigest()
result, result_sha256 = verify_result_digest(payload)
for field, expected in EXPECTED.items():
    require(result.get(field) == expected, f"unexpected {field}: {result.get(field)!r}")
require(result["benchmark"] == "marinskyrl_grug_fixed_replay_paired_matched_ce", "wrong benchmark")
require([arm["arm"] for arm in result["arms"]] == EXPECTED_ARMS, "wrong arm order")

initial_topology = result["initial_topology"]
verify_topology(initial_topology)
require(result["final_topology"] == initial_topology, "final topology changed")
identities = result["initial_worker_identities"]
require([row["rank"] for row in identities] == list(range(32)), "initial identity order changed")
require(result["final_worker_identities"] == identities, "final worker identities changed")
require(
    all(
        row["runtime_grug_module_sha256"] == EXPECTED_MODEL_SHA256
        and row["runtime_worker_sha256"] == EXPECTED_WORKER_SHA256
        for row in identities
    ),
    "runtime model or worker source changed",
)
layer_counts = {row["num_hidden_layers"] for row in identities}
top_ks = {row["num_experts_per_tok"] for row in identities}
require(len(layer_counts) == 1 and len(top_ks) == 1, "worker model shapes disagree")
layer_count = layer_counts.pop()
top_k = top_ks.pop()

staging = result["staging"]
baseline = result["paired_baseline"]
finish = result["paired_finish"]
require([row["rank"] for row in staging] == list(range(32)), "staging rank order changed")
require([row["rank"] for row in baseline] == list(range(32)), "baseline rank order changed")
require([row["rank"] for row in finish] == list(range(32)), "finish rank order changed")
require(all(row["batch_size"] == 128 for row in staging), "rank-local batch size changed")
require(
    sum(row["allocated_tokens"] for row in staging) == result["manifest_batch"]["counts"]["allocated_positions"],
    "staged allocation count changed",
)
baseline_hashes = [row["state_hash"] for row in baseline]
require(
    all(
        row["gradient_tensors"] == 0 and row["sparse_blocks"] == identities[rank]["num_hidden_layers"]
        for rank, row in enumerate(baseline)
    ),
    "paired baseline was not clean",
)
require(
    all(
        row["gradient_tensors"] == 0
        and row["eager_blocks"] == row["sparse_blocks"]
        and row["state_hash_before"] == baseline_hashes[rank]
        and row["state_hash_after"] == baseline_hashes[rank]
        for rank, row in enumerate(finish)
    ),
    "paired finish did not restore clean state",
)

arms = {arm["arm"]: arm for arm in result["arms"]}
for arm in result["arms"]:
    verify_arm(arm, initial_topology, identities, baseline_hashes, staging)

pair = compare_paired_numeric(arms["grouped"], arms["eager"])
semantic_pass = pair["representative_action_log_probs"]["passed"] and pair["matched_global_ce"]["passed"]
expected_semantic = {
    "verdict": "pass" if semantic_pass else "fail",
    "kind": "headline_same_allocation_pair_check",
    "tolerances": {
        "representative_action_log_probs": {"rtol": OUTPUT_RTOL, "atol": OUTPUT_ATOL},
        "matched_global_ce": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
        "representative_gradients": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL},
    },
    "grouped_versus_eager": pair,
    "representative_gradients_are_observational": True,
    "route_contract": "preflight gate prerequisite; full headline routes are not retained",
}
require(result["semantic_check"] == expected_semantic, "embedded semantic check differs from recomputation")

violations = pair["representative_action_log_probs"]["violation_records"]
expected_requested = ordered_target_rows([*ARCHIVED_TARGETS, *violations])
expected_requested = [
    {key: row[key] for key in ("rank", "microbatch", "representative_position")} for row in expected_requested
]
discriminator = result["route_discriminator"]
require(discriminator["enabled"] and discriminator["performance_claim"] is False, "bad discriminator flags")
require(discriminator["capture_contract"].endswith("full router logits are never retained"), "capture contract changed")
require(discriminator["requested_targets"] == expected_requested, "requested target union/order changed")
require(discriminator["semantic_violation_count"] == len(violations), "violation count changed")
require(discriminator["semantic_violation_targets"] == violations, "violation target list changed")
require([row["rank"] for row in discriminator["workers"]] == list(range(32)), "worker order changed")
requested_per_rank = {rank: 0 for rank in range(32)}
for row in expected_requested:
    requested_per_rank[row["rank"]] += 1
for worker in discriminator["workers"]:
    require(worker["available_samples"] == 384, f"rank {worker['rank']} capture incomplete")
    require(
        worker["requested_targets"] == requested_per_rank[worker["rank"]],
        f"rank {worker['rank']} target count changed",
    )

targets = discriminator["targets"]
require(
    [target_key(row["coordinate"]) for row in targets] == [target_key(row) for row in expected_requested],
    "target order changed",
)
violations_by_key = {target_key(row): row for row in violations}
diagnoses = [
    verify_target(target, layer_count, top_k, violations_by_key.get(target_key(target["coordinate"])))
    for target in targets
]

find_forbidden_router_logits(preflight)
find_forbidden_router_logits(result)
summary = {
    "artifact_uri": uri,
    "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "payload_bytes": len(payload),
    "payload_sha256": payload_sha256,
    "result_sha256": result_sha256,
    "preflight_prerequisite": {
        "artifact_uri": PREFLIGHT_URI,
        "payload_sha256": EXPECTED_PREFLIGHT_PAYLOAD_SHA256,
        "result_sha256": preflight_result_sha256,
        "semantic_verdict": "pass",
    },
    "semantic_verdict": expected_semantic["verdict"],
    "embedded_semantic_check_matches_recomputed": True,
    "grouped_versus_eager_recomputed": pair,
    "semantic_violation_count": len(violations),
    "requested_targets": expected_requested,
    "target_diagnoses": diagnoses,
    "topology": topology_tuple(initial_topology),
    "layer_count": layer_count,
    "top_k": top_k,
    "capture_samples": 32 * 384,
    "performance_claim": False,
    "full_router_logits_retained": False,
}
print("GRUG_ROUTE_HEADLINE_READBACK=" + json.dumps(summary, sort_keys=True), flush=True)
