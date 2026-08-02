#!/usr/bin/env bash
set -euo pipefail

PY=/opt/openthoughts/envs/rl/bin/python
RESULT_ROOT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260802/paired-d81f636/preflight
OUT=/tmp/grug-preflight-failure-readback.json

"$PY" - "$RESULT_ROOT" "$OUT" <<'PY'
import copy
import hashlib
import json
import math
import os
import sys
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def client():
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


def load(uri):
    parsed = urlparse(uri)
    payload = client().get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
    result = json.loads(payload)
    claimed = result.pop("result_sha256")
    actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != claimed:
        raise RuntimeError(f"digest mismatch: {uri}")
    result["result_sha256"] = claimed
    return result, hashlib.sha256(payload).hexdigest()


def topology(result):
    return [(item["rank"], item["host"], item["phys_uuid"]) for item in result["topology"]]


def topology_hardware(result):
    return sorted((item["host"], item["phys_uuid"]) for item in result["topology"])


def normalized_config(result):
    value = copy.deepcopy(result["config"])
    value["trainer"]["policy"]["fsdp_config"]["use_grouped_mm"] = "paired-intervention"
    return value


def common_identity(result):
    keys = (
        "schema_version", "objective", "mode", "sample", "source_revision", "image",
        "model", "model_revision", "attention_backend", "world_size", "manifest_s3_uri",
        "manifest_sha256", "logical_batch_sha256", "manifest_batch",
        "manifest_batch_metadata", "runtime_benchmark_sha256",
    )
    return {key: result[key] for key in keys}


def worker_common(worker):
    return {key: value for key, value in worker.items() if key not in {"expert_implementation", "native_grouped_mm"}}


def close(actual, expected, rtol, atol):
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= atol + rtol * abs(expected)


def numeric(candidate, reference):
    output_diffs = []
    gradient_diffs = []
    outputs_pass = True
    gradients_pass = True
    for candidate_worker, reference_worker in zip(candidate["timed_workers"], reference["timed_workers"]):
        for actual, expected in zip(
            candidate_worker["representative_action_log_probs"],
            reference_worker["representative_action_log_probs"],
        ):
            output_diffs.append(abs(actual - expected))
            outputs_pass &= close(actual, expected, 4e-2, 4e-3)
        for name, candidate_gradient in candidate_worker["representative_gradients"].items():
            reference_gradient = reference_worker["representative_gradients"][name]
            for field in ("l2_norm", "max_abs"):
                actual = candidate_gradient[field]
                expected = reference_gradient[field]
                gradient_diffs.append(abs(actual - expected))
                gradients_pass &= close(actual, expected, 8e-2, 1e-4)
            for actual, expected in zip(candidate_gradient["samples"], reference_gradient["samples"]):
                gradient_diffs.append(abs(actual - expected))
                gradients_pass &= close(actual, expected, 8e-2, 1e-4)
    actual_loss = candidate["metrics"]["matched_global_ce_loss"]
    expected_loss = reference["metrics"]["matched_global_ce_loss"]
    return {
        "outputs_pass": outputs_pass,
        "max_output_abs_difference": max(output_diffs, default=0.0),
        "gradients_pass": gradients_pass,
        "max_gradient_abs_difference": max(gradient_diffs, default=0.0),
        "loss_pass": close(actual_loss, expected_loss, 2e-3, 2e-3),
        "loss_abs_difference": abs(actual_loss - expected_loss),
    }


def route_detail(candidate, reference):
    per_rank = []
    global_candidate = None
    global_reference = None
    for candidate_worker, reference_worker in zip(candidate["timed_workers"], reference["timed_workers"]):
        candidate_layers = candidate_worker["expert_attribution"]["route_loads_per_layer"]
        reference_layers = reference_worker["expert_attribution"]["route_loads_per_layer"]
        mismatch_cells = 0
        max_abs = 0
        for candidate_layer, reference_layer in zip(candidate_layers, reference_layers):
            for actual, expected in zip(candidate_layer, reference_layer):
                mismatch_cells += actual != expected
                max_abs = max(max_abs, abs(actual - expected))
        if global_candidate is None:
            global_candidate = copy.deepcopy(candidate_layers)
            global_reference = copy.deepcopy(reference_layers)
        else:
            for layer_index, (candidate_layer, reference_layer) in enumerate(
                zip(candidate_layers, reference_layers)
            ):
                for expert_index, (actual, expected) in enumerate(zip(candidate_layer, reference_layer)):
                    global_candidate[layer_index][expert_index] += actual
                    global_reference[layer_index][expert_index] += expected
        per_rank.append(
            {
                "rank": candidate_worker["rank"],
                "mismatch_cells": mismatch_cells,
                "max_abs_load_difference": max_abs,
                "candidate_total": sum(sum(layer) for layer in candidate_layers),
                "reference_total": sum(sum(layer) for layer in reference_layers),
            }
        )
    global_mismatch_cells = 0
    global_max_abs = 0
    for candidate_layer, reference_layer in zip(global_candidate, global_reference):
        for actual, expected in zip(candidate_layer, reference_layer):
            global_mismatch_cells += actual != expected
            global_max_abs = max(global_max_abs, abs(actual - expected))
    return {
        "per_rank": per_rank,
        "global_mismatch_cells": global_mismatch_cells,
        "global_max_abs_load_difference": global_max_abs,
        "global_candidate_total": sum(sum(layer) for layer in global_candidate),
        "global_reference_total": sum(sum(layer) for layer in global_reference),
    }


def gradient_violations(candidate, reference):
    violations = []
    checked = 0
    for candidate_worker, reference_worker in zip(candidate["timed_workers"], reference["timed_workers"]):
        for name, candidate_gradient in candidate_worker["representative_gradients"].items():
            reference_gradient = reference_worker["representative_gradients"][name]
            values = [("l2_norm", candidate_gradient["l2_norm"], reference_gradient["l2_norm"])]
            values.append(("max_abs", candidate_gradient["max_abs"], reference_gradient["max_abs"]))
            values.extend(
                (f"sample_{index}", actual, expected)
                for index, (actual, expected) in enumerate(
                    zip(candidate_gradient["samples"], reference_gradient["samples"])
                )
            )
            for field, actual, expected in values:
                checked += 1
                allowed = 1e-4 + 8e-2 * abs(expected)
                difference = abs(actual - expected)
                if difference > allowed:
                    violations.append(
                        {
                            "rank": candidate_worker["rank"],
                            "name": name,
                            "field": field,
                            "candidate": actual,
                            "reference": expected,
                            "abs_difference": difference,
                            "allowed_difference": allowed,
                            "excess_ratio": difference / allowed,
                        }
                    )
    return {
        "checked": checked,
        "violation_count": len(violations),
        "worst": sorted(violations, key=lambda item: item["excess_ratio"], reverse=True)[:12],
    }


root, out_path = sys.argv[1:]
oracle, oracle_payload = load(f"{root}/eager-oracle-s1.json")
eager, eager_payload = load(f"{root}/eager-instrumented-s1.json")
grouped, grouped_payload = load(f"{root}/grouped-instrumented-s1.json")

report = {
    "verdict": "fail",
    "failure": "paired topology differs",
    "result_sha256": {
        "oracle": oracle["result_sha256"],
        "eager": eager["result_sha256"],
        "grouped": grouped["result_sha256"],
    },
    "payload_sha256": {"oracle": oracle_payload, "eager": eager_payload, "grouped": grouped_payload},
    "topology": {"oracle": topology(oracle), "eager": topology(eager), "grouped": topology(grouped)},
    "same_hardware_set": {
        "oracle_eager": topology_hardware(oracle) == topology_hardware(eager),
        "eager_grouped": topology_hardware(eager) == topology_hardware(grouped),
    },
    "rank_topology_equal": {
        "oracle_eager": topology(oracle) == topology(eager),
        "eager_grouped": topology(eager) == topology(grouped),
    },
    "common_identity_equal": common_identity(eager) == common_identity(grouped),
    "normalized_config_equal": normalized_config(eager) == normalized_config(grouped),
    "staging_equal": eager["staging"] == grouped["staging"],
    "warmup_equal": eager["warmup_restore"] == grouped["warmup_restore"],
    "worker_common_equal": [
        worker_common(a) == worker_common(b)
        for a, b in zip(eager["worker_identities"], grouped["worker_identities"])
    ],
    "routes_exact": [
        a["expert_attribution"]["route_loads_per_layer"]
        == b["expert_attribution"]["route_loads_per_layer"]
        for a, b in zip(eager["timed_workers"], grouped["timed_workers"])
    ],
    "route_detail": route_detail(grouped, eager),
    "gradient_violations": gradient_violations(grouped, eager),
    "grouped_vs_eager_numeric": numeric(grouped, eager),
    "instrumented_vs_oracle_numeric": numeric(eager, oracle),
    "metrics": {
        key: {
            "wall_seconds": value["metrics"]["synchronized_wall_seconds"],
            "global_ce_loss": value["metrics"]["matched_global_ce_loss"],
            "gradient_tensors_min": value["metrics"]["matched_gradient_tensors_min"],
            "gradient_numel_min": value["metrics"]["matched_gradient_numel_min"],
            "routed_block_seconds": value["metrics"].get("critical_rank_expert_seconds"),
            "nonexpert_seconds": value["metrics"].get("critical_rank_nonexpert_seconds"),
        }
        for key, value in (("oracle", oracle), ("eager", eager), ("grouped", grouped))
    },
}
with open(out_path, "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(report, sort_keys=True), flush=True)
PY
sha256sum "$OUT"
