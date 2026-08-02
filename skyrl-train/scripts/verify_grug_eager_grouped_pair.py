#!/usr/bin/env python3
"""Verify a fixed-replay Grug eager/native-grouped result pair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config

OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-3
LOSS_RTOL = 2e-3
LOSS_ATOL = 2e-3
GRADIENT_RTOL = 8e-2
GRADIENT_ATOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-result", required=True)
    parser.add_argument("--grouped-result", required=True)
    parser.add_argument("--instrumentation-oracle-result")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_bytes(location: str) -> bytes:
    parsed = urlparse(location)
    if parsed.scheme == "s3":
        kwargs: dict[str, Any] = {
            "config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        }
        endpoint = os.environ.get("CW_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
        access_key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key
        return (
            boto3.client("s3", **kwargs)
            .get_object(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip("/"),
            )["Body"]
            .read()
        )
    if parsed.scheme:
        raise ValueError(f"unsupported result location: {location}")
    return Path(location).read_bytes()


def load_result(location: str) -> tuple[dict[str, Any], str]:
    payload = _read_bytes(location)
    result = json.loads(payload)
    claimed = result.pop("result_sha256")
    unsigned = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(unsigned).hexdigest()
    if actual != claimed:
        raise RuntimeError(f"{location}: result digest mismatch: claimed={claimed}, actual={actual}")
    result["result_sha256"] = claimed
    return result, hashlib.sha256(payload).hexdigest()


def require_equal(label: str, left: Any, right: Any) -> None:
    if left != right:
        raise RuntimeError(f"{label} differs")


def require_close(label: str, actual: float, expected: float, *, rtol: float, atol: float) -> float:
    """Require two finite values to be close and return their absolute difference."""

    if not math.isfinite(actual) or not math.isfinite(expected):
        raise RuntimeError(f"{label} is non-finite: actual={actual}, expected={expected}")
    difference = abs(actual - expected)
    if difference > atol + rtol * abs(expected):
        raise RuntimeError(
            f"{label} differs: actual={actual:.9g}, expected={expected:.9g}, "
            f"abs={difference:.9g}, rtol={rtol}, atol={atol}"
        )
    return difference


def _normalized_config(result: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(result["config"])
    config["trainer"]["policy"]["fsdp_config"]["use_grouped_mm"] = "paired-intervention"
    return config


def _common_identity(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "schema_version",
            "objective",
            "mode",
            "sample",
            "source_revision",
            "image",
            "model",
            "model_revision",
            "attention_backend",
            "world_size",
            "manifest_s3_uri",
            "manifest_sha256",
            "logical_batch_sha256",
            "manifest_batch",
            "manifest_batch_metadata",
            "runtime_benchmark_sha256",
        )
    }


def _topology_identity(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(item["rank"], item["host"], item["phys_uuid"]) for item in result["topology"]]


def _worker_common_identity(worker: dict[str, Any]) -> dict[str, Any]:
    ignored = {"expert_implementation", "native_grouped_mm"}
    return {key: value for key, value in worker.items() if key not in ignored}


def _check_result_shape(result: dict[str, Any], expected_implementation: str, *, attribution: bool) -> None:
    if result["objective"] != "matched_ce":
        raise RuntimeError("paired verifier accepts only matched_ce results")
    if result["expert_implementation"] != expected_implementation:
        raise RuntimeError(f"result did not select {expected_implementation}")
    if result["profile_in_timed_sample"] or not result["headline_eligible"]:
        raise RuntimeError("profiled results are not pair-eligible")
    if bool(result["expert_attribution_in_timed_sample"]) != attribution:
        raise RuntimeError("result attribution mode differs from verifier expectation")
    require_equal("worker identity count", len(result["timed_workers"]), len(result["worker_identities"]))
    for worker, identity in zip(result["timed_workers"], result["worker_identities"]):
        expected_microbatches = identity["mini_batch_size_per_gpu"]
        if worker["microbatches"] != expected_microbatches:
            raise RuntimeError("worker completed the wrong number of microbatches")
        if worker["gradient_tensors"] <= 0 or worker["gradient_numel"] <= 0:
            raise RuntimeError("worker returned no gradients")
        if worker["nonfinite_gradient_tensors"] != 0:
            raise RuntimeError("worker returned non-finite gradients")
        if attribution:
            evidence = worker["expert_attribution"]
            expected_blocks = identity["num_hidden_layers"]
            expected_calls = expected_blocks * expected_microbatches
            require_equal("routed-block module count", evidence["module_count"], expected_blocks)
            require_equal(
                "routed-block calls",
                evidence["call_counts"],
                {"forward": expected_calls, "backward": expected_calls},
            )
            require_equal(
                "route calls",
                evidence["route_calls_per_layer"],
                [expected_microbatches] * expected_blocks,
            )


def _compare_numeric_evidence(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    max_output_difference = 0.0
    max_gradient_difference = 0.0
    require_equal(f"{label} worker count", len(candidate["timed_workers"]), len(reference["timed_workers"]))
    for candidate_worker, reference_worker in zip(candidate["timed_workers"], reference["timed_workers"]):
        require_equal(f"{label} rank", candidate_worker["rank"], reference_worker["rank"])
        require_equal(
            f"{label} output sample count",
            len(candidate_worker["representative_action_log_probs"]),
            len(reference_worker["representative_action_log_probs"]),
        )
        for index, (actual, expected) in enumerate(
            zip(
                candidate_worker["representative_action_log_probs"],
                reference_worker["representative_action_log_probs"],
            )
        ):
            max_output_difference = max(
                max_output_difference,
                require_close(
                    f"{label} rank {candidate_worker['rank']} output {index}",
                    actual,
                    expected,
                    rtol=OUTPUT_RTOL,
                    atol=OUTPUT_ATOL,
                ),
            )
        require_equal(
            f"{label} gradient names",
            sorted(candidate_worker["representative_gradients"]),
            sorted(reference_worker["representative_gradients"]),
        )
        for name, candidate_gradient in candidate_worker["representative_gradients"].items():
            reference_gradient = reference_worker["representative_gradients"][name]
            require_equal(
                f"{label} rank {candidate_worker['rank']} {name} local numel",
                candidate_gradient["local_numel"],
                reference_gradient["local_numel"],
            )
            for field in ("l2_norm", "max_abs"):
                max_gradient_difference = max(
                    max_gradient_difference,
                    require_close(
                        f"{label} rank {candidate_worker['rank']} {name} {field}",
                        candidate_gradient[field],
                        reference_gradient[field],
                        rtol=GRADIENT_RTOL,
                        atol=GRADIENT_ATOL,
                    ),
                )
            require_equal(
                f"{label} rank {candidate_worker['rank']} {name} sample count",
                len(candidate_gradient["samples"]),
                len(reference_gradient["samples"]),
            )
            for index, (actual, expected) in enumerate(
                zip(candidate_gradient["samples"], reference_gradient["samples"])
            ):
                max_gradient_difference = max(
                    max_gradient_difference,
                    require_close(
                        f"{label} rank {candidate_worker['rank']} {name} sample {index}",
                        actual,
                        expected,
                        rtol=GRADIENT_RTOL,
                        atol=GRADIENT_ATOL,
                    ),
                )
    loss_difference = require_close(
        f"{label} global CE loss",
        candidate["metrics"]["matched_global_ce_loss"],
        reference["metrics"]["matched_global_ce_loss"],
        rtol=LOSS_RTOL,
        atol=LOSS_ATOL,
    )
    return {
        "max_representative_output_abs_difference": max_output_difference,
        "max_representative_gradient_abs_difference": max_gradient_difference,
        "global_ce_loss_abs_difference": loss_difference,
    }


def verify_pair(eager: dict[str, Any], grouped: dict[str, Any]) -> dict[str, Any]:
    _check_result_shape(eager, "eager", attribution=True)
    _check_result_shape(grouped, "grouped", attribution=True)
    require_equal("common result identity", _common_identity(eager), _common_identity(grouped))
    require_equal("paired config", _normalized_config(eager), _normalized_config(grouped))
    require_equal("paired topology", _topology_identity(eager), _topology_identity(grouped))
    require_equal("paired replay staging", eager["staging"], grouped["staging"])
    require_equal("paired warmup state", eager["warmup_restore"], grouped["warmup_restore"])
    require_equal("paired worker identity count", len(eager["worker_identities"]), len(grouped["worker_identities"]))
    for eager_identity, grouped_identity in zip(eager["worker_identities"], grouped["worker_identities"]):
        require_equal(
            "worker runtime identity",
            _worker_common_identity(eager_identity),
            _worker_common_identity(grouped_identity),
        )
    for eager_worker, grouped_worker in zip(eager["timed_workers"], grouped["timed_workers"]):
        require_equal(
            f"rank {eager_worker['rank']} exact per-layer route loads",
            eager_worker["expert_attribution"]["route_loads_per_layer"],
            grouped_worker["expert_attribution"]["route_loads_per_layer"],
        )
    numeric = _compare_numeric_evidence(grouped, eager, label="grouped versus eager")
    eager_wall = eager["metrics"]["synchronized_wall_seconds"]
    grouped_wall = grouped["metrics"]["synchronized_wall_seconds"]
    return {
        "numeric_evidence": numeric,
        "eager_wall_seconds": eager_wall,
        "grouped_wall_seconds": grouped_wall,
        "paired_wall_recovery_seconds": eager_wall - grouped_wall,
        "paired_wall_recovery_fraction": (eager_wall - grouped_wall) / eager_wall,
        "eager_routed_block_seconds": eager["metrics"]["critical_rank_expert_seconds"],
        "grouped_routed_block_seconds": grouped["metrics"]["critical_rank_expert_seconds"],
    }


def verify_instrumentation_oracle(instrumented: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    implementation = instrumented["expert_implementation"]
    _check_result_shape(oracle, implementation, attribution=False)
    require_equal("oracle common result identity", _common_identity(instrumented), _common_identity(oracle))
    require_equal("oracle config", instrumented["config"], oracle["config"])
    require_equal("oracle topology", _topology_identity(instrumented), _topology_identity(oracle))
    require_equal("oracle replay staging", instrumented["staging"], oracle["staging"])
    require_equal("oracle warmup state", instrumented["warmup_restore"], oracle["warmup_restore"])
    require_equal("oracle worker identities", instrumented["worker_identities"], oracle["worker_identities"])
    return _compare_numeric_evidence(instrumented, oracle, label="instrumented versus uninstrumented")


def main() -> None:
    args = parse_args()
    eager, eager_payload_sha256 = load_result(args.eager_result)
    grouped, grouped_payload_sha256 = load_result(args.grouped_result)
    pair = verify_pair(eager, grouped)
    oracle_evidence = None
    oracle_payload_sha256 = None
    if args.instrumentation_oracle_result:
        oracle, oracle_payload_sha256 = load_result(args.instrumentation_oracle_result)
        oracle_evidence = verify_instrumentation_oracle(eager, oracle)
    report = {
        "schema_version": 1,
        "verdict": "pass",
        "eager_result": args.eager_result,
        "eager_result_sha256": eager["result_sha256"],
        "eager_payload_sha256": eager_payload_sha256,
        "grouped_result": args.grouped_result,
        "grouped_result_sha256": grouped["result_sha256"],
        "grouped_payload_sha256": grouped_payload_sha256,
        "instrumentation_oracle_result": args.instrumentation_oracle_result,
        "instrumentation_oracle_payload_sha256": oracle_payload_sha256,
        "tolerances": {
            "representative_output": {"rtol": OUTPUT_RTOL, "atol": OUTPUT_ATOL},
            "global_ce_loss": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
            "representative_gradient": {"rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL},
            "per_layer_route_loads": "exact",
        },
        "pair": pair,
        "instrumentation_oracle": oracle_evidence,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
