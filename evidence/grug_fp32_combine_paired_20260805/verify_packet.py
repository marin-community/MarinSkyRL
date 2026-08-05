#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Independent structural and arithmetic readback for the paired packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


PARENT_REVISION = "0c213586b5491b8046ca7780e965c4b26dc6a2a2"
CANDIDATE_REVISION = "fbb1fc8378601e0346d00d186809f10d1ad0360d"
PARENT_MODULE_SHA256 = "b1e63368996530dd8fa678ec3b482a1bd63007c0d69901cd13c6a4e42c294d50"
CANDIDATE_MODULE_SHA256 = "2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93"
GPU_COUNT = 8
ITERATIONS_PER_PROCESS = 20
PROCESSES_PER_ARM_PER_GPU = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _range(values: list[float]) -> list[float]:
    return [float(min(values)), float(max(values))]


def _metric(record: dict[str, Any], boundary: str, metric: str) -> list[float]:
    return [float(value) for value in record[boundary][metric]]


def _summarize(raw: dict[str, Any]) -> dict[str, Any]:
    records = raw["performance"]
    per_gpu: list[dict[str, Any]] = []
    metric_specs = {
        "full_forward_ms": ("full_block", "forward_ms"),
        "full_backward_ms": ("full_block", "backward_ms"),
        "full_forward_backward_ms": ("full_block", "forward_backward_ms"),
        "combine_forward_ms": ("combine_boundary", "forward_ms"),
        "combine_backward_ms": ("combine_boundary", "backward_ms"),
        "combine_forward_backward_ms": ("combine_boundary", "forward_backward_ms"),
    }
    for gpu in range(GPU_COUNT):
        gpu_records = [record for record in records if record["physical_gpu"] == gpu]
        arms: dict[str, Any] = {}
        for arm in ("parent", "candidate"):
            arm_records = [record for record in gpu_records if record["arm"] == arm]
            arm_metrics = {}
            for label, (boundary, metric) in metric_specs.items():
                values = [value for record in arm_records for value in _metric(record, boundary, metric)]
                arm_metrics[label] = {
                    "median": _median(values),
                    "range": _range(values),
                    "sample_count": len(values),
                }
            for boundary in ("full_block", "combine_boundary"):
                values = [float(record[boundary]["incremental_peak_allocated_bytes"]) for record in arm_records]
                arm_metrics[f"{boundary}_incremental_peak_allocated_bytes"] = {
                    "median": _median(values),
                    "range": _range(values),
                    "sample_count": len(values),
                }
            arms[arm] = arm_metrics
        deltas = {}
        for label in metric_specs:
            parent = arms["parent"][label]["median"]
            candidate = arms["candidate"][label]["median"]
            deltas[label] = {
                "candidate_minus_parent": candidate - parent,
                "candidate_over_parent": candidate / parent,
            }
        for boundary in ("full_block", "combine_boundary"):
            label = f"{boundary}_incremental_peak_allocated_bytes"
            parent = arms["parent"][label]["median"]
            candidate = arms["candidate"][label]["median"]
            deltas[label] = {
                "candidate_minus_parent": candidate - parent,
                "candidate_over_parent": candidate / parent if parent else None,
            }
        per_gpu.append({"gpu": gpu, "arms": arms, "deltas": deltas})

    cross_gpu = {}
    for label in [*metric_specs, "full_block_incremental_peak_allocated_bytes", "combine_boundary_incremental_peak_allocated_bytes"]:
        delta_values = [row["deltas"][label]["candidate_minus_parent"] for row in per_gpu]
        ratio_values = [row["deltas"][label]["candidate_over_parent"] for row in per_gpu]
        ratio_values = [value for value in ratio_values if value is not None]
        cross_gpu[label] = {
            "candidate_minus_parent_median": _median(delta_values),
            "candidate_minus_parent_range": _range(delta_values),
            "candidate_over_parent_median": _median(ratio_values) if ratio_values else None,
            "candidate_over_parent_range": _range(ratio_values) if ratio_values else None,
        }

    primary_deltas = [row["deltas"]["full_forward_backward_ms"]["candidate_minus_parent"] for row in per_gpu]
    if all(value > 0 for value in primary_deltas):
        direction = "candidate_slower_on_all_eight_gpus"
    elif all(value < 0 for value in primary_deltas):
        direction = "candidate_faster_on_all_eight_gpus"
    else:
        direction = "inconclusive_direction_cross_gpu_range_spans_zero"

    return {
        "schema_version": 1,
        "verdict": {
            "kind": "estimate_only_no_materiality_threshold",
            "primary_metric": "full_forward_backward_ms",
            "observed_direction": direction,
        },
        "per_gpu": per_gpu,
        "cross_gpu": cross_gpu,
    }


def _validate(raw: dict[str, Any], raw_path: Path, protocol_path: Path, driver_path: Path) -> list[str]:
    checks: list[str] = []
    protocol = json.loads(protocol_path.read_text())

    def require(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    require(raw["schema_version"] == 1 and raw["status"] == "complete", "complete schema-v1 packet")
    pins = raw["pins"]
    require(pins["parent_revision"] == PARENT_REVISION, "parent revision pin")
    require(pins["candidate_revision"] == CANDIDATE_REVISION, "candidate revision pin")
    require(pins["parent_module_sha256"] == PARENT_MODULE_SHA256, "parent module pin")
    require(pins["candidate_module_sha256"] == CANDIDATE_MODULE_SHA256, "candidate module pin")
    require(pins["protocol_sha256"] == _sha256_file(protocol_path), "frozen protocol digest")
    require(pins["driver_sha256"] == _sha256_file(driver_path), "executed driver digest")
    require(protocol["pins"]["driver_sha256"] == pins["driver_sha256"], "protocol driver pin")
    runtime = raw["runtime"]
    for field in (
        "cluster",
        "iris_job_id",
        "pod_name",
        "container_image_id",
        "python_package_inventory_sha256",
        "resource_request",
    ):
        require(runtime[field] == protocol["runtime"][field], f"frozen runtime {field}")
    inventory_text = "\n".join(runtime["python_package_inventory"]) + "\n"
    require(
        hashlib.sha256(inventory_text.encode()).hexdigest() == runtime["python_package_inventory_sha256"],
        "Python package inventory digest",
    )
    require(bool(runtime["nvidia_driver_version"]), "NVIDIA driver recorded")
    require(raw["schedule"]["verdict"] == "estimate_only_no_materiality_threshold", "estimate-only freeze")
    require(raw["shape"] == {
        "tokens": 8192,
        "hidden_size": 2560,
        "intermediate_size": 1280,
        "num_experts": 256,
        "top_k": 4,
    }, "real Grug expert shape")
    require(len(raw["runtime"]["gpu_inventory"]) == GPU_COUNT, "eight-GPU inventory")

    gate = raw["correctness_gate"]
    require(gate["pass"] and all(gate["checks"].values()), "frozen correctness gate")
    parent = raw["correctness"]["parent"]
    candidate = raw["correctness"]["candidate"]
    require(parent["fixture"]["hashes"] == candidate["fixture"]["hashes"], "paired correctness fixture identity")
    for path in ("eager", "grouped"):
        require(
            not parent["paths"][path]["actual_vs_fp32_slotwise"]["output"]["exact"],
            f"parent {path} fixture discrimination",
        )
        require(
            candidate["paths"][path]["actual_vs_fp32_slotwise"]["output"]["exact"],
            f"candidate {path} FP32 output contract",
        )
        require(
            all(
                value["allclose"]
                for value in candidate["paths"][path]["actual_vs_fp32_slotwise"]["gradients"].values()
            ),
            f"candidate {path} FP32 gradient contract",
        )

    records = raw["performance"]
    require(len(records) == GPU_COUNT * 4, "32 finite scheduled arm processes")
    for gpu in range(GPU_COUNT):
        gpu_records = [record for record in records if record["physical_gpu"] == gpu]
        expected_schedule = raw["schedule"]["arms_by_gpu"][str(gpu)]
        observed_schedule = [record["arm"] for record in gpu_records]
        require(observed_schedule == expected_schedule, f"GPU {gpu} alternating order")
        for arm in ("parent", "candidate"):
            arm_records = [record for record in gpu_records if record["arm"] == arm]
            require(len(arm_records) == PROCESSES_PER_ARM_PER_GPU, f"GPU {gpu} {arm} repetition count")
            require(
                sorted(record["repetition"] for record in arm_records) == [0, 1],
                f"GPU {gpu} {arm} repetition IDs",
            )
            for record in arm_records:
                require(record["module_sha256"] == pins[f"{arm}_module_sha256"], f"GPU {gpu} {arm} source pin")
                require(record["routes"]["minimum_rows_per_expert"] == 128, f"GPU {gpu} {arm} route minimum")
                require(record["routes"]["maximum_rows_per_expert"] == 128, f"GPU {gpu} {arm} route maximum")
                for boundary in ("full_block", "combine_boundary"):
                    require(
                        len(record[boundary]["forward_ms"]) == ITERATIONS_PER_PROCESS
                        and len(record[boundary]["backward_ms"]) == ITERATIONS_PER_PROCESS,
                        f"GPU {gpu} {arm} {boundary} sample count",
                    )
                    require(
                        record[boundary]["peak_allocated_bytes"] >= record[boundary]["baseline_allocated_bytes"],
                        f"GPU {gpu} {arm} {boundary} HBM accounting",
                    )
        fixture_fields = ("state", "hidden", "selected_experts", "combine_weights", "cotangent", "weighted_bf16_summands")
        for field in fixture_fields:
            values = {record["fixture_hashes"][field] for record in gpu_records}
            require(len(values) == 1, f"GPU {gpu} paired {field} identity")

    samples = raw["memory_monitor"]
    require(bool(samples), "host/cgroup memory samples present")
    require(
        max(sample["cgroup_current_bytes"] or 0 for sample in samples)
        < raw["memory_stop_rules"]["cgroup_stop_bytes"],
        "cgroup memory stayed below stop rule",
    )
    require(
        min(sample["host_mem_available_bytes"] for sample in samples)
        >= raw["memory_stop_rules"]["host_available_stop_bytes"],
        "host memory stayed above stop rule",
    )
    for sample in samples:
        total = sample["host_swap_total_bytes"]
        if total:
            require(
                sample["host_swap_free_bytes"] / total >= raw["memory_stop_rules"]["minimum_swap_free_fraction"],
                "swap stayed above stop rule",
            )
    require(_sha256_file(raw_path) != "", "raw packet SHA-256 computed")
    return checks


def _markdown(raw: dict[str, Any], summary: dict[str, Any], raw_sha256: str) -> str:
    lines = [
        "# Grug FP32 combine paired H100 measurement",
        "",
        f"Raw packet SHA-256: `{raw_sha256}`.",
        "",
        "The primary metric is warmed full sparse-block forward plus backward time. The verdict was frozen as estimate-only; no practical materiality threshold was invented after seeing results.",
        "",
        "| GPU | parent ms | candidate ms | candidate-parent ms | ratio |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["per_gpu"]:
        parent = row["arms"]["parent"]["full_forward_backward_ms"]["median"]
        candidate = row["arms"]["candidate"]["full_forward_backward_ms"]["median"]
        delta = row["deltas"]["full_forward_backward_ms"]["candidate_minus_parent"]
        ratio = row["deltas"]["full_forward_backward_ms"]["candidate_over_parent"]
        lines.append(f"| {row['gpu']} | {parent:.6f} | {candidate:.6f} | {delta:+.6f} | {ratio:.6f} |")
    primary = summary["cross_gpu"]["full_forward_backward_ms"]
    combine = summary["cross_gpu"]["combine_forward_backward_ms"]
    full_hbm = summary["cross_gpu"]["full_block_incremental_peak_allocated_bytes"]
    combine_hbm = summary["cross_gpu"]["combine_boundary_incremental_peak_allocated_bytes"]
    lines += [
        "",
        "## Compact paired summary",
        "",
        f"- Full block candidate-parent median delta: `{primary['candidate_minus_parent_median']:+.6f}` ms; per-GPU range `{primary['candidate_minus_parent_range'][0]:+.6f}` to `{primary['candidate_minus_parent_range'][1]:+.6f}` ms; median ratio `{primary['candidate_over_parent_median']:.6f}`.",
        f"- Complete combine boundary candidate-parent median delta: `{combine['candidate_minus_parent_median']:+.6f}` ms; per-GPU range `{combine['candidate_minus_parent_range'][0]:+.6f}` to `{combine['candidate_minus_parent_range'][1]:+.6f}` ms; median ratio `{combine['candidate_over_parent_median']:.6f}`.",
        f"- Full-block incremental peak allocated HBM delta: median `{full_hbm['candidate_minus_parent_median']:+.0f}` bytes; range `{full_hbm['candidate_minus_parent_range'][0]:+.0f}` to `{full_hbm['candidate_minus_parent_range'][1]:+.0f}` bytes.",
        f"- Combine-boundary incremental peak allocated HBM delta: median `{combine_hbm['candidate_minus_parent_median']:+.0f}` bytes; range `{combine_hbm['candidate_minus_parent_range'][0]:+.0f}` to `{combine_hbm['candidate_minus_parent_range'][1]:+.0f}` bytes.",
        f"- Estimate-only direction: `{summary['verdict']['observed_direction']}`.",
        "",
        "## Correctness and limits",
        "",
        "The fixed fixture distinguishes the parent's BF16 running-sum error. Candidate eager and grouped outputs match the independent FP32 slot-wise contract exactly, their combine-relevant gradients pass the frozen tolerance, and the FP64 fixed-order reduction reports the candidate's reduced local accumulation error.",
        "",
        "This is a local, fixed-route sparse-block result. It does not change the failed 32-H100 action-output gate, prove distributed semantic equivalence, measure end-to-end MFU, identify attention as causal, or put `fbb1fc8` on MarinSkyRL #276.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--summary-md")
    args = parser.parse_args()

    raw_path = Path(args.raw).resolve()
    protocol_path = Path(args.protocol).resolve()
    driver_path = Path(args.driver).resolve()
    raw = json.loads(raw_path.read_text())
    checks = _validate(raw, raw_path, protocol_path, driver_path)
    summary = _summarize(raw)
    summary["readback"] = {
        "checks_passed": len(checks),
        "checks": checks,
        "raw_sha256": _sha256_file(raw_path),
        "reader_sha256": _sha256_file(Path(__file__).resolve()),
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.summary_md:
        Path(args.summary_md).write_text(_markdown(raw, summary, summary["readback"]["raw_sha256"]))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
