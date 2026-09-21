"""Fit conditional publication crossover models from retained raw experiment data."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
for name in ("dense", "sparse", "probe", "relay_send", "relay_receive"):
    parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
FILES = {name: getattr(args, name) for name in ("dense", "sparse", "probe", "relay_send", "relay_receive")}
data = {key: json.loads(path.read_text()) for key, path in FILES.items()}
hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in FILES.items()}


def med(rows, key):
    return statistics.median(row[key] for row in rows)


dense = [row["metrics"] for row in data["dense"]["rows"] if row["kind"] == "train" and row["step"] >= 6]
sparse = [row["metrics"] for row in data["sparse"]["rows"] if row["kind"] == "train" and row["step"] >= 6]
assert len(dense) == len(sparse) == 20
global_expert_bytes = med(dense, "timing/expert_block_sync/logical_dense_bytes")
roots = 16  # PP2 x EP8 unique trainer-side roots; DP2 holds replicas.
root_bytes = global_expert_bytes / roots
baseline_peak = data["sparse"]["rows"][0]["metrics"]["startup/expert_block_sync/sender_commit_gpu_peak_transient_bytes"]
assert abs(root_bytes - baseline_peak) < 1024
rho = med(sparse, "timing/expert_block_sync/changed_density")
dense_expert_seconds = med(dense, "timing/expert_block_sync/expert_seconds")
apparent_bytes_per_second = root_bytes / dense_expert_seconds
messages_dense = med(dense, "timing/expert_block_sync/expert_collectives") / roots
messages_sparse = med(sparse, "timing/expert_block_sync/expert_collectives") / roots
install_delta = med(sparse, "timing/expert_block_sync/install_seconds") - med(
    dense, "timing/expert_block_sync/install_seconds"
)
commit_seconds = med(sparse, "timing/expert_block_sync/commit_seconds")
calibration_latency = 10e-6
fixed_delta = (
    install_delta
    - (messages_sparse - messages_dense) * calibration_latency
    - (3 * rho - 1) * root_bytes / apparent_bytes_per_second
)


def crossover(bandwidth_gb_s: float, latency_us: float, commit_overlap: float) -> float:
    critical_fixed = fixed_delta + (messages_sparse - messages_dense) * latency_us * 1e-6
    critical_fixed += (1 - commit_overlap) * commit_seconds
    return (1 - critical_fixed * bandwidth_gb_s * 1e9 / root_bytes) / 3


topology = []
for name, bandwidth, latency in [
    ("one_nvlink_domain_low", 100, 5),
    ("one_nvlink_domain_high", 300, 2),
    ("same_dc_nic_low", 15, 10),
    ("same_dc_nic_calibrated", apparent_bytes_per_second / 1e9, 10),
    ("same_dc_nic_high", 30, 30),
]:
    topology.append(
        {
            "name": name,
            "bandwidth_GB_per_second": bandwidth,
            "assumed_message_latency_us": latency,
            "install_only_crossover_density": crossover(bandwidth, latency, 1),
            "full_commit_crossover_density": crossover(bandwidth, latency, 0),
        }
    )

send = data["relay_send"]["samples"]
receive = {(row["size_mib"], row["density"], row["encoding"]): row for row in data["relay_receive"]["samples"]}
relay = []
for src in send:
    dst = receive[(src["size_mib"], src["density"], src["encoding"])]
    relay.append(
        {
            "size_mib": src["size_mib"],
            "density": src["density"],
            "encoding": src["encoding"],
            "object_mib": src["object_bytes"] / 2**20,
            "storage_seconds": src["put_seconds"] + dst["get_seconds"],
            "end_to_end_seconds": src["encode_seconds"]
            + src["staging_seconds"]
            + src["put_seconds"]
            + dst["get_seconds"]
            + dst["staging_seconds"]
            + dst["apply_seconds"],
            "local_seconds": src["encode_seconds"]
            + src["staging_seconds"]
            + dst["staging_seconds"]
            + dst["apply_seconds"],
        }
    )


def features(row, encoding):
    if encoding == "dense":
        return [1, row["object_mib"]]
    return [1, row["object_mib"], row["size_mib"]]


def fit(encoding, target):
    selected = [row for row in relay if row["encoding"] == encoding]
    holdout = next(row for row in selected if row["size_mib"] == 128 and row["density"] == 0.231647)
    train = [row for row in selected if row is not holdout]
    design = np.array([features(row, encoding) for row in train])
    values = np.array([row[target] for row in train])
    coeff = np.linalg.lstsq(design, values, rcond=None)[0]
    prediction = float(np.dot(features(holdout, encoding), coeff))
    return {
        "coefficients": coeff.tolist(),
        "training_points": len(train),
        "holdout": {
            "size_mib": 128,
            "density": 0.231647,
            "predicted_seconds": prediction,
            "observed_seconds": holdout[target],
            "relative_error": (prediction - holdout[target]) / holdout[target],
        },
    }


remote_fit = {
    encoding: {target: fit(encoding, target) for target in ("storage_seconds", "end_to_end_seconds")}
    for encoding in ("dense", "indices", "bitmap")
}
cpu_cases = {case["density"]: case for case in data["probe"]["cases"] if case["logical_bytes"] == 128 * 2**20}
cpu_keys = (
    "d2h_seconds",
    "cpu_detect_seconds",
    "cpu_pack_seconds",
    "h2d_seconds",
    "cpu_apply_seconds",
    "cpu_commit_seconds",
)
cpu_points = sorted((density, sum(case["median"][key] for key in cpu_keys)) for density, case in cpu_cases.items())


def cpu_time(density):
    for (left_density, left_time), (right_density, right_time) in zip(cpu_points, cpu_points[1:]):
        if left_density <= density <= right_density:
            weight = (density - left_density) / (right_density - left_density)
            return left_time * (1 - weight) + right_time * weight
    raise ValueError("CPU probe does not cover this density")


def remote_cpu_indices(density):
    coeff = remote_fit["indices"]["storage_seconds"]["coefficients"]
    object_mib = 3 * density * 128
    return coeff[0] + coeff[1] * object_mib + coeff[2] * 128 + cpu_time(density)


dense_local_128 = statistics.median(
    row["local_seconds"]
    for row in relay
    if row["encoding"] == "dense" and row["size_mib"] == 128 and row["density"] != 0.231647
)
dense_storage_coeff = remote_fit["dense"]["storage_seconds"]["coefficients"]
remote_dense_128 = dense_storage_coeff[0] + dense_storage_coeff[1] * 128 + dense_local_128
grid = np.linspace(0.025, 0.5, 1000)
crossing = min(grid, key=lambda density: abs(remote_cpu_indices(float(density)) - remote_dense_128))
result = {
    "schema": "sparse-publication-topology-cost-model-v1",
    "input_sha256": hashes,
    "calibration": {
        "direct_path": {
            "postwarmup_steps_per_arm": 20,
            "unique_roots": roots,
            "expert_bytes_global": global_expert_bytes,
            "expert_bytes_per_root": root_bytes,
            "observed_density": rho,
            "dense_expert_phase_seconds": dense_expert_seconds,
            "apparent_byte_rate_GB_per_second": apparent_bytes_per_second / 1e9,
            "collectives_per_root_dense": messages_dense,
            "collectives_per_root_sparse": messages_sparse,
            "observed_install_delta_seconds": install_delta,
            "observed_sparse_commit_seconds": commit_seconds,
            "assumed_calibration_message_latency_us": 10,
            "fitted_nonwire_delta_seconds": fixed_delta,
            "formula": "sparse_minus_dense = fitted_nonwire_delta + (sparse_messages-dense_messages)*latency + (3*density-1)*expert_bytes_per_root/bandwidth + (1-commit_overlap)*commit_seconds",
        },
        "cross_region_s3_relay": {
            "description": "One H100 sender to GB200 receiver through two S3 legs; one sample per condition.",
            "models": remote_fit,
            "features": "dense: [1, object_MiB]; sparse: [1, object_MiB, logical_MiB]. Intercepts and slopes are effective service costs, not physical network rates.",
        },
    },
    "predictions": {
        "direct_crossover": topology,
        "remote_128mib": {
            "dense_fitted_seconds": remote_dense_128,
            "cpu_index_predicted_seconds_at_2p5pct": remote_cpu_indices(0.025),
            "cpu_index_predicted_seconds_at_23pct": remote_cpu_indices(0.23),
            "cpu_index_crossover_density_fitted": float(crossing),
            "cpu_probe_seconds": cpu_points,
            "condition": "Sequential 128 MiB two-leg S3 object; no full-model concurrency or overlap assumed. CPU phases from separate one-H100 probe; cross-region receiver was GB200.",
        },
    },
}
output = args.output
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(output)
print(json.dumps({"direct": topology, "remote": result["predictions"]["remote_128mib"]}, indent=2))
