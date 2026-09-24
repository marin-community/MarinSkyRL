"""Summarize matched cross-zone samples and a conditional one-root transfer envelope."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

MIB = 2**20
ROOTS = 16
BUCKET_BYTES = 128 * MIB
RELAY_LOGICAL_BYTES = 128 * MIB


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def spread(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "minimum": min(values),
        "maximum": max(values),
    }


def interpolate(rows: list[dict], size: float, *, above: str = "chunks") -> float:
    points = sorted((row["size_mib"] * MIB, row["delivery_seconds"]["p50"]) for row in rows)
    if size > points[-1][0]:
        if above == "linear":
            left_size, left_time = points[-2]
            right_size, right_time = points[-1]
            return right_time + (size - right_size) * (right_time - left_time) / (right_size - left_size)
        return math.ceil(size / points[-1][0]) * points[-1][1]
    for (left_size, left_time), (right_size, right_time) in pairwise(points):
        if left_size <= size <= right_size:
            weight = (size - left_size) / (right_size - left_size)
            return left_time + weight * (right_time - left_time)
    return points[0][1]


def summarize(sender: dict, receiver: dict) -> dict:
    sender_samples = sender["samples"]
    receiver_samples = {row["sample"]: row for row in receiver["samples"]}
    if len(sender_samples) != 60 or len(receiver_samples) != 60:
        raise ValueError("Expected 60 sender and 60 receiver samples")
    groups = defaultdict(list)
    for row in sender_samples:
        name = row["sample"]
        peer = receiver_samples[name]
        if row["sha256"] != row["receiver"]["sha256"] or row["sha256"] != peer["sha256"]:
            raise ValueError(f"Receiver checksum mismatch: {name}")
        groups[(row["size"] // MIB, row["streams"], row["method"])].append((row, peer))
    if set(groups) != {
        (size, streams, method) for size in (8, 128, 512) for streams in (1, 4) for method in ("tcp", "object")
    }:
        raise ValueError("A matched condition is missing")
    table = []
    for (size, streams, method), pairs in sorted(groups.items()):
        if len(pairs) != 5:
            raise ValueError(f"Expected five repeats: {size}, {streams}, {method}")
        rows = [pair[0] for pair in pairs]
        result = {
            "size_mib": size,
            "streams": streams,
            "method": method,
            "sample_ids": [row["sample"] for row in rows],
            "delivery_seconds": spread([row["delivery_seconds"] for row in rows]),
            "setup_seconds": spread([row["setup_seconds"] for row in rows]),
            "send_or_upload_phase_seconds": spread([row["send_or_upload_phase_seconds"] for row in rows]),
            "ack_wait_seconds": spread([row["ack_wait_seconds"] for row in rows]),
        }
        if method == "tcp":
            result["sender_total_retrans"] = [
                sum(part["tcp_info"]["total_retrans"] for part in row["send_or_upload_parts"]) for row in rows
            ]
            result["sender_pmtu"] = sorted(
                {part["tcp_info"]["pmtu"] for row in rows for part in row["send_or_upload_parts"]}
            )
            result["sender_mss"] = sorted(
                {part["tcp_info"]["snd_mss"] for row in rows for part in row["send_or_upload_parts"]}
            )
        else:
            result["visibility_from_command_seconds"] = spread(
                [max(read["visibility_seconds"] for read in peer["reads"]) for _, peer in pairs]
            )
            result["max_part_download_seconds"] = spread(
                [max(read["download_seconds"] for read in peer["reads"]) for _, peer in pairs]
            )
            result["warm_read_seconds"] = spread([peer["warm_get_seconds"] for _, peer in pairs])
            result["head_attempts"] = [[read["head_attempts"] for read in peer["reads"]] for _, peer in pairs]
            result["cache_response_headers"] = sorted(
                {
                    key
                    for _, peer in pairs
                    for read in peer["reads"]
                    for key in read["headers"]
                    if "cache" in key.lower() or "lota" in key.lower()
                }
            )
        table.append(result)
    return {
        "samples": len(sender_samples),
        "pings_seconds": spread([row["rtt_seconds"] for row in sender["pings"]]),
        "reverse_8_mib_seconds": sender["reverse"]["seconds"],
        "sender_host": sender["facts"]["hostname"],
        "sender_ip": sender["facts"]["primary_ip"],
        "receiver_host": receiver["facts"]["hostname"],
        "receiver_ip": receiver["facts"]["primary_ip"],
        "table": table,
    }


def baseline(dense: dict, sparse: dict, screen: dict, relay: dict) -> dict:
    dense_rows = [row["metrics"] for row in dense["rows"] if row["kind"] == "train" and row["step"] >= 6]
    sparse_rows = [row["metrics"] for row in sparse["rows"] if row["kind"] == "train" and row["step"] >= 6]
    if len(dense_rows) != len(sparse_rows) or len(dense_rows) != 20:
        raise ValueError("Expected the corrected 20 post-warmup Grug updates")
    logical = statistics.median(row["timing/expert_block_sync/logical_dense_bytes"] for row in dense_rows)
    encoded = statistics.median(row["timing/expert_block_sync/encoded_bytes"] for row in sparse_rows)
    densities = [row["timing/expert_block_sync/changed_density"] for row in sparse_rows]
    local = {row["requested_density"]: row for row in screen["medians"] if row["values"] == 67_108_864}
    relay_bytes = statistics.median(row["encoded_bytes"] for row in relay["rows"] if row["version"] >= 2)
    return {
        "global_dense_expert_bytes": logical,
        "global_sparse_expert_bytes_p50": encoded,
        "unique_roots": ROOTS,
        "dense_bytes_per_root": logical / ROOTS,
        "sparse_bytes_per_root_p50": encoded / ROOTS,
        "changed_density_p50": statistics.median(densities),
        "changed_density_min": min(densities),
        "changed_density_max": max(densities),
        "cpu_random_xor_relay_packet_bytes_per_128_mib": relay_bytes,
        "cpu_random_xor_relay_density": relay["density"],
        "cpu_local_screen_128_mib": {
            str(density): {
                "cpu_xor_seconds": row["cpu_xor_seconds"],
                "gpu_index_seconds": row["gpu_index_seconds"],
                "cpu_xor_zstd_bytes_low_bit_pattern": row["cpu_xor_zstd_bytes"],
            }
            for density, row in sorted(local.items())
        },
    }


def envelope(table: list[dict], inputs: dict) -> list[dict]:
    tcp = [row for row in table if row["method"] == "tcp" and row["streams"] == 1]
    obj = [row for row in table if row["method"] == "object" and row["streams"] == 4]
    root = inputs["dense_bytes_per_root"]
    relay_ratio = inputs["cpu_random_xor_relay_packet_bytes_per_128_mib"] / RELAY_LOGICAL_BYTES
    screen = inputs["cpu_local_screen_128_mib"]
    rows = []
    for density in (0.001, 0.005, 0.014, inputs["changed_density_p50"], 0.019, 0.024):
        closest = min(screen, key=lambda value: abs(float(value) - density))
        gpu_sparse = 3 * density * root
        cpu_random_xor = relay_ratio * density / inputs["cpu_random_xor_relay_density"] * root
        rows.append(
            {
                "density": density,
                "direct_dense_bytes_per_root": root,
                "direct_gpu_index_bytes_per_root_model": gpu_sparse,
                "object_cpu_xor_bytes_per_root_model": cpu_random_xor,
                "direct_dense_seconds_sequential_512_mib_chunks": interpolate(tcp, root),
                "direct_gpu_index_network_seconds_model": interpolate(tcp, gpu_sparse, above="linear"),
                "object_cpu_xor_network_seconds_model": interpolate(obj, cpu_random_xor),
                "cpu_local_seconds_sequential_bucket_screen_model": (
                    root / BUCKET_BYTES * screen[closest]["cpu_xor_seconds"]
                ),
                "cpu_local_screen_density_used": float(closest),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sender", "receiver", "dense", "sparse", "screen", "relay"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("sender", "receiver", "dense", "sparse", "screen", "relay")}
    loaded = {name: load(path) for name, path in paths.items()}
    measurements = summarize(loaded["sender"], loaded["receiver"])
    inputs = baseline(loaded["dense"], loaded["sparse"], loaded["screen"], loaded["relay"])
    result = {
        "schema": "cross-zone-transport-envelope-v1",
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "measured": measurements,
        "prior_input_sizes_and_local_costs": inputs,
        "modeled_one_unique_root": envelope(measurements["table"], inputs),
        "model_limits": [
            "TCP and object curves interpolate single-payload medians up to 512 MiB; dense 8.179 GB uses sequential 512 MiB chunks.",
            "The 2.4% GPU-index scenario exceeds 512 MiB by about 10% and linearly extrapolates the 128-to-512 MiB TCP median slope.",
            "The GPU index byte formula is 3*rho*BF16 dense bytes and omits headers and nonexpert weights.",
            "The CPU XOR byte ratio comes from one 128 MiB random-XOR RNO object relay at 1.9% and is scaled linearly by density.",
            "CPU local times come from one H100 low-bit-flip bucket screen and are scaled sequentially; full-model batching, B200 encode, H100 apply, and overlap are unmeasured.",
            "One root is 1/16 of global routed-expert bytes; receiver placement, fan-out, nonexpert tensors, and acknowledgement are not modeled.",
            "Visibility time starts when the receiver got the control command and overlaps source upload; components are not additive.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
