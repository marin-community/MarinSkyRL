"""Plot a deliberately simple transfer sensitivity model from measured GPU kernels.

The one-GPU sweep measures encode and apply, without NCCL. Add an explicit
payload/bandwidth term and one latency per logical collective. The result is a
sensitivity envelope, not a fitted prediction of the multi-node job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

ENCODINGS = ("dense", "indices", "bitmap")
FAMILIES = ("expert_fc2", "expert_fc1", "embedding")
LATENCIES_US = (0, 50, 1000)


def measured_kernels(samples: list[dict]) -> dict:
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["family"], sample["encoding"], sample["requested_density"])].append(sample)
    table = defaultdict(dict)
    for (family, encoding, density), rows in grouped.items():
        table[(family, encoding)][density] = {
            "seconds": statistics.median(row["total_seconds"] for row in rows),
            "minimum": min(row["total_seconds"] for row in rows),
            "maximum": max(row["total_seconds"] for row in rows),
            "values": rows[0]["values"],
            "repeats": len(rows),
        }
    return dict(table)


def component(table: dict, family: str, encoding: str, density: np.ndarray) -> np.ndarray:
    points = table[(family, encoding)]
    xs = np.array(sorted(points))
    ys = np.array([points[x]["seconds"] for x in xs])
    return np.interp(density, xs, ys)


def wire_bytes(encoding: str, values: int, density: np.ndarray) -> np.ndarray:
    if encoding == "dense":
        return np.full_like(density, 2 * values)
    if encoding == "indices":
        return 8 + 6 * values * density
    return 8 + np.ceil(values / 8) + 2 * values * density


def prediction(table: dict, family: str, density: np.ndarray, bandwidth_gib_s: np.ndarray, latency_us: int):
    values = next(iter(table[(family, "dense")].values()))["values"]
    return np.stack(
        [
            component(table, family, encoding, density)
            + wire_bytes(encoding, values, density) / (bandwidth_gib_s * 2**30)
            + (1 if encoding == "dense" else 3) * latency_us * 1e-6
            for encoding in ENCODINGS
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("microbench", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw = args.microbench.read_bytes()
    source = json.loads(raw)
    if not source["complete"]:
        raise ValueError("The GPU sweep is incomplete")
    table = measured_kernels(source["samples"])
    density = np.geomspace(0.001, 0.8, 240)
    bandwidth = np.geomspace(0.1, 1000, 240)
    d, b = np.meshgrid(density, bandwidth)
    fig, axes = plt.subplots(len(LATENCIES_US), len(FAMILIES), figsize=(12, 9), sharex=True, sharey=True)
    colors = ListedColormap(["#315e9d", "#e9903b", "#2a9d8f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N)
    counts = {}
    for row, latency in enumerate(LATENCIES_US):
        for col, family in enumerate(FAMILIES):
            winner = prediction(table, family, d, b, latency).argmin(axis=0)
            ax = axes[row, col]
            ax.pcolormesh(d * 100, b, winner, cmap=colors, norm=norm, shading="auto")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{family}, {latency} µs/call")
            counts[f"{family}:{latency}"] = {
                encoding: int((winner == i).sum()) for i, encoding in enumerate(ENCODINGS)
            }
    for ax in axes[-1]:
        ax.set_xlabel("Changed values (%)")
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 80], labels=["0.1", "0.3", "1", "3", "10", "30", "80"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Effective bandwidth (GiB/s)")
    legend = [plt.Rectangle((0, 0), 1, 1, facecolor=colors(i)) for i in range(3)]
    fig.legend(legend, ENCODINGS, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Measured H100 GPU kernels + analytical payload / bandwidth + collective latency")
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    png = args.output_prefix.with_suffix(".png")
    result = args.output_prefix.with_suffix(".json")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=170)
    result.write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_commit": source["source_commit"],
                "gpu": source["gpu"],
                "formula": "T = measured median GPU component + logical bytes / bandwidth + calls * latency",
                "collective_calls": {"dense": 1, "indices": 3, "bitmap": 3},
                "density_range": [float(density[0]), float(density[-1])],
                "bandwidth_gib_s_range": [float(bandwidth[0]), float(bandwidth[-1])],
                "latencies_us": LATENCIES_US,
                "kernel_measurements": {
                    f"{family}:{encoding}": points for (family, encoding), points in table.items()
                },
                "grid_winner_counts": counts,
                "limitations": [
                    "GPU timings are measured on one H100; bandwidth and latency are sensitivity inputs.",
                    "NCCL collective contention, metadata serialization, bucket packing and pauses are excluded.",
                    "Interpolation is only within measured 0.1%-80% changed-value densities.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(png, result)


if __name__ == "__main__":
    main()
