"""Transport sensitivity for the actual Grug expert-bucket inventory.

This is an arithmetic envelope, not a fit to NCCL wall time. Local compute,
bucket packing and receiver apply are represented by an explicit penalty input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


BUCKET_MIB = (16, 32, 64, 128)
LATENCY_US = (0, 50, 1000)
COMPUTE_PENALTY_MS = (0, 50, 100)
DTYPE_BYTES = {"bfloat16": 2, "float32": 4}


def expert_buckets(experts: list[dict], maximum_bytes: int) -> list[list[dict]]:
    buckets: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for item in experts:
        size = item["entry"]["nbytes"]
        if current and (item["group"] != current[0]["group"] or current_bytes + size > maximum_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        buckets.append(current)
    return buckets


def inventory(schedule: dict, bucket_mib: int, changed: dict[str, int] | None = None) -> dict:
    experts = schedule["experts"]
    dense = schedule["dense"]
    buckets = expert_buckets(experts, bucket_mib * 2**20)
    expert_values = sum(item["entry"]["nbytes"] // 2 for item in experts)
    dense_bytes = sum(item["source"]["numel"] * DTYPE_BYTES[item["source"]["wire_dtype"]] for item in dense)
    dense_payload = dense_bytes + sum(item["entry"]["nbytes"] for item in experts)
    result = {
        "bucket_mib": bucket_mib,
        "expert_buckets": len(buckets),
        "expert_tensors": len(experts),
        "expert_values": expert_values,
        "dense_nonexpert_bytes": dense_bytes,
        "dense_bucket_logical_bytes": dense_payload,
        "dense_bucket_sender_collectives": len(buckets) + len(dense),
    }
    if changed is not None:
        assert len(changed) == len(experts)
        result.update(
            hybrid_logical_bytes=dense_bytes + 8 * len(experts) + 6 * sum(changed.values()),
            hybrid_sender_collectives=len(dense)
            + sum(1 + 2 * any(changed[item["entry"]["name"]] for item in bucket) for bucket in buckets),
            changed_expert_values=sum(changed.values()),
            zero_expert_tensors=sum(value == 0 for value in changed.values()),
        )
    return result


def observed_changed(update: dict) -> dict[str, int]:
    rows = (row for rank in update["distribution"] for row in rank)
    return {row["name"]: row["changed"] for row in rows if row["expert"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("real_raw", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    raw = args.real_raw.read_bytes()
    data = json.loads(raw)
    if not data["complete"] or len(data["updates"]) < 2:
        raise ValueError("Need a complete multi-update Grug artifact")
    schedule = data["schedule"]
    observed = {
        str(update["version"]): {
            str(cap): inventory(schedule, cap, observed_changed(update)) for cap in BUCKET_MIB
        }
        for update in data["updates"]
    }
    # The synthetic grid assumes every expert tensor has the same changed-value
    # density. Actual Grug updates have many zero tensors; their exact rows above
    # preserve that different activity pattern.
    density = np.geomspace(0.001, 0.8, 240)
    bandwidth = np.geomspace(0.1, 1000, 240)
    d, b = np.meshgrid(density, bandwidth)
    cap = 128
    base = inventory(schedule, cap)
    dense_payload = base["dense_bucket_logical_bytes"]
    sparse_payload = base["dense_nonexpert_bytes"] + 8 * base["expert_tensors"] + 6 * base["expert_values"] * d
    dense_calls = base["dense_bucket_sender_collectives"]
    sparse_calls = len(schedule["dense"]) + 3 * base["expert_buckets"]
    colors = ListedColormap(["#315e9d", "#e9903b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], colors.N)
    fig, axes = plt.subplots(len(COMPUTE_PENALTY_MS), len(LATENCY_US), figsize=(11, 9), sharex=True, sharey=True)
    counts = {}
    for row, penalty in enumerate(COMPUTE_PENALTY_MS):
        for col, latency in enumerate(LATENCY_US):
            dense_time = dense_payload / (b * 2**30) + dense_calls * latency * 1e-6
            sparse_time = penalty * 1e-3 + sparse_payload / (b * 2**30) + sparse_calls * latency * 1e-6
            winner = (sparse_time < dense_time).astype(int)
            axes[row, col].pcolormesh(d * 100, b, winner, cmap=colors, norm=norm, shading="auto")
            axes[row, col].set_xscale("log")
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(f"{penalty} ms compute penalty, {latency} µs/call")
            counts[f"{penalty}:{latency}"] = {
                "dense_expert_bucket": int((winner == 0).sum()),
                "sparse_expert_hybrid": int((winner == 1).sum()),
            }
    for ax in axes[-1]:
        ax.set_xlabel("Uniform expert changed values (%)")
        ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 80], labels=["0.1", "0.3", "1", "3", "10", "30", "80"])
    fig.text(0.018, 0.5, "Effective aggregate bandwidth (GiB/s)", rotation=90, va="center")
    legend = [plt.Rectangle((0, 0), 1, 1, facecolor=colors(i)) for i in range(2)]
    fig.legend(
        legend,
        ["dense expert buckets", "sparse expert hybrid"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
    )
    fig.suptitle("Grug 128 MiB bucket transport model; all experts active, local compute penalty is an input", y=0.99)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90, hspace=0.32, wspace=0.12)
    png = args.output_prefix.with_suffix(".png")
    result = args.output_prefix.with_suffix(".json")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=170)
    result.write_text(
        json.dumps(
            {
                "source_raw_sha256": hashlib.sha256(raw).hexdigest(),
                "source_commit": data["metadata"]["source_commit"],
                "observed_updates_by_bucket_mib": observed,
                "formula": "T_dense = dense logical bytes/B + dense sender calls*L; T_hybrid = compute penalty + hybrid logical bytes/B + hybrid sender calls*L",
                "uniform_sweep": {
                    "bucket_mib": cap,
                    "density_range": [float(density[0]), float(density[-1])],
                    "bandwidth_gib_s_range": [float(bandwidth[0]), float(bandwidth[-1])],
                    "latency_us": LATENCY_US,
                    "local_compute_penalty_ms": COMPUTE_PENALTY_MS,
                    "expert_activity": "every expert tensor active at the stated uniform changed-value density",
                    "winner_counts": counts,
                },
                "limitations": [
                    "Logical sender bytes and collective counts come from the actual Grug schedule, not physical NCCL link bytes.",
                    "Effective aggregate bandwidth, collective latency and local compute penalty are independent inputs, not fitted H100 measurements.",
                    "Bucket packing, receiver apply, NCCL contention and pause/resume are not separately predicted.",
                    "Uniform density makes every expert tensor active; observed updates contain many zero tensors and are tabulated separately.",
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
