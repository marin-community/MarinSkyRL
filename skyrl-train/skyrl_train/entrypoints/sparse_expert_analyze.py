"""Reduce raw routed-expert experiment JSON to small reproducible tables."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _range(values: list[float]) -> dict:
    return {"minimum": min(values), "median": statistics.median(values), "maximum": max(values)}


def _distribution(rows_by_rank: list[list[dict]]) -> dict:
    rows = [row for rank in rows_by_rank for row in rank]
    by_family = defaultdict(list)
    by_expert = defaultdict(list)
    by_size = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
        by_size[row["values"]].append(row)
        if row["expert"] is not None:
            by_expert[row["expert"]].append(row)

    def aggregate(items: list[dict]) -> dict:
        values = sum(row["values"] for row in items)
        changed = sum(row["changed"] for row in items)
        blocks = sum(row["total_256_value_blocks"] for row in items)
        return {
            "tensors": len(items),
            "values": values,
            "changed": changed,
            "density": changed / values if values else 0,
            "zero_tensors": sum(row["changed"] == 0 for row in items),
            "adjacent_changed_pair_fraction": sum(row["adjacent_changed_pairs"] for row in items) / changed
            if changed
            else 0,
            "occupied_256_value_block_fraction": sum(row["occupied_256_value_blocks"] for row in items) / blocks
            if blocks
            else 0,
        }

    expert_density = [aggregate(items)["density"] for items in by_expert.values()]
    return {
        "all": aggregate(rows),
        "family": {family: aggregate(items) for family, items in sorted(by_family.items())},
        "expert_density": _range(expert_density) if expert_density else None,
        "tensor_size": {str(size): aggregate(items) for size, items in sorted(by_size.items())},
    }


def _participant_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    summed = ("transfers", "changed_values", "total_values", "logical_bytes", "metadata_bytes", "collectives")
    timed = ("detect_seconds", "construct_seconds", "pack_allocation_seconds", "transfer_seconds", "apply_seconds")
    return {
        "participants": len(rows),
        "sum": {key: sum(row[key] for row in rows) for key in summed},
        "critical_rank": {key: max(row[key] for row in rows) for key in timed},
        "wall_range_seconds": _range([row["seconds"] for row in rows]),
        "max_gpu_allocated_start_bytes": max(row["gpu_allocated_start"] for row in rows),
        "max_gpu_peak_allocated_bytes": max(row["gpu_peak_allocated"] for row in rows),
        "max_gpu_peak_increment_bytes": max(row["gpu_peak_allocated"] - row["gpu_allocated_start"] for row in rows),
        "min_gpu_free_end_bytes": min(row["gpu_free_end"] for row in rows),
        "max_host_peak_rss_bytes": max(row["host_peak_rss"] for row in rows),
    }


def _trial(trial: dict) -> dict:
    row = {
        "encoding": trial["encoding"],
        "pause_seconds": trial["pause_seconds"],
        "install_seconds": trial["install_seconds"],
        "resume_seconds": trial["resume_seconds"],
        "publication_seconds": trial["publication_seconds"],
        "verification_seconds_outside_timer": trial["verification"]["verify_seconds"],
    }
    if trial["encoding"] == "dense":
        row["dense_report"] = trial["detail"]
    else:
        row["sender"] = _participant_metrics(trial["detail"]["policy"])
        row["receiver"] = _participant_metrics(trial["detail"]["receivers"])
    return row


def summarize(data: dict) -> dict:
    trials_by_encoding = defaultdict(list)
    updates = []
    for update in data.get("updates", []):
        trials = [_trial(trial) for trial in update["trials"]]
        for trial in trials:
            trials_by_encoding[trial["encoding"]].append(trial["publication_seconds"])
        updates.append(
            {
                "version": update["version"],
                "train_seconds": update.get("train_seconds"),
                "distribution": _distribution(update["distribution"]) if "distribution" in update else None,
                "trials": trials,
            }
        )
    return {
        "complete": data.get("complete", False),
        "stage": data.get("stage"),
        "model": data.get("model"),
        "geometry": data.get("geometry"),
        "source_inventory_ranks": len(data.get("source_inventory", [])),
        "receiver_inventory_ranks": len(data.get("receiver_inventory", [])),
        "schedule": {
            "dense_transfers": len(data["schedule"]["dense"]),
            "expert_transfers": len(data["schedule"]["experts"]),
            "groups": len(data["schedule"]["groups"]),
            "receiver_logical_bytes": sum(value for _, value in data["schedule"]["receiver_bytes"]),
        }
        if "schedule" in data
        else None,
        "initial_dense": data.get("initial_dense"),
        "initial_verify": data.get("initial_verify"),
        "updates": updates,
        "publication_seconds_by_encoding": {
            encoding: _range(samples) for encoding, samples in sorted(trials_by_encoding.items())
        },
        "failure": data.get("failure", {}).get("message"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.result.read_text()))
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
