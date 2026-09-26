from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import tarfile
from pathlib import Path

import fsspec
import zstandard

CELLS = ("old-fsdp2", "old-fsdp2-fp32", "old-megatron", "new-megatron")


def _read_archive(uri: str) -> dict[str, bytes]:
    with fsspec.open(uri, "rb") as source:
        compressed = source.read()
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
        data = reader.read()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {
            Path(member.name).name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith(".json")
        }


def _masked_difference(a: list[list[float]], b: list[list[float]]) -> dict[str, float]:
    values = [abs(a[row][column] - b[row][column]) for row in range(64) for column in range(6 + row * 7 % 11)]
    return {"mean_abs": statistics.mean(values), "max_abs": max(values)}


def analyze(uri: str) -> dict:
    files = _read_archive(uri)
    setup = json.loads(files["setup.json"])
    fixture = json.loads(files["remote-store.json"])
    result = {
        "archive_uri": uri,
        "source_commit": setup["historical_commit"],
        "current_commit": setup["current_commit"],
        "harness_sha256": setup["harness_sha256"],
        "model_uri": setup["model_uri"],
        "model_identity": setup["model_identity"],
        "model_stage_seconds": setup["stage_seconds"],
        "gpu_names": setup["gpu_names"],
        "remote_store": fixture,
        "cells": {},
        "pairs": {},
    }
    cells: dict[str, list[dict]] = {}
    available_cells = tuple(cell for cell in CELLS if f"{cell}-repeat-0.json" in files)
    for cell in available_cells:
        rows = [json.loads(files[f"{cell}-repeat-{repeat}.json"]) for repeat in range(3)]
        if any(row["steps"] != 8 or row["valid_tokens"] != 5626 for row in rows):
            raise RuntimeError(f"Unexpected work amount in {cell}")
        if len({row["fixture_sha256_tensors"] for row in rows}) != 1:
            raise RuntimeError(f"Fixture changed within {cell}")
        durations = [row["total_train_seconds"] for row in rows]
        cells[cell] = rows
        result["cells"][cell] = {
            "fixture_sha256_tensors": rows[0]["fixture_sha256_tensors"],
            "training_seconds": durations,
            "mean_training_seconds": statistics.mean(durations),
            "stdev_training_seconds": statistics.stdev(durations),
            "mean_updates_per_second": 8 / statistics.mean(durations),
            "mean_valid_tokens_per_second": 5626 / statistics.mean(durations),
            "initial_vs_frozen_logprob_mean_abs": [row["initial_vs_frozen_logprob_mean_abs"] for row in rows],
            "initial_vs_frozen_logprob_max_abs": [row["initial_vs_frozen_logprob_max_abs"] for row in rows],
            "initial_attempts": [row.get("initial_attempts", []) for row in rows],
            "post_update_probe_mean_abs_change": [row["post_update_probe_mean_abs_change"] for row in rows],
            "step_optimizer_updates": [
                [step["rank0_status"]["policy_update_steps"] for step in row["step_rows"]] for row in rows
            ],
            "step_seconds": [[step["max_rank_elapsed_seconds"] for step in row["step_rows"]] for row in rows],
            "rank0_status": [[step["rank0_status"] for step in row["step_rows"]] for row in rows],
        }
    if len({result["cells"][cell]["fixture_sha256_tensors"] for cell in available_cells}) != 1:
        raise RuntimeError("Fixture differs across cells")
    pairs = (
        ("old-fsdp2", "old-megatron"),
        ("old-fsdp2-fp32", "old-megatron"),
        ("old-fsdp2", "old-fsdp2-fp32"),
        ("old-megatron", "new-megatron"),
    )
    for left, right in pairs:
        if left not in cells or right not in cells:
            continue
        comparisons = []
        for repeat, (left_row, right_row) in enumerate(zip(cells[left], cells[right])):
            comparisons.append(
                {
                    "repetition": repeat,
                    "time_ratio_right_over_left": right_row["total_train_seconds"] / left_row["total_train_seconds"],
                    "initial_logprob": _masked_difference(left_row["initial_probe"], right_row["initial_probe"]),
                    "final_logprob": _masked_difference(left_row["final_probe"], right_row["final_probe"]),
                    "per_update_mean_abs": [
                        _masked_difference(a, b)["mean_abs"]
                        for a, b in zip(left_row["update_probes"], right_row["update_probes"])
                    ],
                }
            )
        result["pairs"][f"{left}_vs_{right}"] = comparisons
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_uri", help="Full archive URI, or the task-output prefix through the job name")
    args = parser.parse_args()
    uri = args.archive_uri
    if not uri.endswith(".tar.zst"):
        fs, path = fsspec.core.url_to_fs(uri)
        matches = fs.glob(path.rstrip("/") + "/0/*/outputs.tar.zst")
        if len(matches) != 1:
            raise RuntimeError(f"Expected one result archive for {uri}; found {len(matches)}")
        uri = fs.unstrip_protocol(matches[0])
    result = analyze(uri)
    output = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp")) / "iceball-replay-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
