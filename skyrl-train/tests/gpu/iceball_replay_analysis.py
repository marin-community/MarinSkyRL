from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import re
import statistics
import tarfile
from pathlib import Path

import torch
import zstandard
from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.resource_locator import join_resource_path

_CELL_RESULT = re.compile(r"(.+)-repeat-(\d+)\.json\Z")


def _read_archive(uri: str) -> dict[str, bytes]:
    filesystem, path = filesystem_and_path(uri)
    with filesystem.open(path, "rb") as source:
        compressed = source.read()
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(compressed)) as reader:
        data = reader.read()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {
            Path(member.name).name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith((".json", "fixture.pt"))
        }


def _masked_difference(a: list[list[float]], b: list[list[float]], mask: list[list[int]]) -> dict[str, float]:
    if len(a) != len(b) or len(a) != len(mask):
        raise ValueError("Probe and response-mask row counts differ")
    values = []
    for left, right, valid in zip(a, b, mask):
        if len(left) != len(right) or len(left) != len(valid):
            raise ValueError("Probe and response-mask widths differ")
        values.extend(abs(x - y) for x, y, selected in zip(left, right, valid) if selected)
    if not values:
        raise ValueError("Response probe contains no valid tokens")
    return {"mean_abs": statistics.mean(values), "max_abs": max(values)}


def _compare_cells(cells: dict[str, list[dict]], response_mask: list[list[int]]) -> dict[str, list[dict]]:
    pairs = {}
    for left, right in itertools.combinations(sorted(cells), 2):
        if len(cells[left]) != len(cells[right]):
            raise RuntimeError(f"Repetition counts differ between {left} and {right}")
        comparisons = []
        for repeat, (left_row, right_row) in enumerate(zip(cells[left], cells[right])):
            comparisons.append(
                {
                    "repetition": repeat,
                    "time_ratio_right_over_left": right_row["total_train_seconds"] / left_row["total_train_seconds"],
                    "initial_logprob": _masked_difference(
                        left_row["initial_probe"], right_row["initial_probe"], response_mask
                    ),
                    "final_logprob": _masked_difference(
                        left_row["final_probe"], right_row["final_probe"], response_mask
                    ),
                    "per_update_mean_abs": [
                        _masked_difference(a, b, response_mask)["mean_abs"]
                        for a, b in zip(left_row["update_probes"], right_row["update_probes"])
                    ],
                }
            )
        pairs[f"{left}_vs_{right}"] = comparisons
    return pairs


def analyze(uri: str) -> dict:
    files = _read_archive(uri)
    setup = json.loads(files["setup.json"])
    fixture = json.loads(files["remote-store.json"])
    batches = torch.load(io.BytesIO(files["fixture.pt"]), map_location="cpu", weights_only=True)["batches"]
    response_mask = batches[0]["response_mask"].tolist()
    result = {
        "archive_uri": uri,
        "historical_commit": setup["historical_commit"],
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
    result_files: dict[str, dict[int, str]] = {}
    for filename in files:
        if match := _CELL_RESULT.fullmatch(filename):
            result_files.setdefault(match.group(1), {})[int(match.group(2))] = filename
    available_cells = tuple(sorted(result_files))
    for cell in available_cells:
        repetitions = result_files[cell]
        if set(repetitions) != set(range(len(repetitions))):
            raise RuntimeError(f"Nonconsecutive repetitions in {cell}")
        rows = [json.loads(files[repetitions[repeat]]) for repeat in sorted(repetitions)]
        if any(
            row["steps"] < 1
            or row["steps"] > len(batches)
            or row["valid_tokens"] != sum(int(batch["response_mask"].sum()) for batch in batches[: row["steps"]])
            for row in rows
        ):
            raise RuntimeError(f"Unexpected work amount in {cell}")
        if len({row["steps"] for row in rows}) != 1:
            raise RuntimeError(f"Step counts differ within {cell}")
        if len({row["fixture_sha256_tensors"] for row in rows}) != 1:
            raise RuntimeError(f"Fixture changed within {cell}")
        durations = [row["total_train_seconds"] for row in rows]
        cells[cell] = rows
        result["cells"][cell] = {
            "fixture_sha256_tensors": rows[0]["fixture_sha256_tensors"],
            "training_seconds": durations,
            "mean_training_seconds": statistics.mean(durations),
            "stdev_training_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "mean_updates_per_second": rows[0]["steps"] / statistics.mean(durations),
            "mean_valid_tokens_per_second": rows[0]["valid_tokens"] / statistics.mean(durations),
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
    result["pairs"] = _compare_cells(cells, response_mask)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_uri", help="Full archive URI, or the task-output prefix through the job name")
    parser.add_argument("--output", type=Path, help="Write the analysis JSON to this path")
    args = parser.parse_args()
    uri = args.archive_uri
    if not uri.endswith(".tar.zst"):
        fs, path = filesystem_and_path(join_resource_path(uri, "0", "*", "outputs.tar.zst"))
        matches = fs.glob(path)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one result archive for {uri}; found {len(matches)}")
        uri = fs.unstrip_protocol(matches[0])
    result = analyze(uri)
    output = args.output
    if output is None and "IRIS_OUTPUT_DIR" in os.environ:
        output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "iceball-replay-summary.json"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
