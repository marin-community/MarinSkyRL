"""Sample, replay, grade, and prepare the four released Pivot datasets locally."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice, zip_longest
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_url
from skyrl_gym.envs.nemotron_ultra.pivot import (
    NEMO_REFERENCE_COMMIT,
    PIVOT_PROFILES,
    grade_pivot_response,
    reference_response,
)

from infra.rl_data.sources import prepare_pivot_row

MAX_SAMPLE_BYTES = 32 * 1024 * 1024


def sample_dataset(dataset: str, output: Path, *, limit: int, revision: str) -> dict[str, Any]:
    """Stream a bounded JSONL prefix; record the resolved dataset commit beside it."""
    profile = PIVOT_PROFILES[dataset]
    resolved = HfApi().dataset_info(profile.dataset_id, revision=revision).sha
    url = hf_hub_url(profile.dataset_id, profile.filename, repo_type="dataset", revision=resolved)
    size = 0
    rows = []
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        buffer = b""
        for chunk in response.iter_content(chunk_size=16384):
            size += len(chunk)
            if size > MAX_SAMPLE_BYTES:
                raise ValueError("Sample exceeds the 32 MiB read budget; request fewer rows")
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) == limit:
                    break
            if len(rows) == limit:
                break
        else:
            if buffer.strip():
                rows.append(json.loads(buffer))
    if not rows:
        raise ValueError("Dataset contains no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": profile.dataset_id,
        "revision": resolved,
        "filename": profile.filename,
        "rows": len(rows),
        "bytes_read": size,
        "verifier_reference": NEMO_REFERENCE_COMMIT,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open() as stream:
        rows = [json.loads(line) for line in islice(stream, limit)]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def replay_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check reference and empty-response rewards; this is not model evaluation."""
    failures = []
    for index, row in enumerate(rows):
        positive, details = grade_pivot_response(dataset, row, reference_response(row))
        empty, _ = grade_pivot_response(dataset, row, {})
        if positive != 1.0 or empty != 0.0:
            failures.append({"index": index, "reference_reward": positive, "empty_reward": empty, **details})
    return {"mode": "reference_replay", "rows": len(rows), "passed": len(rows) - len(failures), "failures": failures}


def grade_predictions(
    dataset: str, rows: list[dict[str, Any]], predictions: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    """Grade indexed predictions against the corresponding sampled input rows."""
    results = []
    for index, (row, prediction) in enumerate(zip_longest(rows, predictions)):
        if row is None or prediction is None or prediction.get("index") != index:
            raise ValueError("Predictions must have one entry per input row, with consecutive zero-based indices")
        reward, details = grade_pivot_response(dataset, row, prediction["response"])
        results.append({"index": index, "uuid": row.get("uuid"), "reward": reward, **details})
    with output.open("w") as stream:
        for result in results:
            stream.write(json.dumps(result) + "\n")
    categories = Counter(result.get("failure_reason", result.get("category")) for result in results)
    return {
        "mode": "predictions",
        "rows": len(results),
        "mean_reward": sum(r["reward"] for r in results) / len(results),
        "categories": dict(categories),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("sample", "replay", "grade", "prepare"):
        child = commands.add_parser(command)
        child.add_argument("--dataset", choices=sorted(PIVOT_PROFILES), required=True)
        child.add_argument("--limit", type=int, default=16)
        if command == "sample":
            child.add_argument("--revision", default="main")
        else:
            child.add_argument("--input", type=Path, required=True)
        if command != "replay":
            child.add_argument("--output", type=Path, required=True)
        if command == "grade":
            child.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "sample":
        summary = sample_dataset(args.dataset, args.output, limit=args.limit, revision=args.revision)
    else:
        rows = read_rows(args.input, args.limit)
        if args.command == "replay":
            summary = replay_rows(args.dataset, rows)
        elif args.command == "grade":
            # One extra line detects accidentally grading a prefix of a prediction file.
            predictions = read_rows(args.predictions, len(rows) + 1)
            summary = grade_predictions(args.dataset, rows, predictions, args.output)
        else:
            prepared = [prepare_pivot_row(row, index, dataset=args.dataset) for index, row in enumerate(rows)]
            Dataset.from_list(prepared).to_parquet(str(args.output))
            summary = {"rows": len(prepared), "output": str(args.output)}
    print(json.dumps(summary, indent=2))
    if args.command == "replay" and summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
