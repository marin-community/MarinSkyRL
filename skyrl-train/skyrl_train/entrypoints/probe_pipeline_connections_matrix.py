"""Run four fresh CUDA contexts to distinguish queue and allocation effects."""

import argparse
import itertools
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from skyrl_train.entrypoints.probe_pipeline_connections import BUCKET_BYTES, BUCKET_COUNT, validate_pair
from skyrl_train.weight_sync.pipeline_timing import overlap_seconds


def validate_matrix(cells, source_commit):
    assert len(cells) == 4
    keyed = {}
    processes = set()
    gpu_sets = set()
    attempts = set()
    summaries = []
    for cell in cells:
        assert cell["source_commit"] == source_commit
        rows = cell["rows"]
        connections, allocation = rows[0]["connections"], rows[0]["allocation"]
        key = (connections, allocation)
        assert key not in keyed, "Duplicate connection/allocation cell"
        keyed[key] = cell
        validate_pair(rows, connections, allocation)
        for rank in (0, 1):
            pair = [row for row in rows if row["rank"] == rank]
            assert pair[0]["identity"] == pair[1]["identity"]
            process = (pair[0]["identity"]["host"], pair[0]["identity"]["pid"])
            assert process not in processes, "Connection/allocation cells must use fresh processes"
            processes.add(process)
        gpu_sets.add(tuple(sorted({row["identity"]["gpu_uuid"] for row in rows})))
        for row in rows:
            attempts.add((row["identity"]["task_id"], row["identity"]["attempt_uid"]))
            expected_pool = BUCKET_BYTES * BUCKET_COUNT if row["rank"] == 0 and allocation == "preallocated" else 0
            assert row["source_pool_bytes"] == expected_pool
            stages = row["timing"]["intervals"]
            summaries.append(
                {
                    "connections": connections,
                    "allocation": allocation,
                    "mode": row["mode"],
                    "rank": row["rank"],
                    "export_send_overlap_seconds": overlap_seconds(
                        stages.get("export", []), stages.get("nccl_send", [])
                    ),
                    "receive_load_overlap_seconds": overlap_seconds(
                        stages.get("nccl_receive", []), stages.get("load", [])
                    ),
                    "stage_seconds": row["timing"]["stage_seconds"],
                }
            )
    assert set(keyed) == set(itertools.product((1, 8), ("streaming", "preallocated")))
    assert len(gpu_sets) == 1 and len(attempts) == 1
    assert next(iter(attempts))[0] and next(iter(attempts))[1]
    assert len({row["manifest_id"] for cell in cells for row in cell["rows"]}) == 1
    return summaries


def run_fresh_pair(command, environment):
    with subprocess.Popen(command, env=environment, start_new_session=True) as process:
        try:
            return process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    cells = []
    for connections, allocation in itertools.product((1, 8), ("streaming", "preallocated")):
        output = args.output / f"connections-{connections}-{allocation}.json"
        environment = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS=str(connections))
        exit_code = run_fresh_pair(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "skyrl_train.entrypoints.probe_pipeline_connections",
                "--source-commit",
                args.source_commit,
                "--connections",
                str(connections),
                "--allocation",
                allocation,
                "--output",
                str(output),
            ],
            environment,
        )
        results.append({"connections": connections, "allocation": allocation, "exit_code": exit_code})
        if output.exists():
            cells.append(json.loads(output.read_text()))
    # Retain all observed cells before evaluating the complete diagnostic matrix.
    print(
        "K9_CONNECTION_MATRIX_RECEIPT "
        + json.dumps({"source_commit": args.source_commit, "results": results, "cells": cells}),
        flush=True,
    )
    assert all(result["exit_code"] == 0 for result in results)
    summaries = validate_matrix(cells, args.source_commit)
    print("K9_CONNECTION_MATRIX_OBSERVATIONS " + json.dumps(summaries), flush=True)
    print(
        "K9_CONNECTION_MATRIX_PASS connections=1,8 allocations=streaming,preallocated fresh_process_pairs=4 serial_controls=4 exact_bytes=true operation_complete=true overlap_is_observation=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
