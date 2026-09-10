"""Synthetic proof acceptance rejects corrupt bytes and misleading controls."""

import copy

import pytest

from skyrl_train.entrypoints.probe_pipeline_connections_matrix import validate_matrix

from skyrl_train.entrypoints.probe_pipeline_connections import (
    BUCKET_BYTES,
    BUCKET_COUNT,
    REQUIRED_ENVIRONMENT,
    validate_pair,
)


def receipt_rows(connections=1, allocation="streaming"):
    required_environment = {**REQUIRED_ENVIRONMENT, "CUDA_DEVICE_MAX_CONNECTIONS": str(connections)}
    rows = []
    for mode in ("serial", "pipeline"):
        for rank in (0, 1):
            starts = [10 * i for i in range(BUCKET_COUNT)]
            intervals = {"export": [[t, t + 2] for t in starts], "pack": [[t + 2, t + 3] for t in starts]}
            intervals["nccl_send"] = [[t + (1 if mode == "pipeline" else 3), t + 4] for t in starts]
            if rank:
                intervals = {"nccl_receive": [[t, t + 2] for t in starts], "load": [[t + 2, t + 3] for t in starts]}
            rows.append(
                {
                    "mode": mode,
                    "connections": connections,
                    "allocation": allocation,
                    "rank": rank,
                    "manifest_id": "one-manifest",
                    "identity": {"gpu_uuid": f"gpu-{rank}"},
                    "environment": dict(required_environment),
                    "operation_complete": True,
                    "timing": {"events_complete": True, "intervals": intervals},
                    "installed_bytes": BUCKET_BYTES * BUCKET_COUNT if rank else 0,
                    "mismatched_bytes": 0,
                }
            )
    return rows


def test_complete_serial_and_concurrent_controls():
    validate_pair(receipt_rows(), 1, "streaming")


@pytest.mark.parametrize(
    "failure",
    ["bytes", "missing", "duplicate", "unfinished", "nonfinite", "serial_overlap", "environment"],
)
def test_rejects_invalid_device_proof(failure):
    rows = copy.deepcopy(receipt_rows())
    if failure == "environment":
        rows[3]["environment"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    elif failure == "bytes":
        rows[1]["mismatched_bytes"] = 1
    elif failure == "missing":
        rows[0]["timing"]["intervals"]["export"].pop()
    elif failure == "duplicate":
        rows[3] = rows[1]
    elif failure == "unfinished":
        rows[1]["operation_complete"] = False
    elif failure == "nonfinite":
        rows[0]["timing"]["intervals"]["export"][0][1] = float("nan")
    elif failure == "serial_overlap":
        rows[0]["timing"]["intervals"]["nccl_send"][0] = [1, 4]
    with pytest.raises((AssertionError, ValueError)):
        validate_pair(rows, 1, "streaming")


def test_preserves_zero_pipeline_overlap_as_diagnostic_observation():
    rows = receipt_rows()
    rows[2]["timing"]["intervals"]["nccl_send"] = rows[0]["timing"]["intervals"]["nccl_send"]
    validate_pair(rows, 1, "streaming")


def matrix_cells():
    cells = []
    for index, (connections, allocation) in enumerate(
        ((1, "streaming"), (1, "preallocated"), (8, "streaming"), (8, "preallocated"))
    ):
        rows = receipt_rows(connections, allocation)
        for row in rows:
            rank = row["rank"]
            row["identity"].update(host="host", pid=index * 2 + rank, task_id="task:0", attempt_uid="original")
            row["source_pool_bytes"] = BUCKET_BYTES * BUCKET_COUNT if rank == 0 and allocation == "preallocated" else 0
            row["timing"]["stage_seconds"] = {
                key: sum(end - start for start, end in intervals)
                for key, intervals in row["timing"]["intervals"].items()
            }
        cells.append({"source_commit": "frozen", "rows": rows})
    return cells


def test_accepts_complete_fresh_process_matrix():
    summary = validate_matrix(matrix_cells(), "frozen")
    assert len(summary) == 16
    assert all(row["export_send_overlap_seconds"] == 0 for row in summary if row["mode"] == "serial")


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "reused_process", "changed_attempt", "pool_allocation", "changed_source"]
)
def test_rejects_confounded_matrix(failure):
    cells = matrix_cells()
    if failure == "missing":
        cells.pop()
    elif failure == "duplicate":
        cells[3] = cells[0]
    elif failure == "reused_process":
        for row in cells[1]["rows"]:
            row["identity"]["pid"] = row["rank"]
    elif failure == "changed_attempt":
        for row in cells[1]["rows"]:
            row["identity"]["attempt_uid"] = "replacement"
    elif failure == "pool_allocation":
        cells[1]["rows"][0]["source_pool_bytes"] = 0
    else:
        cells[1]["source_commit"] = "different"
    with pytest.raises(AssertionError):
        validate_matrix(cells, "frozen")
