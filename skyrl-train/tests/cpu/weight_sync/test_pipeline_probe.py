"""Synthetic proof acceptance rejects corrupt bytes and misleading controls."""

import copy

import pytest

from skyrl_train.entrypoints.probe_pipeline_timing import BUCKET_BYTES, BUCKET_COUNT, validate_pair


def receipt_rows():
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
                    "rank": rank,
                    "manifest_id": "one-manifest",
                    "identity": {"gpu_uuid": f"gpu-{rank}"},
                    "operation_complete": True,
                    "timing": {"events_complete": True, "intervals": intervals},
                    "installed_bytes": BUCKET_BYTES * BUCKET_COUNT if rank else 0,
                    "mismatched_bytes": 0,
                }
            )
    return rows


def test_complete_serial_and_concurrent_controls():
    validate_pair(receipt_rows())


@pytest.mark.parametrize(
    "failure", ["bytes", "missing", "duplicate", "unfinished", "nonfinite", "serial_overlap", "no_overlap"]
)
def test_rejects_invalid_device_proof(failure):
    rows = copy.deepcopy(receipt_rows())
    if failure == "bytes":
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
    else:
        rows[2]["timing"]["intervals"]["nccl_send"] = rows[0]["timing"]["intervals"]["nccl_send"]
    with pytest.raises((AssertionError, ValueError)):
        validate_pair(rows)
