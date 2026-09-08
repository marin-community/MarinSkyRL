"""CPU receipt and process-lifecycle checks; these do not qualify NCCL transport."""

import json
import os
import sys

import pytest

from tests.gpu.diagnostics.weight_sync_env_probe import MARKER, PAYLOAD_BYTES, run_case


@pytest.mark.parametrize("corrupt", [False, True])
def test_zero_exit_requires_matching_payload_receipts(tmp_path, corrupt):
    events = [
        {
            "stage": "broadcast_end",
            "rank": rank,
            "payload_bytes": size,
            "sha256": "corrupted" if corrupt and rank == 1 else "expected",
            "expected_sha256": "expected",
        }
        for rank in (0, 1)
        for size in PAYLOAD_BYTES
    ] + [{"stage": "completed", "rank": rank} for rank in (0, 1)]
    code = "\n".join(f"print({MARKER + json.dumps(event)!r}, flush=True)" for event in events)
    receipt = run_case("baseline", tmp_path / "case", command=[sys.executable, "-c", code], timeout_seconds=5)
    assert receipt["passed"] is not corrupt
    assert receipt["reaped"] is True
    assert json.loads((tmp_path / "case" / "receipt.json").read_text()) == receipt


def test_timeout_retains_partial_stage_and_reaps_worker(tmp_path):
    code = f"import threading; print({MARKER + json.dumps({'stage': 'broadcast_start', 'rank': 0})!r}, flush=True); threading.Event().wait()"
    receipt = run_case("receiver_only", tmp_path / "case", command=[sys.executable, "-c", code], timeout_seconds=0.3)
    assert receipt["passed"] is False
    assert receipt["timeout"] is True
    assert receipt["events"] == [{"stage": "broadcast_start", "rank": 0}]
    assert receipt["reaped"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(receipt["leader_pid"], 0)
