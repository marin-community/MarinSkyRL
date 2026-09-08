"""CPU receipt and process-lifecycle checks; these do not qualify NCCL transport."""

import json
import os
import sys
import socket

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


@pytest.mark.parametrize("corrected", [False, True])
def test_real_torchrun_fresh_tcp_rendezvous(tmp_path, corrected):
    """Reproduce the no-server failure, then exercise the GPU harness fix on CPU."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    script = tmp_path / "tcp_probe.py"
    script.write_text(f"""
import os
from datetime import timedelta
import torch.distributed as dist
import torch
from tests.gpu.diagnostics.weight_sync_env_probe import prepare_cross_world_rendezvous
from skyrl_train.distributed.utils import init_custom_process_group
assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "True"
if {corrected!r}:
    prepare_cross_world_rendezvous()
rank = int(os.environ["RANK"])
dist.init_process_group("gloo", init_method="file://{tmp_path}/default-"+str(rank), rank=0, world_size=1)
group = init_custom_process_group("gloo", init_method="tcp://127.0.0.1:{port}", rank=rank, world_size=2, group_name="probe", timeout=timedelta(seconds=1))
tensor = torch.tensor([rank + 1])
dist.all_reduce(tensor, group=group)
assert tensor.item() == 3
dist.destroy_process_group(group)
dist.destroy_process_group()
print("TCP_RENDEZVOUS_PASS", rank, flush=True)
""")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2", str(script)]
    receipt = run_case("baseline", tmp_path / "gang", command=command, timeout_seconds=30)
    assert receipt["reaped"]
    if corrected:
        assert receipt["returncode"] == 0, receipt.get("worker_output", receipt)
        assert receipt["worker_output"].count("TCP_RENDEZVOUS_PASS") == 2
    else:
        assert receipt["returncode"] != 0
        assert "DistNetworkError" in receipt["worker_output"]
