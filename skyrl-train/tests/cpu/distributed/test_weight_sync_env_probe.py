"""CPU receipt and process-lifecycle checks; these do not qualify NCCL transport."""

import json
import os
import sys
import socket

import pytest

from tests.gpu.diagnostics.weight_sync_env_probe import MARKER, PAYLOAD_BYTES, run_case
from tests.gpu.diagnostics import weight_sync_env_probe as probe


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


def test_actual_concatenated_stdout_records_are_all_parsed(tmp_path):
    events = [{"stage": "completed", "rank": rank} for rank in (0, 1)]
    # Two real children write complete event objects without newlines into one pipe.
    code = "import subprocess,sys\nchildren=[]\n"
    for event in events:
        child = f"import os; os.write(1,{(MARKER + json.dumps(event)).encode()!r})"
        code += f"children.append(subprocess.Popen([sys.executable,'-c',{child!r}]))\n"
    code += "assert all(child.wait()==0 for child in children)\n"
    receipt = run_case("baseline", tmp_path / "joined", command=[sys.executable, "-c", code])
    assert {e["rank"] for e in receipt["events"]} == {0, 1}
    assert receipt["parse_errors"] == []
    assert not receipt["passed"]  # Completion lines alone never certify payloads.


def test_raw_receipt_survives_unexpected_parser_failure(tmp_path, monkeypatch):
    def fail_parse(*args):
        raise ValueError("injected parser failure")

    monkeypatch.setattr(probe, "audit_output", fail_parse)
    receipt = run_case("baseline", tmp_path / "case", command=[sys.executable, "-c", "print('unparsed raw evidence')"])
    assert receipt["passed"] is False and receipt["reaped"]
    assert receipt["audit_error"] == "ValueError: injected parser failure"
    raw = json.loads((tmp_path / "case" / "raw-receipt.json").read_text())
    assert raw["worker_output"] == "unparsed raw evidence\n"
    assert json.loads((tmp_path / "case" / "receipt.json").read_text()) == receipt


def test_malformed_event_fails_closed_without_throwing():
    receipt = probe.audit_output(MARKER + '{"stage":bad json}\n', 0)
    assert not receipt["passed"] and receipt["parse_errors"]


def test_parent_owns_store_listener_throughout_child_start_window():
    with probe.held_rendezvous_store() as port:
        with socket.socket() as contender:
            with pytest.raises(OSError):
                contender.bind(("127.0.0.1", port))


def test_full_three_case_owned_store_orchestration_on_cpu(tmp_path):
    receipts = probe.run_comparison(tmp_path / "comparison", backend="gloo")
    assert [receipt["case"] for receipt in receipts] == list(probe.CASES)
    assert all(receipt["passed"] and receipt["reaped"] for receipt in receipts)
    for receipt in receipts:
        assert receipt["backend"] == "gloo"
        assert set(receipt["rank_streams"]) == {"rank-0.jsonl", "rank-1.jsonl"}
        assert len([e for e in receipt["events"] if e["stage"] == "broadcast_end"]) == 6
        assert (tmp_path / "comparison" / receipt["case"] / "raw-receipt.json").exists()


def test_case_setup_error_does_not_drop_later_receipts(tmp_path, monkeypatch):
    def setup_failure(case, *args, **kwargs):
        raise RuntimeError("injected setup error " + case)

    monkeypatch.setattr(probe, "run_case", setup_failure)
    receipts = probe.run_comparison(tmp_path)
    assert [r["case"] for r in receipts] == list(probe.CASES)
    assert all(not r["passed"] and r["orchestration_error"] for r in receipts)
