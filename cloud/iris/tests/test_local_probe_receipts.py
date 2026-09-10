"""Exercise the pinned Iris Task/Attempt log boundary with retained transport responses."""

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from iris.client.client import IrisClient, TaskLogEntry
from iris.cluster.types import JobName
from iris.resources.state import TaskState
from cloud.iris.local_probe_receipts import CHUNK_PREFIX, EVENT_PREFIX, LocalProbeReceipts, encode_event


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    client = object.__new__(IrisClient)
    rows = []
    history = [SimpleNamespace(attempt_number=0, attempt_uid="0" * 16, state=TaskState.RUNNING)]
    status = SimpleNamespace(attempts=history, current_attempt_number=0)
    monkeypatch.setattr(client, "task_status", lambda *args, **kwargs: status)
    calls = []

    def fetch(task, query, *, attempt_id):
        calls.append((str(task), query.max_lines, query.substring, attempt_id))
        return list(rows)

    monkeypatch.setattr(client, "_fetch_logs", fetch)
    recorder = LocalProbeReceipts(client, "/atqamar/probe/0:0", "0" * 16, "a" * 40, tmp_path)
    return recorder, rows, status, calls


def append(rows, recorder, phase, **values):
    rows.append(
        TaskLogEntry(
            None,
            JobName.from_wire("/atqamar/probe/0"),
            "stdout",
            encode_event(recorder.identity, phase, **values),
            recorder.number,
        )
    )


def retry(recorder, status):
    status.attempts[0].state = TaskState.FAILED
    status.attempts.append(SimpleNamespace(attempt_number=1, attempt_uid="1" * 16, state=TaskState.RUNNING))
    status.current_attempt_number = 1
    recorder.number = 1
    recorder.identity.update(attempt_id=1, attempt_uid="1" * 16)


def test_explicit_startup_failure_admits_retry_through_actual_attempt_logs(receipts):
    recorder, rows, status, calls = receipts
    append(rows, recorder, "startup-failed")
    retry(recorder, status)
    recorder.require_unmeasured()
    assert calls == [("/atqamar/probe/0", 64, EVENT_PREFIX, 0)]


@pytest.mark.parametrize("fault", ["empty", "measured", "uid", "source", "truncated", "live", "missing"])
def test_retry_fails_closed_on_measured_or_unknown_prior_attempt(receipts, fault):
    recorder, rows, status, calls = receipts
    if fault == "measured":
        append(rows, recorder, "measurement-started")
    if fault != "empty":
        append(rows, recorder, "startup-failed")
    if fault in ("uid", "source"):
        key = "attempt_uid" if fault == "uid" else "source_commit"
        payload = {**recorder.identity, key: "b" * (16 if fault == "uid" else 40)}
        rows[0] = TaskLogEntry(None, rows[0].task_id, "stdout", encode_event(payload, "startup-failed"), 0)
    if fault == "truncated":
        rows *= 64
    retry(recorder, status)
    if fault == "live":
        status.attempts[0].state = TaskState.RUNNING
    if fault == "missing":
        status.attempts.pop(0)
    with pytest.raises(ValueError):
        recorder.require_unmeasured()


def test_acknowledges_exact_controller_marker_before_measurement(receipts, monkeypatch):
    recorder, rows, status, calls = receipts
    original = recorder.event

    def captured(phase):
        expected = original(phase)
        append(rows, recorder, phase)
        return expected

    monkeypatch.setattr(recorder, "event", captured)
    recorder.mark({"groups": []})
    assert recorder.marked
    assert calls == [("/atqamar/probe/0", 64, EVENT_PREFIX, 0)]
    recorder.mark({})
    assert len(calls) == 1


def test_marker_query_error_does_not_become_startup_only(receipts, monkeypatch):
    recorder, rows, status, calls = receipts
    monkeypatch.setattr(recorder, "acknowledge", lambda event: (_ for _ in ()).throw(OSError("unavailable")))
    with pytest.raises(OSError):
        recorder.mark({})
    assert recorder.marked


def test_missing_current_marker_ack_is_not_measurement_permission(receipts):
    recorder, rows, status, calls = receipts
    with pytest.raises(TimeoutError, match="did not acknowledge"):
        recorder.acknowledge({**recorder.identity, "phase": "measurement-started"}, timeout=0)


def test_local_receipt_chunks_reassemble_exact_unicode_bytes(receipts, capsys):
    recorder, rows, status, calls = receipts
    row = {"text": "é雪" * 5000}
    receipt = recorder.persist("result", row)
    chunks = [json.loads(line.removeprefix(CHUNK_PREFIX)) for line in capsys.readouterr().out.splitlines()]
    assert len(chunks) == receipt["chunks"] > 1
    assert [part["index"] for part in chunks] == list(range(len(chunks)))
    raw = b"".join(base64.b64decode(part["base64"], validate=True) for part in chunks)
    assert len(raw) == receipt["bytes"]
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    assert json.loads(raw) == row
    assert recorder.output.joinpath("result.json").read_bytes() == raw
    with pytest.raises(ValueError, match="overwritten"):
        recorder.persist("result", row)


@pytest.mark.parametrize("failure", [None, "before-marker", "after-marker"])
def test_actual_local_matrix_retains_attempt_scope_before_interpretation(receipts, monkeypatch, failure):
    from skyrl_train.entrypoints import probe_shard_groups_local as entry

    recorder, rows, status, calls = receipts
    original = recorder.event

    def event(phase, **values):
        result = original(phase, **values)
        append(rows, recorder, phase, **values)
        return result

    monkeypatch.setattr(recorder, "event", event)
    cases = []

    def run(schedule, backend, output, mark, *, two_hosts):
        cases.append(len(schedule.receiver_global_ranks))
        assert backend == "nccl" and two_hosts
        state = {"groups": [{"readiness": {"phase": "groups-ready"}}], "error": None}
        if failure != "before-marker":
            mark(state)
        if failure:
            state["error"] = failure
        return state

    monkeypatch.setattr(entry, "run_group_probe", run)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            entry.run_matrix(recorder, recorder.output)
        assert cases == [2]
        assert not (recorder.output / "complete.json").exists()
        assert (recorder.output / "receivers-2.json").exists()
        assert ('"phase":"startup-failed"' if failure == "before-marker" else '"phase":"measurement-failed"') in rows[
            -1
        ].data
    else:
        entry.run_matrix(recorder, recorder.output)
        assert cases == [2, 4]
        complete = json.loads((recorder.output / "complete.json").read_bytes())
        assert complete["attempt_uid"] == "0" * 16
        assert len(complete["receipts"]) == 2
        assert calls == [("/atqamar/probe/0", 64, EVENT_PREFIX, 0)]


@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "byte", "uid"])
def test_native_chunk_decoder_rejects_partial_mixed_or_changed_bytes(receipts, capsys, fault):
    from cloud.iris.local_probe_receipts import decode_receipt_chunks

    recorder, rows, status, calls = receipts
    recorder.persist("matrix", {"values": list(range(2000))})
    lines = capsys.readouterr().out.splitlines()
    original = (recorder.output / "matrix.json").read_bytes()
    if fault == "missing":
        lines.pop()
    elif fault == "duplicate":
        lines.append(lines[0])
    elif fault in ("byte", "uid"):
        row = json.loads(lines[0][len(CHUNK_PREFIX) :])
        if fault == "uid":
            row["attempt_uid"] = "f" * 16
        else:
            row["base64"] = base64.b64encode(b"corrupted").decode()
        lines[0] = CHUNK_PREFIX + json.dumps(row)
    if fault:
        with pytest.raises(ValueError):
            decode_receipt_chunks(lines, recorder.identity)
    else:
        assert decode_receipt_chunks(lines, recorder.identity) == {"matrix": original}
