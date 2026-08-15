"""Unit tests for ``LiteralLogStore``, the incremental trial-indexed reader over the
shared opencode literal log."""

import json
import os
import sys
import threading

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from skyrl_train.trajectory_runners.harbor.literal_log_store import LiteralLogStore  # noqa: E402


def _entry(trial_id, ts, cids):
    return {
        "timestamp": ts,
        "status_code": 200,
        "trial_id": trial_id,
        "request": {"messages": [{"role": "user", "content": "task"}]},
        "literal": {"prompt_token_ids": [1], "completion_token_ids": cids, "logprobs": [-0.1] * len(cids)},
    }


def _write(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(path)


def test_incremental_append_advances_offset_without_reparse(tmp_path):
    """A second call after new lines are appended returns ALL entries, and the byte offset
    advances — the tail is read incrementally, not re-parsed from the top."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()

    first = store.all_entries(str(p))
    assert len(first) == 1
    off1 = store._state.offset
    assert off1 == os.path.getsize(p)

    with open(p, "a") as fh:
        fh.write(json.dumps(_entry("A", 2.0, [11])) + "\n")
    second = store.all_entries(str(p))
    assert [e["literal"]["completion_token_ids"] for e in second] == [[10], [11]]
    assert store._state.offset == os.path.getsize(p) > off1


def test_entries_for_trial_matches_full_scan(tmp_path):
    """entries_for_trial returns exactly the trial-scoped subset a full-scan filter would."""
    entries = [
        _entry("A", 1.0, [10]),
        _entry("B", 1.1, [20]),
        _entry("A", 2.0, [11]),
        _entry("B", 2.1, [21]),
    ]
    p = tmp_path / "literal.jsonl"
    _write(p, entries)
    store = LiteralLogStore()
    all_entries = store.all_entries(str(p))
    for tid in ("A", "B", "Z"):
        assert store.entries_for_trial(str(p), tid) == [e for e in all_entries if e.get("trial_id") == tid]
    assert [e["literal"]["completion_token_ids"] for e in store.entries_for_trial(str(p), "A")] == [[10], [11]]


def test_partial_trailing_record_carried_until_complete(tmp_path):
    """A record split across two reads (append mid-line, no newline yet) is carried as a
    partial and parsed once complete — never dropped, never double-counted."""
    p = tmp_path / "literal.jsonl"
    full = json.dumps(_entry("A", 1.0, [10]))
    half = len(full) // 2
    p.write_text(full[:half])  # incomplete, no newline
    store = LiteralLogStore()
    assert store.all_entries(str(p)) == []
    with open(p, "a") as fh:
        fh.write(full[half:] + "\n")
    out = store.all_entries(str(p))
    assert len(out) == 1 and out[0]["trial_id"] == "A"


def test_missing_final_newline_defers_last_record(tmp_path):
    """The RecordProxy flushes a line before its trailing newline: the last record stays
    partial until the newline lands, then appears (no loss)."""
    p = tmp_path / "literal.jsonl"
    p.write_text(json.dumps(_entry("A", 1.0, [10])))  # no trailing newline
    store = LiteralLogStore()
    assert store.entries_for_trial(str(p), "A") == []
    with open(p, "a") as fh:
        fh.write("\n")
    assert len(store.entries_for_trial(str(p), "A")) == 1


def test_shrink_resets_and_rereads(tmp_path):
    """A truncation below the cursor (same path rewritten smaller) resets the reader
    instead of returning stale/garbled state."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10]), _entry("A", 2.0, [11])])
    store = LiteralLogStore()
    assert len(store.all_entries(str(p))) == 2
    _write(p, [_entry("A", 9.0, [99])])  # rewrite smaller
    out = store.all_entries(str(p))
    assert len(out) == 1 and out[0]["literal"]["completion_token_ids"] == [99]


def test_inode_replacement_resets_even_when_not_smaller(tmp_path):
    """Rotation by atomic replace: a NEW file (new inode) of equal-or-greater size at the
    same path must reset — a size-only check would seek into the middle of the new file."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()
    assert len(store.all_entries(str(p))) == 1
    inode_before = os.stat(p).st_ino

    # Replace in place with a >= sized, different-inode file via atomic rename.
    other = tmp_path / "literal.jsonl.new"
    _write(other, [_entry("B", 5.0, [50]), _entry("B", 6.0, [51])])
    os.replace(other, p)
    assert os.stat(p).st_ino != inode_before  # sanity: really a new inode

    out = store.all_entries(str(p))
    assert [e["trial_id"] for e in out] == ["B", "B"]
    assert store.entries_for_trial(str(p), "A") == []  # old trial gone after rotation


def test_transient_read_failure_recovers_without_reset(tmp_path):
    """A transient open failure (file briefly gone) returns [] for the blip, but the
    cursor and index survive — once the file is back, the data returns in full with no
    rotation reset and no re-ingest. (The store retains byte spans, not payloads, so it
    cannot serve rows while the file is unreadable.)"""
    p = tmp_path / "literal.jsonl"
    away = tmp_path / "literal.jsonl.away"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()
    assert len(store.entries_for_trial(str(p), "A")) == 1
    offset_before = store._state.offset

    os.rename(p, away)  # transiently unreadable at the indexed path
    assert store.entries_for_trial(str(p), "A") == []
    assert store._state.offset == offset_before  # cursor not blanked by the hiccup

    os.rename(away, p)  # blip over (same inode)
    assert len(store.entries_for_trial(str(p), "A")) == 1
    assert len(store.all_entries(str(p))) == 1


def test_missing_log_returns_empty(tmp_path):
    store = LiteralLogStore()
    assert store.all_entries(str(tmp_path / "nope.jsonl")) == []
    assert store.entries_for_trial(str(tmp_path / "nope.jsonl"), "A") == []


def test_malformed_lines_skipped(tmp_path):
    """Corrupt JSON and non-object scalar/array lines are skipped; the entry list stays
    honestly List[Dict] and a stray line never crashes the reader."""
    p = tmp_path / "literal.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_entry("A", 1.0, [10])),
                "{not valid json",
                "123",  # scalar
                json.dumps([1, 2, 3]),  # array, not an object
                json.dumps(_entry("A", 2.0, [11])),
            ]
        )
        + "\n"
    )
    store = LiteralLogStore()
    out = store.all_entries(str(p))
    assert [e["trial_id"] for e in out] == ["A", "A"]
    assert all(isinstance(e, dict) for e in out)


def test_non_string_trial_id_not_indexed(tmp_path):
    """A record whose trial_id is a list/number (malformed) is kept in all_entries but not
    indexed — indexing an unhashable id would otherwise crash the whole refresh."""
    p = tmp_path / "literal.jsonl"
    good = _entry("A", 1.0, [10])
    bad = _entry("A", 2.0, [11])
    bad["trial_id"] = ["not", "a", "string"]
    _write(p, [good, bad])
    store = LiteralLogStore()
    assert len(store.all_entries(str(p))) == 2  # both kept
    assert len(store.entries_for_trial(str(p), "A")) == 1  # only the well-formed one indexed


def test_returned_list_is_a_snapshot(tmp_path):
    """The returned list is a fresh list: mutating it, or a later append, does not corrupt
    what a prior caller holds."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()
    held = store.entries_for_trial(str(p), "A")
    held.append("caller-junk")  # mutate the returned list
    with open(p, "a") as fh:
        fh.write(json.dumps(_entry("A", 2.0, [11])) + "\n")
    fresh = store.entries_for_trial(str(p), "A")
    assert len(fresh) == 2  # new append visible
    assert "caller-junk" not in fresh  # prior caller's mutation didn't leak into the cache


def test_path_change_resets_state(tmp_path):
    """Pointing the store at a different path resets the cursor and reads the new file."""
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write(p1, [_entry("A", 1.0, [10])])
    _write(p2, [_entry("B", 1.0, [20]), _entry("B", 2.0, [21])])
    store = LiteralLogStore()
    assert len(store.all_entries(str(p1))) == 1
    assert [e["trial_id"] for e in store.all_entries(str(p2))] == ["B", "B"]


def test_ingest_does_not_retain_payload_bytes(tmp_path):
    """Ingesting the shared log must retain an O(1)-per-row index, NOT the row payloads.

    Every RolloutCoordinator ingests the ONE shared log while release_trial only fires
    for the trials THAT process consumes, so any per-payload retention grows without
    bound with the log (measured: 4 coordinators each permanently held the other
    coordinators' ~3/4 of a 98 GB log as parsed objects, ~85 GiB/h of RSS growth).
    Here trial B plays the foreign trial this process never consumes: after ingest, the
    store's retained containers must stay tiny relative to B's payload bytes, while B's
    rows remain fully readable on demand."""
    p = tmp_path / "literal.jsonl"
    rows = [_entry("A", 0.0, [1])]
    for i in range(50):
        e = _entry("B", float(i + 1), [i])
        e["request"]["messages"][0]["content"] = "x" * 20_000  # ~20 KB payload per row
        rows.append(e)
    _write(p, rows)
    file_size = os.path.getsize(p)

    store = LiteralLogStore()
    assert len(store.entries_for_trial(str(p), "A")) == 1  # triggers full ingest

    state = store._state
    seen = set()
    retained = 0
    for container in [state.spans, *state.by_trial.values()]:
        for obj in [container, *container, *(x for span in container for x in span)]:
            if id(obj) not in seen:
                seen.add(id(obj))
                retained += sys.getsizeof(obj)
    assert retained < file_size * 0.05, f"index retained {retained} B of a {file_size} B log"

    # The foreign trial's payloads are still fully recoverable on demand.
    foreign = store.entries_for_trial(str(p), "B")
    assert len(foreign) == 50
    assert foreign[0]["request"]["messages"][0]["content"] == "x" * 20_000


def test_release_trial_frees_rows_from_both_index_and_flat_list(tmp_path):
    """release_trial drops a consumed trial's rows from BOTH by_trial and the flat
    file-order span list, so the trial is gone from every view (bounding the index to
    in-flight + foreign trials)."""
    entries = [_entry("A", 1.0, [10]), _entry("B", 1.1, [20]), _entry("A", 2.0, [11])]
    p = tmp_path / "literal.jsonl"
    _write(p, entries)
    store = LiteralLogStore()
    assert len(store.all_entries(str(p))) == 3
    store.release_trial("A")
    # A is gone from both views; B is untouched.
    assert store.entries_for_trial(str(p), "A") == []
    assert [e["literal"]["completion_token_ids"] for e in store.entries_for_trial(str(p), "B")] == [[20]]
    assert [e["trial_id"] for e in store.all_entries(str(p))] == ["B"]
    assert "A" not in store._state.by_trial


def test_release_trial_fences_out_late_appends(tmp_path):
    """A row appended for a trial AFTER it was released (orphaned opencode still hitting
    the proxy post-timeout) is ignored — a released trial never regrows."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()
    assert len(store.entries_for_trial(str(p), "A")) == 1
    store.release_trial("A")
    with open(p, "a") as fh:
        fh.write(json.dumps(_entry("A", 2.0, [11])) + "\n")
    # The late A row is read past (offset still advances) but not retained.
    assert store.entries_for_trial(str(p), "A") == []
    assert store.all_entries(str(p)) == []
    assert store._state.offset == os.path.getsize(p)


def test_release_trial_idempotent_and_noop_when_unknown(tmp_path):
    """Releasing an unknown trial, or the same trial twice, or before any read, is a
    safe no-op that never raises."""
    store = LiteralLogStore()
    store.release_trial("never-read")  # no state yet
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    assert len(store.all_entries(str(p))) == 1
    store.release_trial("Z")  # unknown trial
    store.release_trial("A")
    store.release_trial("A")  # twice
    assert store.all_entries(str(p)) == []


def test_release_of_one_trial_does_not_disturb_a_concurrent_trial(tmp_path):
    """Interleaved A/B rows: releasing A leaves B's rows and file order intact."""
    entries = [_entry("A", 1.0, [10]), _entry("B", 1.1, [20]), _entry("A", 2.0, [11]), _entry("B", 2.1, [21])]
    p = tmp_path / "literal.jsonl"
    _write(p, entries)
    store = LiteralLogStore()
    store.all_entries(str(p))
    store.release_trial("A")
    assert [e["literal"]["completion_token_ids"] for e in store.entries_for_trial(str(p), "B")] == [[20], [21]]
    assert store.entries_for_trial(str(p), "A") == []


def test_concurrent_access_is_thread_safe(tmp_path):
    """Concurrent readers + a concurrent appender never corrupt the index or raise. The
    log grows while N threads repeatedly read; every read returns a self-consistent list."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 0.0, [0])])
    store = LiteralLogStore()
    errors = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                for tid in ("A", "B"):
                    got = store.entries_for_trial(str(p), tid)
                    assert all(e["trial_id"] == tid for e in got)
        except Exception as exc:  # surface any race as a test failure
            errors.append(exc)

    def appender():
        try:
            with open(p, "a") as fh:
                for i in range(1, 200):
                    tid = "A" if i % 2 else "B"
                    fh.write(json.dumps(_entry(tid, float(i), [i])) + "\n")
                    fh.flush()
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()

    threads = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=appender)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    final = store.all_entries(str(p))
    assert len(final) == 200  # 1 seed + 199 appended, none lost
