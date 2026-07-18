"""Unit tests for ``LiteralLogStore``, the incremental trial-indexed reader over the
shared opencode literal log."""

import json
import os
import sys
import threading

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from terminal_bench.literal_log_store import LiteralLogStore  # noqa: E402


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


def test_transient_read_failure_keeps_cached_entries(tmp_path):
    """A stat/open failure (file transiently gone) returns the already-recovered entries,
    not [] — a filesystem hiccup must not blank out rollout data mid-generation."""
    p = tmp_path / "literal.jsonl"
    _write(p, [_entry("A", 1.0, [10])])
    store = LiteralLogStore()
    assert len(store.entries_for_trial(str(p), "A")) == 1
    os.remove(p)  # now unreadable
    assert len(store.entries_for_trial(str(p), "A")) == 1  # cached view survives
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
