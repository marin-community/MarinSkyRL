"""Incremental, trial-indexed reader over the shared opencode literal log.

The "literal log" is how an opencode rollout recovers its per-token training signal.
opencode is a CLI agent that talks to vLLM over its own transport and bypasses harbor's
``Chat``, so its ``TrialResult`` comes back with empty ``rollout_details``. A co-located
RecordProxy (external, in harbor) appends one JSON line per served request to a single
shared file at ``$OTAGENT_LITERAL_LOG_PATH``, each line carrying the ``x-ot-trial-id``
correlation id plus the served ``prompt_token_ids`` / ``completion_token_ids`` /
``logprobs``. ``TerminalBenchGenerator`` reads this log, filtered by trial id, to rebuild
rollout_details and chat_history (see the two ``_maybe_*_opencode_*`` helpers).

This module owns the *read* side. The log grows on every append during generation, so a
naive cache re-parses the whole (10s-of-MB) file on every miss — twice per trial across up
to ``n_concurrent_trials`` (288) — an ``O(trials · logsize)`` re-parse that holds the GIL
on the RolloutCoordinator event loop and stalls result draining. ``LiteralLogStore`` reads
only the bytes appended since the last call into a persistent entry list plus a
``{trial_id: [entries]}`` index, so steady-state cost is ``O(new bytes)`` and a per-trial
lookup is ``O(that trial's entries)``. It is thread-safe, handles a record split across
two reads, and resets on file rotation (a preempt-resume serve replaces the file at the
same path → a new inode or a shrink → reset).

The writer (harbor RecordProxy) is external to this repo; this module never writes.
"""

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _LiteralLogState:
    """The incremental read cursor and parsed-entry index for one log path.

    ``offset`` is the byte position already consumed; ``partial`` is a trailing fragment
    of a record split across reads, carried until the rest arrives. ``identity`` is the
    ``(st_dev, st_ino)`` of the file the offset refers to, so a replace-in-place (new
    inode, possibly same/larger size) is detected as rotation, not silently seeked into.
    ``entries`` is every parsed record (objects only) in file order; ``by_trial`` indexes
    them by ``trial_id`` so a single trial's rows are an ``O(own-entries)`` lookup rather
    than an ``O(whole-log)`` scan.
    """

    path: str
    offset: int = 0
    partial: bytes = b""
    identity: Optional[Tuple[int, int]] = None
    entries: List[Dict[str, Any]] = field(default_factory=list)
    by_trial: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class LiteralLogStore:
    """Thread-safe incremental reader over the append-only shared RecordProxy literal log.

    Accessors return a fresh list (a membership snapshot), never the live index list, so a
    caller can iterate its result while another thread appends new records under the lock.
    The list is a shallow copy: the entry dicts inside are shared with the cache, so
    callers must treat returned entries as read-only.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[_LiteralLogState] = None

    def entries_for_trial(self, log_path: str, trial_id: str) -> List[Dict[str, Any]]:
        """This trial's parsed entries in file order; ``[]`` if the trial is absent.

        A fresh list, but the entry dicts are borrowed — read-only.
        """
        with self._lock:
            state = self._refresh(log_path)
            return list(state.by_trial.get(trial_id, ()))

    def all_entries(self, log_path: str) -> List[Dict[str, Any]]:
        """A membership snapshot of every parsed entry (incrementally maintained).

        Diagnostic / test helper: this copies the whole entry list under the lock, so it
        is ``O(all entries)`` and not for the per-trial hot path — prefer
        :meth:`entries_for_trial`. Entries are borrowed — read-only.
        """
        with self._lock:
            state = self._refresh(log_path)
            return list(state.entries)

    def _refresh(self, log_path: str) -> _LiteralLogState:
        """Parse newly-appended bytes into the index. Caller must hold ``self._lock``.

        Always returns the live state for ``log_path`` (never ``None``); on a stat/read
        failure it returns the already-cached state unchanged rather than dropping it, so
        a transient filesystem error does not blank out recovered rollout data. Do not
        leak the returned state's lists — the accessors copy them.
        """
        state = self._state
        if state is None or state.path != log_path:
            state = _LiteralLogState(path=log_path)
            self._state = state

        # Open first, then fstat the descriptor: this reads size + inode for the exact
        # bytes we go on to read (no stat/open TOCTOU) and returns the cached view — not
        # an empty one — if the file is transiently missing/unreadable.
        try:
            fh = open(log_path, "rb")
        except OSError:
            return state
        try:
            stat = os.fstat(fh.fileno())
            identity = (stat.st_dev, stat.st_ino)
            size = stat.st_size
            # Rotation: the file was replaced in place (new inode) or truncated below our
            # cursor (a preempt-resume serve). Either way the byte offset is meaningless →
            # reset and re-read from the start of the new file.
            if state.identity is not None and (identity != state.identity or size < state.offset):
                state = _LiteralLogState(path=log_path)
                self._state = state
            state.identity = identity
            if size <= state.offset:
                return state
            fh.seek(state.offset)
            chunk = fh.read()
        except OSError:
            return state
        finally:
            fh.close()

        state.offset += len(chunk)
        data = state.partial + chunk
        parts = data.split(b"\n")
        # An append can split a record across reads — carry the trailing fragment.
        state.partial = parts.pop()
        for raw in parts:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)  # json.loads accepts utf-8 bytes
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # Every real record is a JSON object; skip a stray scalar/array line so the
            # entry list stays honestly ``List[Dict]`` and consumers can ``entry.get(...)``.
            if not isinstance(entry, dict):
                continue
            state.entries.append(entry)
            trial_id = entry.get("trial_id")
            # trial_id is the string correlation id; ignore a malformed non-string (a
            # list/dict would be unhashable and crash the index build).
            if isinstance(trial_id, str):
                state.by_trial.setdefault(trial_id, []).append(entry)
        return state
