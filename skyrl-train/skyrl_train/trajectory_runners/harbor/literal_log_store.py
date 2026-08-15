"""Incremental, trial-indexed reader over the shared opencode literal log.

The "literal log" is how an opencode rollout recovers its per-token training signal.
opencode is a CLI agent that talks to vLLM over its own transport and bypasses harbor's
``Chat``, so its ``TrialResult`` comes back with empty ``rollout_details``. A co-located
RecordProxy (external, in harbor) appends one JSON line per served request to a single
shared file at ``$OTAGENT_LITERAL_LOG_PATH``, each line carrying the ``x-ot-trial-id``
correlation id plus the served ``prompt_token_ids`` / ``completion_token_ids`` /
``logprobs``. ``HarborTrajectoryRunner`` reads this log, filtered by trial id, to rebuild
rollout_details and chat_history (see the two ``_maybe_*_opencode_*`` helpers).

This module owns the *read* side. The log grows on every append during generation, so a
naive cache re-parses the whole (10s-of-GB) file on every miss. ``LiteralLogStore`` reads
only the bytes appended since the last call, but it must NOT retain the parsed rows
either: every RolloutCoordinator ingests the ONE shared log, while ``release_trial`` can
only drop the trials THIS process consumed. With K coordinators, the other (K-1)/K of the
log — parsed into Python objects at ~5x the raw bytes — was retained forever (measured:
a 98 GB log became ~1.4 TiB of coordinator RSS across 4 coordinators in 17 h, ~85 GiB/h,
until the driver pod OOMed).

So the index maps ``trial_id -> [(byte offset, length), ...]`` and payloads are parsed
ON DEMAND from the file at ``entries_for_trial`` time. Steady-state ingest cost is still
``O(new bytes)`` (each appended line is json-parsed once for validation + trial_id, then
discarded); memory is ``O(rows)`` at ~100 bytes per row instead of ``O(payload bytes)``.
It is thread-safe, handles a record split across two reads, and resets on file rotation
(a preempt-resume serve replaces the file at the same path -> a new inode or a shrink ->
reset).

The writer (harbor RecordProxy) is external to this repo; this module never writes.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A retained row reference: (byte offset of the raw line, line length in bytes).
_Span = Tuple[int, int]

logger = logging.getLogger(__name__)


@dataclass
class _LiteralLogState:
    """The incremental read cursor and trial-indexed span index for one log path.

    ``offset`` is the byte position already consumed; ``partial`` is a trailing fragment
    of a record split across reads, carried until the rest arrives. ``identity`` is the
    ``(st_dev, st_ino)`` of the file the offset refers to, so a replace-in-place (new
    inode, possibly same/larger size) is detected as rotation, not silently seeked into.
    ``spans`` locates every validated record (objects only) in file order; ``by_trial``
    indexes the spans by ``trial_id``. Only these O(1)-sized spans are retained — the
    parsed row dicts are rebuilt from the file on demand and never cached.
    """

    path: str
    offset: int = 0
    partial: bytes = b""
    identity: Optional[Tuple[int, int]] = None
    spans: List[_Span] = field(default_factory=list)
    by_trial: Dict[str, List[_Span]] = field(default_factory=dict)
    # Trials whose rows have been consumed + dropped (see LiteralLogStore.release_trial).
    # A released trial's spans are pruned from ``spans`` / ``by_trial`` and any LATER
    # appended row for it is ignored on refresh, so a released trial never regrows.
    released: set = field(default_factory=set)


class LiteralLogStore:
    """Thread-safe incremental reader over the append-only shared RecordProxy literal log.

    Accessors parse the requested rows from the file under the lock and return fresh
    dicts, so a caller can hold and even mutate its result without affecting the store or
    other callers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[_LiteralLogState] = None

    def entries_for_trial(self, log_path: str, trial_id: str) -> List[Dict[str, Any]]:
        """This trial's parsed entries in file order; ``[]`` if the trial is absent.

        Parsed fresh from the file on every call (``O(this trial's bytes)``) — the store
        retains only byte spans, never payloads.
        """
        with self._lock:
            state = self._refresh(log_path)
            return self._read_spans(state, state.by_trial.get(trial_id, ()))

    def release_trial(self, trial_id: str) -> None:
        """Drop a fully-consumed trial's spans and fence out its late appends.

        Once a trial's rollout_details + chat_history have been rebuilt (both
        ``_maybe_*_opencode_*`` consumers have run in ``_process_trial_result``), its rows
        are dead weight — release them here.

        Idempotent. Late appends for a released trial (e.g. an orphaned opencode process
        that keeps calling the proxy after harbor timed the trial out) are ignored on the
        next :meth:`_refresh`, so a released trial never silently regrows. A no-op when the
        trial is unknown / already released / no log has been read yet.
        """
        with self._lock:
            state = self._state
            if state is None:
                return
            state.released.add(trial_id)
            dropped = state.by_trial.pop(trial_id, None)
            if dropped:
                drop = set(dropped)
                state.spans = [s for s in state.spans if s not in drop]

    def all_entries(self, log_path: str) -> List[Dict[str, Any]]:
        """Every retained record, parsed fresh from the file in file order.

        Diagnostic / test helper: this re-reads and parses the WHOLE retained span set,
        so it is ``O(all retained bytes)`` and not for the per-trial hot path — prefer
        :meth:`entries_for_trial`.
        """
        with self._lock:
            state = self._refresh(log_path)
            return self._read_spans(state, state.spans)

    def _read_spans(self, state: _LiteralLogState, spans: Sequence[_Span]) -> List[Dict[str, Any]]:
        """Parse the rows the given spans locate. Caller must hold ``self._lock``.

        Returns ``[]`` when the file is unreadable or was rotated since the spans were
        indexed (the next :meth:`_refresh` then resets the state); a row that no longer
        parses is skipped rather than raising.
        """
        if not spans:
            return []
        try:
            fh = open(state.path, "rb")
        except OSError:
            return []
        try:
            stat = os.fstat(fh.fileno())
            if state.identity is not None and (stat.st_dev, stat.st_ino) != state.identity:
                return []
            out: List[Dict[str, Any]] = []
            for start, length in spans:
                fh.seek(start)
                raw = fh.read(length).strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Rows were validated at ingest; a re-parse failure means the span
                    # index is stale (in-place rewrite) or buggy — surface it.
                    logger.warning(
                        "literal-log span at offset %d (len %d) no longer parses; skipping row", start, length
                    )
                    continue
                if isinstance(entry, dict):
                    out.append(entry)
            return out
        except OSError:
            return []
        finally:
            fh.close()

    def _refresh(self, log_path: str) -> _LiteralLogState:
        """Index newly-appended bytes. Caller must hold ``self._lock``.

        Always returns the live state for ``log_path`` (never ``None``); on a stat/read
        failure it returns the already-cached state unchanged rather than dropping it, so
        a transient filesystem error does not lose the index or the cursor.
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

        # The carried partial's first byte sits len(partial) bytes BEFORE the old cursor.
        position = state.offset - len(state.partial)
        state.offset += len(chunk)
        data = state.partial + chunk
        parts = data.split(b"\n")
        # An append can split a record across reads — carry the trailing fragment.
        state.partial = parts.pop()
        for raw_line in parts:
            span: _Span = (position, len(raw_line))
            position += len(raw_line) + 1  # +1 for the newline split off
            raw = raw_line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)  # json.loads accepts utf-8 bytes
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # Every real record is a JSON object; skip a stray scalar/array line so
            # consumers can ``entry.get(...)`` on whatever a span parses back into.
            if not isinstance(entry, dict):
                continue
            trial_id = entry.get("trial_id")
            # A row appended AFTER its trial was released (e.g. an orphaned opencode still
            # calling the proxy post-timeout) is dead weight — drop it so a released trial
            # cannot regrow the index. Checked before retention so nothing is stored.
            if isinstance(trial_id, str) and trial_id in state.released:
                continue
            state.spans.append(span)
            # trial_id is the string correlation id; ignore a malformed non-string (a
            # list/dict would be unhashable and crash the index build). The parsed
            # ``entry`` is discarded here — retaining it is the RolloutCoordinator
            # RSS leak this design exists to prevent.
            if isinstance(trial_id, str):
                state.by_trial.setdefault(trial_id, []).append(span)
        return state
