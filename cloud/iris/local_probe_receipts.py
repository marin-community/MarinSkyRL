"""Keep synthetic receipts local and acknowledge measurement markers through Iris logs."""

import base64
import hashlib
import json
from pathlib import Path
import re
import time

from iris.cluster.types import TaskAttempt
from iris.resources.state import TERMINAL_TASK_STATES

EVENT_PREFIX = "K10_LOCAL_EVENT "
CHUNK_PREFIX = "K10_LOCAL_RECEIPT "
MAX_EVENTS = 64


def encode_event(identity, phase, **values):
    return EVENT_PREFIX + json.dumps({**identity, "phase": phase, **values}, sort_keys=True, separators=(",", ":"))


def attempt_events(handle):
    """Read one bounded attempt, retaining positive identity and completeness checks."""
    rows = handle.logs(max_lines=MAX_EVENTS, substring=EVENT_PREFIX)
    if len(rows) >= MAX_EVENTS:
        raise ValueError("Controller marker query may be truncated")
    events = []
    for row in rows:
        if row.attempt_id != handle.attempt_number or str(row.task_id) != str(handle.task_id):
            raise ValueError("Controller returned another task attempt's marker")
        for line in row.data.splitlines():
            if EVENT_PREFIX in line:
                events.append(json.loads(line.split(EVENT_PREFIX, 1)[1]))
    return events


class LocalProbeReceipts:
    """Separate complete per-attempt evidence; unknown prior logs fail before measurement."""

    def __init__(self, client, task_wire, uid, source_commit, output):
        attempt = TaskAttempt.from_wire(task_wire)
        if (
            attempt.attempt_id is None
            or not re.fullmatch("[0-9a-f]{16}", uid)
            or not re.fullmatch("[0-9a-f]{40}", source_commit)
            or not str(attempt.task_id).endswith("/0")
        ):
            raise ValueError("Local probe needs explicit head attempt and source identity")
        self.task = client.task(attempt.task_id)
        self.number = attempt.attempt_id
        self.identity = {
            "task_id": str(attempt.task_id),
            "attempt_id": self.number,
            "attempt_uid": uid,
            "source_commit": source_commit,
            "class": "X",
        }
        self.output = Path(output) / uid
        self.output.mkdir(parents=True, exist_ok=True)
        self.marked = False

    def require_unmeasured(self):
        status = self.task.status()
        history = {row.attempt_number: row for row in status.attempts}
        if set(history) != set(range(self.number + 1)) or status.current_attempt_number != self.number:
            raise ValueError("Incomplete or changed native attempt history")
        if history[self.number].attempt_uid != self.identity["attempt_uid"]:
            raise ValueError("Native current attempt UID changed")
        for number in range(self.number):
            prior = history[number]
            if prior.state not in TERMINAL_TASK_STATES:
                raise ValueError("Prior native attempt is not terminal")
            events = attempt_events(self.task.attempt(number))
            expected = {**self.identity, "attempt_id": number, "attempt_uid": prior.attempt_uid}
            if any(any(event.get(key) != value for key, value in expected.items()) for event in events):
                raise ValueError("Prior event source or attempt identity differs")
            if any(event.get("phase") == "measurement-started" for event in events):
                raise ValueError("Prior attempt already entered measurement")
            # Absence is insufficient: the prior process must have reported its
            # terminal premeasurement disposition. Crashes without it fail closed.
            if not events or events[-1].get("phase") != "startup-failed":
                raise ValueError("Prior measurement disposition is unknown")

    def event(self, phase, **values):
        line = encode_event(self.identity, phase, **values)
        if len(line.encode()) > 8192:
            raise ValueError("Controller marker exceeds bounded event size")
        print(line, flush=True)
        return json.loads(line[len(EVENT_PREFIX) :])

    def acknowledge(self, expected, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            if expected in attempt_events(self.task.attempt(self.number)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Controller did not acknowledge native measurement marker")
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))

    def mark(self, state):
        if self.marked:
            return
        self.require_unmeasured()
        self.persist("prepared", state)
        # Set before emission: a failed ACK must never become startup-failed.
        self.marked = True
        expected = self.event("measurement-started")
        self.acknowledge(expected)

    def persist(self, name, row):
        if not re.fullmatch(r"[a-z0-9-]{1,64}", name):
            raise ValueError("Local receipt name is invalid")
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > 1024**2:
            raise ValueError("Synthetic receipt exceeds one MiB")
        path = self.output / f"{name}.json"
        if path.exists():
            raise ValueError("A local attempt receipt cannot be overwritten")
        path.write_bytes(payload)
        if path.read_bytes() != payload:
            raise ValueError("Local receipt byte readback differs")
        digest = hashlib.sha256(payload).hexdigest()
        chunks = [payload[offset : offset + 3072] for offset in range(0, len(payload), 3072)]
        for index, chunk in enumerate(chunks):
            print(
                CHUNK_PREFIX
                + json.dumps(
                    {
                        **self.identity,
                        "name": name,
                        "bytes": len(payload),
                        "sha256": digest,
                        "index": index,
                        "count": len(chunks),
                        "base64": base64.b64encode(chunk).decode(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
        return {"path": str(path), "bytes": len(payload), "sha256": digest, "chunks": len(chunks)}


def decode_receipt_chunks(lines, identity):
    """Reassemble complete native bytes from one controller attempt's bounded logs."""
    parts = {}
    metadata = {}
    for line in lines:
        if CHUNK_PREFIX not in line:
            continue
        row = json.loads(line.split(CHUNK_PREFIX, 1)[1])
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError("Receipt chunks mix source or native attempt identities")
        name, index, count = row["name"], row["index"], row["count"]
        if (
            not re.fullmatch(r"[a-z0-9-]{1,64}", name)
            or type(index) is not int
            or type(count) is not int
            or not 0 <= index < count <= 342
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= 1024**2
        ):
            raise ValueError("Receipt chunk geometry exceeds the bounded contract")
        info = (count, row["bytes"], row["sha256"])
        if name in metadata and metadata[name] != info:
            raise ValueError("Receipt metadata changed within one native attempt")
        metadata[name] = info
        bucket = parts.setdefault(name, {})
        if index in bucket:
            raise ValueError("Receipt chunk was duplicated")
        bucket[index] = base64.b64decode(row["base64"], validate=True)
    results = {}
    for name, bucket in parts.items():
        count, size, digest = metadata[name]
        if set(bucket) != set(range(count)):
            raise ValueError("Native receipt is incomplete")
        raw = b"".join(bucket[index] for index in range(count))
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Native receipt byte identity differs")
        results[name] = raw
    return results
