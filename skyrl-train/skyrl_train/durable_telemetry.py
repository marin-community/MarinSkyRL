"""Opt-in, bounded diagnostic receipts independent of the telemetry query service."""

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import fsspec

PREFIX_ENV = "SKYRL_DURABLE_TELEMETRY_PREFIX"
MAX_RECORD_BYTES = 768 * 1024
EVENTS = frozenset({"consumed_source_order", "policy_update", "consumed_age", "terminal"})


class DurableTelemetryReceipt:
    def __init__(self, prefix: str, identity: dict[str, str]):
        if not prefix.startswith("s3://marin-us-east-02a/marin/"):
            raise ValueError("Durable telemetry receipts require the east Marin bucket")
        self.identity = {**identity, "pid": str(os.getpid()), "started_ns": str(time.time_ns())}
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.uri = prefix.rstrip("/") + "/" + digest + ".json"
        self.rows: list[str] = []
        self.bytes = 0
        self.overflow_records = 0
        self._lock = threading.Lock()

    def record(self, kind: str, name: str, body: dict, attributes: dict) -> None:
        if kind == "event" and name not in EVENTS:
            return
        encoded = json.dumps(
            {"kind": kind, "name": name, "body": body, "attributes": attributes},
            separators=(",", ":"),
            allow_nan=False,
        )
        size = len(encoded.encode()) + 1
        with self._lock:
            if self.bytes + size > MAX_RECORD_BYTES:
                self.overflow_records += 1
                return
            self.rows.append(encoded)
            self.bytes += size

    def finish(self, *, export_status, flush_succeeded: bool, outcome: str) -> str:
        import rigging.telemetry

        distribution = metadata.distribution("marin-rigging")
        with self._lock:
            records = [json.loads(row) for row in self.rows]
            overflow_records = self.overflow_records
            record_bytes = self.bytes
        payload = {
            "schema_version": 1,
            "identity": self.identity,
            "outcome": outcome,
            "records": records,
            "overflow_records": overflow_records,
            "record_bytes": record_bytes,
            "flush_succeeded": flush_succeeded,
            "export_status_after_flush_before_shutdown": asdict(export_status),
            "post_shutdown_counters_available": False,
            "rigging": {
                "version": distribution.version,
                "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
                "telemetry_module_sha256": hashlib.sha256(Path(rigging.telemetry.__file__).read_bytes()).hexdigest(),
            },
        }
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        if len(data) > 1024 * 1024:
            raise ValueError("Durable telemetry receipt exceeds the one-MiB artifact limit")
        filesystem = fsspec.filesystem(
            "s3", config_kwargs={"connect_timeout": 5, "read_timeout": 5, "retries": {"max_attempts": 0}}
        )
        filesystem.pipe(self.uri, data)
        return hashlib.sha256(data).hexdigest()
