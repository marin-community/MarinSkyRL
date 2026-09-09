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
BUDGETS = {"event": 512 * 1024, "scalar": 240 * 1024, "terminal": 16 * 1024}
EVENTS = frozenset({"consumed_source_order", "policy_update", "consumed_age", "terminal"})
SCALAR_FIELDS = frozenset(
    {
        "policy_update_steps",
        "updates_attempted",
        "updates_completed",
        "updates_completed_valid",
        "update_index",
        "update_age",
        "optimizer_step_succeeded",
        "raw_grad_norm",
        "grad_norm_reduced",
        "grad_norm_valid",
        "stale/selected_tokens",
    }
)


def selected_scalar(name: str) -> bool:
    if name == "consumed/uid_digest_u52":
        return True
    if not name.startswith("policy/"):
        return False
    field = name.removeprefix("policy/")
    if field.startswith("by_update/"):
        parts = field.split("/", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            return False
        field = parts[2]
    return field in SCALAR_FIELDS or field.startswith(("offpolicy_mask/", "m2_mask/"))


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
        self.capture_errors = 0
        self.bytes_by_kind = dict.fromkeys(BUDGETS, 0)
        self.overflow_by_kind = dict.fromkeys(BUDGETS, 0)
        self._lock = threading.Lock()

    def record(self, kind: str, name: str, body: dict, attributes: dict) -> None:
        if kind == "event" and name not in EVENTS:
            return
        if kind == "scalar" and not selected_scalar(name):
            return
        budget_kind = "terminal" if name == "terminal" else kind
        try:
            encoded = json.dumps(
                {"kind": kind, "name": name, "body": body, "attributes": attributes},
                separators=(",", ":"),
                allow_nan=False,
            )
            size = len(encoded.encode()) + 1
        except Exception:
            with self._lock:
                self.capture_errors += 1
            return
        with self._lock:
            if self.bytes_by_kind[budget_kind] + size > BUDGETS[budget_kind]:
                self.overflow_records += 1
                self.overflow_by_kind[budget_kind] += 1
                return
            self.rows.append(encoded)
            self.bytes += size
            self.bytes_by_kind[budget_kind] += size

    def finish(self, *, export_status, flush_succeeded: bool, outcome: str) -> str:
        import rigging.telemetry

        distribution = metadata.distribution("marin-rigging")
        with self._lock:
            records = [json.loads(row) for row in self.rows]
            overflow_records = self.overflow_records
            record_bytes = self.bytes
            capture_errors = self.capture_errors
            bytes_by_kind = dict(self.bytes_by_kind)
            overflow_by_kind = dict(self.overflow_by_kind)
        payload = {
            "schema_version": 1,
            "identity": self.identity,
            "outcome": outcome,
            "records": records,
            "overflow_records": overflow_records,
            "record_bytes": record_bytes,
            "capture_errors": capture_errors,
            "bytes_by_kind": bytes_by_kind,
            "overflow_by_kind": overflow_by_kind,
            "budgets": BUDGETS,
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
