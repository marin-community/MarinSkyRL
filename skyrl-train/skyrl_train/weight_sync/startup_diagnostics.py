"""Attempt-scoped startup evidence for the zero-update diagnostic only."""

from contextlib import contextmanager
import faulthandler
import os
from pathlib import Path
import re
import threading
import time
import traceback

from skyrl_train.weight_sync.readback_diagnostics import persist_readback


class StartupDiagnostics:
    def __init__(self, output_uri: str, role: str):
        identity = os.environ.get("IRIS_ATTEMPT_UID", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
            raise ValueError("Startup diagnostics require an explicit native attempt identity")
        self.output_uri = f"{output_uri.rstrip('/')}/startup/{identity}/{role}"
        root = Path(os.environ.get("OT_AGENT_DEBUG_ARTIFACTS_DIR", "/tmp/skyrl-readback-debug"))
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"faulthandler-{role}-{os.getpid()}.log"
        self.stop = threading.Event()
        self.phase_name = "before_setup"
        self.errors = []

    def phase(self, name: str, **fields) -> None:
        self.phase_name = name
        persist_readback(
            self.output_uri,
            name,
            {
                "phase": name,
                "unix_ns": time.time_ns(),
                "pid": os.getpid(),
                "ray_log_sync_enabled": os.environ.get("OT_AGENT_RAY_LOG_SYNC", "1"),
                "ray_log_sync_interval_seconds": os.environ.get("OT_AGENT_RAY_LOG_SYNC_INTERVAL_S", "300"),
                "runtime_observability_uri": os.environ.get("SKYRL_READBACK_RUNTIME_OBSERVABILITY_URI"),
                **fields,
            },
        )

    def capture_stack(self) -> None:
        if not self.path.exists():
            return
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            stream.seek(max(0, size - 32768))
            raw = stream.read(32768)
        persist_readback(
            self.output_uri,
            "stack",
            {
                "phase": self.phase_name,
                "unix_ns": time.time_ns(),
                "file_bytes": size,
                "truncated": size > len(raw),
                "traceback": raw.decode("utf-8", "replace"),
            },
        )

    def poll(self) -> None:
        while not self.stop.wait(10):
            try:
                self.capture_stack()
            except Exception as error:
                self.errors.append(type(error).__name__)
                self.errors[:] = self.errors[-8:]


@contextmanager
def startup_diagnostics(output_uri: str, role: str):
    diagnostic = StartupDiagnostics(output_uri, role)
    diagnostic.phase("before_setup")
    with diagnostic.path.open("w") as stream:
        faulthandler.enable(file=stream, all_threads=True)
        faulthandler.dump_traceback_later(60, repeat=True, file=stream)
        worker = threading.Thread(target=diagnostic.poll, daemon=True)
        worker.start()
        try:
            yield diagnostic
        except BaseException as error:
            try:
                diagnostic.phase(
                    "python_exception", error_type=type(error).__name__, traceback=traceback.format_exc()[-32768:]
                )
            except Exception as upload_error:
                diagnostic.errors.append(type(upload_error).__name__)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
            diagnostic.stop.set()
            worker.join(timeout=1)
            faulthandler.dump_traceback(file=stream, all_threads=True)
            stream.flush()
            try:
                diagnostic.capture_stack()
                diagnostic.phase("scope_finished", upload_error_types=diagnostic.errors)
            except Exception as upload_error:
                print("READBACK_DIAGNOSTIC_UPLOAD_FAILED " + type(upload_error).__name__, flush=True)
            finally:
                faulthandler.disable()
