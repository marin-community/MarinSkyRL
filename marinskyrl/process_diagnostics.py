"""Dependency-light process failure receipts and live stack capture."""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from marinskyrl.runtime_environment import (
    DEBUG_ARTIFACT_DIR_ENV,
    LIVE_STACK_INTERVAL_ENV,
    PYTHONFAULTHANDLER_ENV,
    ensure_debug_artifact_directories,
)


@dataclass(frozen=True)
class ProcessOutcome:
    """Normalized subprocess outcome that preserves signal termination."""

    kind: str
    raw_returncode: int
    public_exit_code: int
    signal: int | None = None
    signal_name: str | None = None

    @classmethod
    def from_returncode(cls, returncode: int) -> "ProcessOutcome":
        if returncode >= 0:
            return cls(kind="exit", raw_returncode=returncode, public_exit_code=returncode)
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIG{signal_number}"
        return cls(
            kind="signal",
            raw_returncode=returncode,
            public_exit_code=128 + signal_number,
            signal=signal_number,
            signal_name=signal_name,
        )


def write_process_outcome(
    role: str,
    returncode: int,
    *,
    pid: int | None = None,
    environment: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ProcessOutcome, Path | None]:
    """Write an atomic, secret-free outcome receipt when diagnostics are active."""
    outcome = ProcessOutcome.from_returncode(returncode)
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    artifact_root = values.get(DEBUG_ARTIFACT_DIR_ENV)
    if not artifact_root:
        return outcome, None
    ensure_debug_artifact_directories(artifact_root)
    process_id = os.getpid() if pid is None else pid
    timestamp_ns = time.time_ns()
    hostname = socket.gethostname()
    path = Path(artifact_root) / "outcomes" / f"{_safe_component(role)}.{hostname}.{process_id}.{timestamp_ns}.json"
    payload = {
        "schema_version": 1,
        "role": role,
        "hostname": hostname,
        "pid": process_id,
        "observed_at_ns": timestamp_ns,
        **asdict(outcome),
        "metadata": dict(metadata or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)
    return outcome, path


_live_stack_files: dict[Path, TextIO] = {}


def enable_fatal_stack_capture(*, environment: Mapping[str, str] | None = None) -> bool:
    """Enable fatal-signal Python tracebacks in an already-running interpreter."""
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    if values.get(PYTHONFAULTHANDLER_ENV) != "1":
        return False
    if not faulthandler.is_enabled():
        faulthandler.enable()
    return True


def install_live_stack_capture(
    role: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Install periodic all-thread dumps when the distributed preset requests them."""
    # Read the manager-projected process contract; this module never defines environment values.
    values = os.environ if environment is None else environment
    raw_interval = values.get(LIVE_STACK_INTERVAL_ENV)
    artifact_root = values.get(DEBUG_ARTIFACT_DIR_ENV)
    if raw_interval is None or artifact_root is None:
        return None
    interval = int(raw_interval)
    if interval <= 0:
        raise ValueError(f"{LIVE_STACK_INTERVAL_ENV} must be positive")
    ensure_debug_artifact_directories(artifact_root)
    path = Path(artifact_root) / "stacks" / f"{_safe_component(role)}.{socket.gethostname()}.{os.getpid()}.log"
    if path in _live_stack_files:
        return path
    output = path.open("a")
    _live_stack_files[path] = output
    faulthandler.dump_traceback_later(interval, repeat=True, file=output)
    return path


def _safe_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_.-" else "-" for character in value)
    return cleaned.strip("-.") or "process"
