"""Bounded lifecycle helpers for disposable torchrun gangs."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def kill_and_reap_torchrun(process: subprocess.Popen[str], timeout_seconds: float) -> bool:
    """Kill a torchrun process group and report whether its leader was reaped."""

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The gang may exit between poll() and killpg(); wait() still reaps
            # its leader and establishes the cleanup result.
            pass
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def read_process_output(log_path: Path) -> str:
    """Read captured worker output, tolerating partial non-UTF-8 writes."""

    return log_path.read_text(errors="replace") if log_path.exists() else ""
