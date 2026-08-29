"""Behavior tests for runtime source selection inside an Iris GPU task."""

import contextlib
import io
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.iris import task_runtime


def test_training_driver_starts_from_the_immutable_runtime_checkout(monkeypatch) -> None:
    observed: dict[str, object] = {}
    process = object()

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return process

    monkeypatch.setattr(task_runtime.subprocess, "Popen", fake_popen)
    environment = {"SKYRL_HOME": "/app/marinskyrl", "PYTHONPATH": "/app/marinskyrl:/app"}

    launched = task_runtime.launch_training_driver(["python", "-m", "cloud.iris.training_driver"], environment)

    assert launched is process
    assert observed == {
        "argv": ["python", "-m", "cloud.iris.training_driver"],
        "env": environment,
        "cwd": "/app/marinskyrl",
        "start_new_session": True,
        "stdout": task_runtime.subprocess.PIPE,
        "stderr": task_runtime.subprocess.STDOUT,
    }


def test_training_driver_requires_the_runtime_checkout() -> None:
    with pytest.raises(RuntimeError, match="SKYRL_HOME"):
        task_runtime.launch_training_driver(["python"], {})


def test_ray_stop_collapses_cli_process_spam(monkeypatch) -> None:
    noisy_output = "VINFO scripts.py -- Killed `ray::IDLE`\n" * 200

    def fake_run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=noisy_output)

    real_subprocess = task_runtime.subprocess
    monkeypatch.setattr(
        task_runtime,
        "subprocess",
        SimpleNamespace(
            PIPE=real_subprocess.PIPE,
            STDOUT=real_subprocess.STDOUT,
            TimeoutExpired=real_subprocess.TimeoutExpired,
            run=fake_run,
        ),
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        task_runtime.ray_stop()

    assert output.getvalue() == "[task-runtime] Ray stop completed (exit 0)\n"


def test_head_returns_driver_abort_when_failure_artifact_upload_blocks(tmp_path, monkeypatch) -> None:
    upload_started = threading.Event()
    release_upload = threading.Event()

    class BlockingWrite:
        def __enter__(self):
            upload_started.set()
            release_upload.wait()
            return self

        def __exit__(self, *_args):
            return False

        def write(self, _payload):
            return None

    class BlockingFilesystem:
        def open(self, _path, _mode):
            return BlockingWrite()

    args = _runtime_args(tmp_path, rendezvous_dir="s3://incident/rendezvous")
    _isolate_head_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(task_runtime, "FAILURE_ARTIFACT_TIMEOUT", 0.01)
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (BlockingFilesystem(), "incident/report"))

    result = []

    def run_head():
        result.append(task_runtime.run_head(args, [sys.executable, "-c", "import os; os.abort()"]))

    runtime_thread = threading.Thread(target=run_head)
    runtime_thread.start()
    try:
        assert upload_started.wait(timeout=5)
        runtime_thread.join(timeout=1)
        assert result == [-signal.SIGABRT]
    finally:
        release_upload.set()
        runtime_thread.join(timeout=5)


def _runtime_args(
    tmp_path: Path,
    *,
    liveness_timeout: float = 0,
    rendezvous_dir: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        ray_port=6379,
        ray_log_dir=None,
        rendezvous_dir=rendezvous_dir,
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
        driver_liveness_timeout=liveness_timeout,
    )


def _isolate_head_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(task_runtime, "DRIVER_WATCHDOG_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(task_runtime, "FAILURE_ARTIFACT_TIMEOUT", 2)
    monkeypatch.setattr(task_runtime, "_num_tasks", lambda: 1)
    monkeypatch.setattr(task_runtime, "_own_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(task_runtime, "ray_start_head", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "ray_stop", lambda: None)
    monkeypatch.setattr(task_runtime.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "training_driver_env", lambda _ifname: {"SKYRL_HOME": str(tmp_path)})
    monkeypatch.setattr(task_runtime, "ray_metrics_telemetry", lambda *_args: contextlib.nullcontext())


def test_head_kills_a_silent_driver_and_records_the_stall_reason(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    rendezvous_dir = tmp_path / "rendezvous"
    (rendezvous_dir / "term_artifacts").mkdir(parents=True)
    script = """
import os
import signal
import threading

def block_abort(*_args):
    while True:
        print("[fd-monitor] process is still allocated", flush=True)
        threading.Event().wait(0.01)

signal.signal(signal.SIGABRT, block_abort)
print("training started", flush=True)
os.kill(os.getpid(), signal.SIGABRT)
"""

    exit_code = task_runtime.run_head(
        _runtime_args(tmp_path, liveness_timeout=0.1, rendezvous_dir=str(rendezvous_dir)),
        [sys.executable, "-c", script],
    )

    artifacts = list((rendezvous_dir / "term_artifacts").glob("*.txt"))
    assert exit_code == task_runtime.DRIVER_STALLED_EXIT_CODE
    assert len(artifacts) == 1
    assert "driver stalled" in artifacts[0].read_text()


def test_head_keeps_a_driver_alive_while_it_emits_progress(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    script = """
import threading

for phase in range(5):
    print(f"completed phase {phase}", flush=True)
    threading.Event().wait(0.03)
"""

    exit_code = task_runtime.run_head(
        _runtime_args(tmp_path, liveness_timeout=0.05),
        [sys.executable, "-c", script],
    )

    assert exit_code == 0


def test_head_allows_a_silent_driver_when_the_watchdog_is_disabled(tmp_path, monkeypatch) -> None:
    _isolate_head_runtime(tmp_path, monkeypatch)
    script = """
import threading

threading.Event().wait(0.1)
"""

    exit_code = task_runtime.run_head(
        _runtime_args(tmp_path, liveness_timeout=0),
        [sys.executable, "-c", script],
    )

    assert exit_code == 0
