"""Behavior tests for runtime source selection inside an Iris GPU task."""

import contextlib
import signal
import sys
import threading
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
    }


def test_training_driver_requires_the_runtime_checkout() -> None:
    with pytest.raises(RuntimeError, match="SKYRL_HOME"):
        task_runtime.launch_training_driver(["python"], {})


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

    args = SimpleNamespace(
        ray_port=6379,
        ray_log_dir=None,
        rendezvous_dir="s3://incident/rendezvous",
        ray_spill_backend=task_runtime.RaySpillBackend.LOCAL,
        ray_spill_dir=str(tmp_path / "spill"),
        cluster_join_timeout=1,
    )
    monkeypatch.setattr(task_runtime, "FAILURE_ARTIFACT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(task_runtime, "_num_tasks", lambda: 1)
    monkeypatch.setattr(task_runtime, "_own_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(task_runtime, "ray_start_head", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "ray_stop", lambda: None)
    monkeypatch.setattr(task_runtime.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (BlockingFilesystem(), "incident/report"))
    monkeypatch.setattr(task_runtime, "training_driver_env", lambda _ifname: {"SKYRL_HOME": str(tmp_path)})
    monkeypatch.setattr(task_runtime, "ray_metrics_telemetry", lambda *_args: contextlib.nullcontext())

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
