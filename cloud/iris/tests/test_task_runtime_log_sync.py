import glob
import threading

from cloud.iris import task_runtime


class _RecordingFilesystem:
    def __init__(self, *, wait_for_parallel_upload: bool = False):
        self._lock = threading.Lock()
        self._parallel_upload = threading.Event()
        self._wait_for_parallel_upload = wait_for_parallel_upload
        self.active_uploads = 0
        self.max_active_uploads = 0

    def put(self, local_path, _remote_path):
        with self._lock:
            self.active_uploads += 1
            self.max_active_uploads = max(self.max_active_uploads, self.active_uploads)
            if self.active_uploads >= 2:
                self._parallel_upload.set()
        if self._wait_for_parallel_upload:
            self._parallel_upload.wait(timeout=1)
        with open(local_path, "rb") as source:
            source.read()
        with self._lock:
            self.active_uploads -= 1


def _ray_log_tree(tmp_path, monkeypatch, *payloads: bytes):
    log_dir = tmp_path / "session_test" / "logs"
    log_dir.mkdir(parents=True)
    paths = []
    for index, payload in enumerate(payloads):
        path = log_dir / f"worker-{index}.out"
        path.write_bytes(payload)
        paths.append(path)
    monkeypatch.setattr(glob, "glob", lambda _pattern: [str(log_dir)])
    return paths


def test_ray_log_sync_uploads_files_concurrently(tmp_path, monkeypatch):
    _ray_log_tree(tmp_path, monkeypatch, b"first", b"second", b"third")
    filesystem = _RecordingFilesystem(wait_for_parallel_upload=True)
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))

    result = task_runtime.sync_ray_session_logs("s3://logs", "node-0", "periodic")

    assert result.uploaded_files == 3
    assert filesystem.max_active_uploads >= 2


def test_ray_log_sync_skips_unchanged_files(tmp_path, monkeypatch):
    first_path, _ = _ray_log_tree(tmp_path, monkeypatch, b"first", b"second")
    filesystem = _RecordingFilesystem()
    monkeypatch.setattr(task_runtime, "fs_and_path", lambda _uri: (filesystem, "bucket/logs"))
    state = task_runtime.RayLogSyncState()

    first = task_runtime.sync_ray_session_logs("s3://logs", "node-0", "periodic", state)
    second = task_runtime.sync_ray_session_logs("s3://logs", "node-0", "periodic", state)
    first_path.write_bytes(b"first changed")
    third = task_runtime.sync_ray_session_logs("s3://logs", "node-0", "final", state)

    assert (first.uploaded_files, first.unchanged_files) == (2, 0)
    assert (second.uploaded_files, second.unchanged_files) == (0, 2)
    assert (third.uploaded_files, third.unchanged_files) == (1, 1)


def test_final_ray_log_sync_completes_inline(monkeypatch):
    calls = []
    monkeypatch.setenv("OT_AGENT_RAY_LOG_FINAL_SYNC_TIMEOUT_S", "1")
    monkeypatch.setattr(task_runtime, "sync_ray_session_logs", lambda *args: calls.append(args))

    task_runtime.sync_ray_session_logs_bounded("s3://logs", "node-0", "complete")

    assert calls == [("s3://logs", "node-0", "complete")]


def test_final_ray_log_sync_timeout_does_not_block_teardown(monkeypatch):
    release = threading.Event()
    messages = []
    monkeypatch.setenv("OT_AGENT_RAY_LOG_FINAL_SYNC_TIMEOUT_S", "0.01")
    monkeypatch.setattr(task_runtime, "sync_ray_session_logs", lambda *_args: release.wait())
    monkeypatch.setattr(task_runtime, "_log", messages.append)

    task_runtime.sync_ray_session_logs_bounded("s3://logs", "node-0", "timeout")
    release.set()

    assert any("continuing teardown with a partial upload" in message for message in messages)
