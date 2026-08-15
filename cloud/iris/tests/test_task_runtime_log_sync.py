import threading

from cloud.iris import task_runtime


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
