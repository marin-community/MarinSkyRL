"""Exercise durable pre-setup, failure preservation, and actual runtime log paths."""

import json
from types import SimpleNamespace

import pytest

from cloud.iris.task_runtime import persist_readback_runtime_observability
from skyrl_train.weight_sync.startup_diagnostics import startup_diagnostics


def test_pre_setup_and_failure_are_durable_and_attempt_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "attempt-a")
    monkeypatch.setenv("OT_AGENT_DEBUG_ARTIFACTS_DIR", str(tmp_path / "debug"))
    with pytest.raises(RuntimeError, match="original causal failure"):
        with startup_diagnostics(str(tmp_path / "out"), "entrypoint") as diagnostic:
            assert list((tmp_path / "out/startup/attempt-a/entrypoint").glob("before_setup-*.json"))
            diagnostic.phase("trainer_setup_started")
            raise RuntimeError("original causal failure")
    files = list((tmp_path / "out/startup/attempt-a/entrypoint").glob("*.json"))
    receipts = [json.loads(path.read_text()) for path in files]
    failure = next(row for row in receipts if row.get("error_type"))
    assert failure["error_type"] == "RuntimeError"
    assert "original causal failure" in failure["traceback"]
    stack = next(row for row in receipts if "file_bytes" in row)
    assert stack["file_bytes"] > 0 and "test_startup_diagnostics" in stack["traceback"]


def test_failure_upload_does_not_replace_original_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "attempt-b")
    monkeypatch.setenv("OT_AGENT_DEBUG_ARTIFACTS_DIR", str(tmp_path / "debug"))
    with pytest.raises(KeyError, match="causal"):
        with startup_diagnostics(str(tmp_path / "out"), "entrypoint"):

            def failed_upload(*args, **kwargs):
                raise OSError("object store unavailable")

            monkeypatch.setattr("skyrl_train.weight_sync.startup_diagnostics.persist_readback", failed_upload)
            raise KeyError("causal")


def test_runtime_receipt_reports_actual_resolved_log_destination(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYRL_READBACK_STARTUP_DIAGNOSTICS", "1")
    monkeypatch.setenv("IRIS_ATTEMPT_UID", "native-attempt")
    monkeypatch.setenv("OT_AGENT_RAY_LOG_SYNC", "1")
    monkeypatch.setenv("OT_AGENT_RAY_LOG_SYNC_INTERVAL_S", "15")
    args = SimpleNamespace(rendezvous_dir=str(tmp_path / "rendezvous"), ray_log_dir=str(tmp_path / "separate-ray-root"))
    persist_readback_runtime_observability(args, "rank0-host")
    path = tmp_path / "rendezvous/runtime_observability/native-attempt/rank0-host.json"
    receipt = json.loads(path.read_text())
    assert receipt["ray_log_destination"] == str(tmp_path / "separate-ray-root/rank0-host")
    assert receipt["ray_log_sync_enabled"] is True
    assert receipt["ray_log_sync_interval_seconds"] == 15
    assert "WANDB_API_KEY" not in receipt
