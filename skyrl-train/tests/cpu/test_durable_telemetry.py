import json
from types import SimpleNamespace

import pytest
import zstandard

from cloud.iris.env_vars import ALL_RUNTIME_SCOPES, EnvVarManager
from skyrl_train import durable_telemetry as durable
from skyrl_train import telemetry as training


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_lifecycle_preserves_uid_vector_and_live_export_status(monkeypatch, enabled):
    exporter = training.telemetry
    exporter.shutdown(0)
    delivered = []
    stored = []

    def post(session, endpoint, *, data, headers, timeout):
        if headers.get("Content-Encoding") == "zstd":
            data = zstandard.ZstdDecompressor().decompress(data)
        envelope = json.loads(data)
        delivered.extend(envelope["records"])
        return SimpleNamespace(
            status_code=200, headers={}, json=lambda: {"batch_id": envelope["batch_id"], "status": "accepted"}
        )

    def filesystem(protocol, *, config_kwargs):
        assert enabled, "Default telemetry must never open a durable sink"
        assert protocol == "s3" and config_kwargs["retries"]["max_attempts"] == 0
        return SimpleNamespace(pipe=lambda uri, data: stored.append((uri, json.loads(data))))

    monkeypatch.setattr(exporter.requests.Session, "post", post)
    monkeypatch.setattr(durable.fsspec, "filesystem", filesystem)
    monkeypatch.delenv(durable.PREFIX_ENV, raising=False)
    if enabled:
        monkeypatch.setenv(durable.PREFIX_ENV, "s3://marin-us-east-02a/marin/test-receipts")
    config = training.TelemetryConfig(endpoint="http://finelog.test/v1/ingest", run_id="run", execution_uid="attempt")
    with training.ProcessTelemetry(config, "trainer"):
        for start in range(0, 512, 64):
            training.record_event(
                "consumed_source_order",
                {"prompt_offset": start, "uids_json": json.dumps([f"uid-{n}" for n in range(start, start + 64)])},
                attributes={"step": "1", "role": "trainer"},
            )
        training.record_training_metrics({"policy/correction/m2_m_before": 0.05}, step=4, kind="train")
        assert not stored, "Durable storage belongs to lifecycle exit, not the update path"
    assert not exporter.runtime_status().configured
    assert len(stored) == int(enabled)
    if enabled:
        uri, receipt = stored[0]
        assert uri.startswith("s3://marin-us-east-02a/marin/test-receipts/")
        assert receipt["flush_succeeded"] and receipt["overflow_records"] == 0
        status = receipt["export_status_after_flush_before_shutdown"]
        assert status["configured"] and status["queued_records"] == status["lost_records"] == 0
        assert receipt["post_shutdown_counters_available"] is False
        rows = receipt["records"]
        source = [row for row in rows if row["name"] == "consumed_source_order"]
        assert [row["body"]["prompt_offset"] for row in source] == list(range(0, 512, 64))
        assert [uid for row in source for uid in json.loads(row["body"]["uids_json"])] == [
            f"uid-{n}" for n in range(512)
        ]
        assert any(row["kind"] == "scalar" and row["body"]["value"] == 0.05 for row in rows)
        assert any(row["name"] == "terminal" for row in rows)
        assert receipt["identity"]["execution_uid"] == "attempt"
        assert len(receipt["rigging"]["telemetry_module_sha256"]) == 64
    assert any(row["name"] == "terminal" for row in delivered)


def test_overflow_is_explicit_and_cross_region_prefix_is_rejected(monkeypatch):
    monkeypatch.setattr(durable, "MAX_RECORD_BYTES", 1)
    receipt = durable.DurableTelemetryReceipt("s3://marin-us-east-02a/marin/receipts", {"run_id": "run"})
    receipt.record("event", "consumed_source_order", {"uids_json": '["a"]'}, {})
    assert receipt.rows == [] and receipt.overflow_records == 1
    with pytest.raises(ValueError, match="east"):
        durable.DurableTelemetryReceipt("s3://marin-us-west-02/marin/receipts", {})


def test_explicit_receipt_prefix_reaches_every_runtime_scope():
    prefix = "s3://marin-us-east-02a/marin/receipts"
    manager = EnvVarManager.from_config({}, environ={durable.PREFIX_ENV: prefix})
    for scope in ALL_RUNTIME_SCOPES:
        assert manager.environment_for(scope)[durable.PREFIX_ENV] == prefix
