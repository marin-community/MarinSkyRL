import hashlib
import json

import pytest
from rigging import telemetry

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.telemetry import record_event
from skyrl_train.weight_sync.publication_accounting import PublicationRequestAccounting
from skyrl_train.weight_sync.publication_receipts import MAX_RECEIPT_BYTES, publication_receipt_fields


class RecordingRuntime:
    def __init__(self):
        self.records = []
        self.lost = 0

    def max_record_bytes(self):
        return 1 << 20

    def emit(self, data):
        self.records.append(json.loads(data))
        return True

    def count_lost(self):
        self.lost += 1


@pytest.mark.parametrize("requests", [32, 256, 2048])
def test_populated_native_ledger_survives_actual_telemetry_serializer(monkeypatch, requests):
    ledger = PublicationRequestAccounting()
    for index in range(requests):
        request_id = f"{index:032x}"
        ledger.start(request_id)
        ledger.finish(
            request_id, reason="length", tokens=2048, first_token_time=100.25, policy_version_at_first_token=2
        )
    state = {"host": "engine-雪", "request_accounting": ledger.drain()}
    runtime = RecordingRuntime()
    monkeypatch.setattr(telemetry, "_runtime", runtime)
    fields = publication_receipt_fields(state)
    assert len(fields) > 1
    for part in fields:
        record_event("publication_request_accounting", part, attributes={"moment": "before_pause"})
    assert runtime.lost == 0 and len(runtime.records) == len(fields)
    bodies = [record["body"] for record in runtime.records]
    assert [part["part_index"] for part in bodies] == list(range(len(bodies)))
    assert {part["part_count"] for part in bodies} == {len(bodies)}
    payload = "".join(part["receipt_json"] for part in bodies).encode("ascii")
    assert all(len(part["receipt_json"].encode()) <= 3072 for part in bodies)
    assert {part["receipt_bytes"] for part in bodies} == {len(payload)}
    assert {part["receipt_sha256"] for part in bodies} == {hashlib.sha256(payload).hexdigest()}
    assert json.loads(payload) == state


def test_small_receipt_preserves_exact_state():
    state = {"request_accounting": {"active_ids": []}}
    parts = publication_receipt_fields(state)
    assert len(parts) == 1 and json.loads(parts[0]["receipt_json"]) == state


@pytest.mark.asyncio
async def test_actual_driver_hook_emits_all_parts_through_real_serializer(monkeypatch):
    state = {
        "shared_time_and_uts_namespaces": True,
        "paused": True,
        "request_accounting": {"started_ids": [f"{index:032x}" for index in range(256)]},
    }

    class Client:
        async def read_publication_request_state(self, **kwargs):
            assert kwargs["drain_accounting"] is True
            return [state]

        def publication_inflight_snapshot(self):
            return (256,)

    driver = object.__new__(FullyAsyncRayPPOTrainer)
    driver.inference_engine_client = Client()
    driver.global_step = 1
    runtime = RecordingRuntime()
    monkeypatch.setattr(telemetry, "_runtime", runtime)
    await driver._record_publication_requests("after_pause")
    assert runtime.lost == 0 and len(runtime.records) > 1
    assert all(
        record["attributes"] == {"role": "trainer", "step": "1", "engine_index": "0", "moment": "after_pause"}
        for record in runtime.records
    )
    payload = "".join(record["body"]["receipt_json"] for record in runtime.records)
    assert json.loads(payload) == state


@pytest.mark.parametrize("state", [{"large": "x" * MAX_RECEIPT_BYTES}, {"nonfinite": float("nan")}])
def test_invalid_receipt_fails_before_emission(state):
    with pytest.raises(ValueError):
        publication_receipt_fields(state)
