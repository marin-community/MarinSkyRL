"""Audit actual CPU protocol receipts; allocation/device fields are emulated."""

import copy
import json
from pathlib import Path

import pytest
import torch

from skyrl_train.weight_sync.bucket_timing_audit import audit_timing_pair, audit_timing_receipts
from skyrl_train.weight_sync.wire_inventory import WireInventory
from skyrl_train.weight_sync.reference_bucket_protocol import begin_reference_sync, finish_reference_sync
from tests.cpu.weight_sync.test_megatron_bucket_protocol import gate as gate_fixture
from tests.cpu.weight_sync.test_worker_bucket_protocol import native_protocol as protocol_fixture
from tests.cpu.weight_sync.test_reference_bucket_protocol import replace_and_load

gate = gate_fixture
native_protocol = protocol_fixture


async def measured_documents(gate, mode):
    if mode == "reference":
        gate.policy.cfg.generator.weight_sync_wire_inventory = True

        async def begin_reference(manifest_id, publication_id):
            return [[begin_reference_sync(gate.receiver.worker, manifest_id, publication_id)]]

        async def finish_reference(manifest_id, publication_id):
            return [[finish_reference_sync(gate.receiver.worker, manifest_id, publication_id)]]

        gate.client.begin_reference_bucket_sync = begin_reference
        gate.client.finish_reference_bucket_sync = finish_reference

        async def original_broadcast(client):
            replace_and_load(gate.receiver)
            inventory = WireInventory()
            for name, tensor in gate.receiver.parts[1].items():
                inventory.observe(name, tensor)
            gate.policy._inventory = inventory.finish(completed_update=gate.policy._completed_update)

        gate.policy.broadcast_to_inference_engines = original_broadcast
        gate.policy.read_weight_sync_wire_inventory = lambda: gate.policy._inventory
        await gate.policy.prepare_reference_timing(gate.client)
    else:
        await gate.policy.prepare_bucket_timing(gate.client)
    for version in (1, 2):
        with torch.no_grad():
            for tensor in gate.receiver.parts[1].values():
                tensor.add_(2)
        gate.policy._completed_update = version
        await gate.policy.begin_bucket_timing(gate.client, version)
        await gate.policy.install_bucket_timing(gate.client, version)
        await gate.policy.replay_bucket_timing(gate.client, version)
    await gate.policy.close_bucket_timing(gate.client, 2)
    documents = {
        path.name: json.loads(path.read_bytes())
        for path in Path(gate.policy.cfg.trainer.weight_sync_readback_output).glob("*.json")
    }
    # CUDA accounting in this CPU fixture reports a constant emulated baseline.
    # Supply only the known two CPU buffer sizes for the native storage gate.
    for name, row in documents.items():
        if name.startswith("bucket-timing-prepared-"):
            row["buffer_allocated_delta"] = 2 * gate.receiver.parts[0].bucket_bytes
    return documents


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["reference", "bucket"])
async def test_actual_two_sync_receipts_pass_and_native_gate_corruptions_fail(gate, monkeypatch, mode):
    monkeypatch.setenv("IRIS_TASK_ID", "cpu-timing-task")
    documents = await measured_documents(gate, mode)
    metrics = [{"step": step, "weight_broadcast": 8.0 if mode == "reference" else 3.0} for step in (1, 2)]

    def audit(value, measured=metrics):
        return audit_timing_receipts(
            value,
            measured,
            mode=mode,
            attempt_uids=["cpu-timing-original-attempt"],
            policy_ranks=1,
            receiver_ranks=1,
            syncs=2,
        )

    result = audit(documents)
    assert result["syncs"] == 2 and result["coverage"] == 1.0 and result["mismatches"] == 0
    assert result["wire_bytes_per_sync"] == sum(part.nbytes for part in gate.receiver.parts[0].entries)
    replay_name = next(name for name in documents if name.startswith("bucket-replay-receivers-sync-2-"))
    begin_name = next(name for name in documents if name.startswith("bucket-timing-begin-sync-2-"))
    preparation_name = next(name for name in documents if name.startswith("bucket-timing-prepared-"))
    faults = []
    missing = copy.deepcopy(documents)
    del missing[replay_name]
    faults.append(missing)
    stale = copy.deepcopy(documents)
    stale[replay_name]["receivers"][0]["publication_id"] = 1
    faults.append(stale)
    corrupt = copy.deepcopy(documents)
    corrupt[replay_name]["receivers"][0]["mismatches"] = 1
    faults.append(corrupt)
    memory = copy.deepcopy(documents)
    memory[replay_name]["receivers"][0]["replay_peak_extra_bytes"] = 1048577
    faults.append(memory)
    restart = copy.deepcopy(documents)
    restart[begin_name]["identity"]["pid"] += 1
    faults.append(restart)
    environment = copy.deepcopy(documents)
    environment[preparation_name]["prepared_receivers"][0]["environment"]["values"]["VLLM_BATCH_INVARIANT"] = "1"
    faults.append(environment)
    source = copy.deepcopy(documents)
    source[preparation_name]["source_byte_coverage"] = 0.5
    faults.append(source)
    for bad in faults:
        with pytest.raises(ValueError):
            audit(bad)
    with pytest.raises(ValueError, match="timing coverage"):
        audit(documents, metrics[:1])
    with pytest.raises(ValueError, match="Invalid driver"):
        audit(documents, [{"step": 1, "weight_broadcast": float("nan")}, metrics[1]])
    other = copy.deepcopy(result)
    other["mode"] = "bucket" if mode == "reference" else "reference"
    reference, candidate = (result, other) if mode == "reference" else (other, result)
    paired = audit_timing_pair(reference, candidate, expected_syncs=2)
    assert paired["full_byte_proof"] and paired["equal_wire_bytes"]
    assert paired["historical_timing_pass"] == (candidate["weight_broadcast_p50_seconds"] <= 7.477672202046961)
    slow = copy.deepcopy(candidate)
    slow["weight_broadcast_seconds"] = [8.0, 9.0]
    slow["weight_broadcast_p50_seconds"] = 8.5
    assert not audit_timing_pair(reference, slow, expected_syncs=2)["historical_timing_pass"]
    other["wire_bytes_per_sync"] += 2
    with pytest.raises(ValueError, match="wire_bytes"):
        audit_timing_pair(result, other, expected_syncs=2) if mode == "reference" else audit_timing_pair(
            other, result, expected_syncs=2
        )
