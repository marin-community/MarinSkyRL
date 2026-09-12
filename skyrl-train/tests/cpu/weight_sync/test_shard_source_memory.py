"""Source proof accounting/error paths; CUDA snapshots here are explicit doubles."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.shard_replica_proof import FullReplicaComparator


def comparator(monkeypatch, *, capture, peak=900000):
    instance = object.__new__(FullReplicaComparator)
    instance.transfer = torch.empty(32, dtype=torch.uint8)
    instance.comparison = torch.empty(65536, dtype=torch.bool)
    instance.plan = SimpleNamespace(identity="plan")
    instance.capture = capture
    instance._compare = lambda *args: "complete result"
    snapshots = iter(
        [
            {"cuda_measured": True, "allocated_bytes": 500000},
            {"cuda_measured": True, "peak_allocated_bytes": peak, "allocated_bytes": 500000},
        ]
    )
    monkeypatch.setattr("skyrl_train.weight_sync.shard_replica_proof.device_memory", lambda device: next(snapshots))
    return instance


@pytest.mark.parametrize("peak", [900000, 1500000])
def test_retained_bool_workspace_counts_and_raw_values_persist_before_gate(monkeypatch, peak):
    captured = []
    instance = comparator(monkeypatch, capture=lambda row: captured.append(dict(row)), peak=peak)
    assert instance({}, "manifest", 4, 0) == "complete result"
    assert captured[0]["proof_peak_extra_bytes"] == peak - 500000 + 65536
    assert captured[0]["proof_memory_within_limit"] == (peak - 500000 + 65536 <= 1024 * 1024)
    assert captured[0]["retained_comparison_bytes"] == 65536


def test_source_failure_preserves_memory_and_capture_failure_notes(monkeypatch):
    captured = []

    def capture(row):
        captured.append(dict(row))
        raise RuntimeError("durable write failed")

    instance = comparator(monkeypatch, capture=capture)

    def fail(*args):
        raise ValueError("initiating source proof failure")

    instance._compare = fail
    with pytest.raises(ValueError, match="initiating source proof failure") as caught:
        instance({}, "manifest", 4, 0)
    assert captured[0]["phase"] == "failed"
    assert captured[0]["proof_peak_extra_bytes"] == 465536
    assert any("durable write failed" in note for note in caught.value.__notes__)


def test_successful_source_proof_cannot_hide_durable_capture_failure(monkeypatch):
    def capture(row):
        raise RuntimeError("durable write failed")

    instance = comparator(monkeypatch, capture=capture)
    with pytest.raises(RuntimeError, match="durable write failed"):
        instance({}, "manifest", 4, 0)


def test_session_rejects_completed_source_proof_over_memory_limit():
    from threading import Lock
    from skyrl_train.weight_sync.shard_session import ShardSession, ShardPhase, SourceReplicaProof, storage_versions

    source = {"weight": torch.tensor([1.0, 2.0])}
    session = object.__new__(ShardSession)
    session.lock = Lock()
    session.manifest_id = "manifest"
    session.publication_id = 4
    session.phase = ShardPhase.FROZEN
    session.runner = SimpleNamespace(rank=0, sources=source)
    session.versions = storage_versions(source)

    class Verifier:
        last_receipt = {"proof_memory_within_limit": False}

        def __call__(self, sources, manifest, version, rank):
            return SourceReplicaProof(
                manifest, version, rank, 8, 8, 0, storage_versions(sources), "full-byte-comparison"
            )

    session.replica_verifier = Verifier()
    with pytest.raises(ValueError, match="complete additional scratch limit"):
        session.verify_replicas("manifest", 4)
    assert session.phase is ShardPhase.FAILED
