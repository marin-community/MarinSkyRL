import gc
import os
import threading

import pytest

from skyrl_train.trajectory_runners.trajectory_retention_publisher import (
    ProcessTrajectoryPublisher,
    PublicationOperation,
    PublicationRequest,
    PublicationResult,
)


def _storage_worker(request: PublicationRequest) -> PublicationResult:
    if request.request_id == "stall":
        threading.Event().wait()
    if request.request_id == "crash":
        os._exit(7)
    return PublicationResult(
        request.request_id,
        request.record_count,
        ledger={"pid": os.getpid(), "published": request.request_id},
        error="storage unavailable" if request.request_id == "error" else None,
    )


def _request(identity: str) -> PublicationRequest:
    return PublicationRequest(identity, PublicationOperation.PUBLISH, "/unused", record_count=8)


def test_acknowledged_publications_reuse_storage_process_and_close_stops_it():
    publisher = ProcessTrajectoryPublisher(
        _storage_worker,
        publish_timeout_seconds=5,
        shutdown_timeout_seconds=1,
    )
    try:
        first = publisher.execute(_request("first"))
        second = publisher.execute(_request("second"))
        assert first.error is None and second.error is None
        assert first.ledger["pid"] == second.ledger["pid"]
        assert second.ledger["published"] == "second"
        assert second.record_count == 8
    finally:
        publisher.close()
    with pytest.raises(ProcessLookupError):
        os.kill(first.ledger["pid"], 0)


@pytest.mark.parametrize("failure", ["stall", "crash", "error"])
def test_failed_publication_discards_worker_and_next_operation_recovers(failure):
    publisher = ProcessTrajectoryPublisher(
        _storage_worker,
        publish_timeout_seconds=1,
        shutdown_timeout_seconds=1,
    )
    try:
        first = publisher.execute(_request("first"))
        assert first.error is None
        failed = publisher.execute(_request(failure))
        assert failed.error is not None
        assert failed.timed_out is (failure == "stall")
        with pytest.raises(ProcessLookupError):
            os.kill(first.ledger["pid"], 0)
        recovered = publisher.execute(_request("recovered"))
        assert recovered.error is None
        assert recovered.ledger["published"] == "recovered"
        assert recovered.ledger["pid"] != first.ledger["pid"]
    finally:
        publisher.close()


def test_shutdown_terminates_pending_storage_operation_and_returns_its_failure():
    publisher = ProcessTrajectoryPublisher(
        _storage_worker,
        publish_timeout_seconds=10,
        shutdown_timeout_seconds=0.1,
    )
    try:
        assert publisher.submit(_request("stall"))
        assert not publisher.submit(_request("other"))
        result = publisher.close()
        assert result is not None and result.error is not None
        assert result.request_id == "stall"
    finally:
        publisher.close()


def test_discarded_publisher_does_not_leave_an_idle_process():
    publisher = ProcessTrajectoryPublisher(
        _storage_worker,
        publish_timeout_seconds=5,
        shutdown_timeout_seconds=1,
    )
    result = publisher.execute(_request("first"))
    assert result.error is None
    del publisher
    gc.collect()
    with pytest.raises(ProcessLookupError):
        os.kill(result.ledger["pid"], 0)
