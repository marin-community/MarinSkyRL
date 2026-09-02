import json

import pytest

from skyrl_train.distributed import collective_phase_diagnostics, flight_recorder_summary
from skyrl_train.distributed.flight_recorder_summary import CollectiveBucket


def _entry(record_id: int, *, name: str = "nccl:reduce_scatter_tensor", duration_ms: float | None = 4.0):
    return {
        "record_id": record_id,
        "process_group": ["0", "default_pg"],
        "profiling_name": name,
        "state": "completed",
        "input_sizes": [[2, 3]],
        "input_dtypes": ["BFloat16"],
        "duration_ms": duration_ms,
    }


@pytest.fixture(autouse=True)
def reset_capture_state(monkeypatch):
    monkeypatch.setattr(flight_recorder_summary, "_state", flight_recorder_summary._CaptureState())


def test_summary_folds_entries_into_one_bucket_per_group_and_collective():
    delta = flight_recorder_summary.summarize(
        [_entry(1), _entry(2), _entry(3, name="nccl:all_gather_into_tensor", duration_ms=None)],
        after_record_id=-1,
        covers_global_step=5,
    )

    reduce_scatter = CollectiveBucket("0:default_pg", "nccl:reduce_scatter_tensor", "BFloat16", "completed")
    all_gather = CollectiveBucket("0:default_pg", "nccl:all_gather_into_tensor", "BFloat16", "completed")
    assert delta.totals[reduce_scatter].count == 2
    assert delta.totals[reduce_scatter].input_elements == 12
    assert delta.totals[reduce_scatter].duration_seconds == pytest.approx(0.008)
    assert delta.totals[reduce_scatter].timed_count == 2
    # An untimed entry still counts and still contributes bytes; only its duration is unknown.
    assert delta.totals[all_gather].count == 1
    assert delta.totals[all_gather].timed_count == 0
    assert delta.totals[all_gather].duration_seconds == 0.0
    assert (delta.first_record_id, delta.last_record_id) == (1, 3)
    assert delta.dropped_records == 0


def test_summary_skips_records_the_previous_capture_already_reported():
    delta = flight_recorder_summary.summarize(
        [_entry(1), _entry(2), _entry(3)],
        after_record_id=2,
        covers_global_step=6,
    )

    bucket = CollectiveBucket("0:default_pg", "nccl:reduce_scatter_tensor", "BFloat16", "completed")
    assert delta.totals[bucket].count == 1
    assert (delta.first_record_id, delta.last_record_id) == (3, 3)


def test_summary_reports_records_the_ring_buffer_overwrote():
    delta = flight_recorder_summary.summarize(
        [_entry(9), _entry(10)],
        after_record_id=4,
        covers_global_step=7,
    )

    assert delta.dropped_records == 4


def test_step_boundary_publishes_once_per_global_step_and_advances_the_cursor(monkeypatch):
    traces = [
        {"entries": [_entry(1), _entry(2)]},
        {"entries": [_entry(1), _entry(2), _entry(3)]},
    ]
    monkeypatch.setattr(flight_recorder_summary, "_dump_trace", lambda: traces.pop(0))
    messages: list[str] = []
    monkeypatch.setattr(flight_recorder_summary.logger, "info", messages.append)

    flight_recorder_summary.capture_at_step_boundary(2, 0)
    flight_recorder_summary.capture_at_step_boundary(2, 0)
    flight_recorder_summary.capture_at_step_boundary(2, 1)

    assert len(messages) == 2
    first = json.loads(messages[0].removeprefix(flight_recorder_summary.LOG_PREFIX))
    second = json.loads(messages[1].removeprefix(flight_recorder_summary.LOG_PREFIX))
    # The first capture covers everything before step 0, which is model initialisation.
    assert first["covers_global_step"] is None
    assert first["rank"] == 2
    assert [bucket["count"] for bucket in first["buckets"]] == [2]
    assert second["covers_global_step"] == 0
    assert [bucket["count"] for bucket in second["buckets"]] == [1]
    assert second["first_record_id"] == 3


def test_step_boundary_publishes_nothing_when_the_recorder_holds_no_collectives(monkeypatch):
    monkeypatch.setattr(flight_recorder_summary, "_dump_trace", lambda: {"entries": []})
    messages: list[str] = []
    monkeypatch.setattr(flight_recorder_summary.logger, "info", messages.append)

    flight_recorder_summary.capture_at_step_boundary(0, 0)
    flight_recorder_summary.capture_at_step_boundary(0, 1)

    assert messages == []


def test_training_step_region_captures_the_recorder_and_inference_region_does_not(monkeypatch):
    monkeypatch.setenv("SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS", "1")
    monkeypatch.setattr(collective_phase_diagnostics, "_default_process_group", lambda: _FakeGroup())
    captured: list[tuple[int, int | None]] = []
    monkeypatch.setattr(
        flight_recorder_summary,
        "capture_at_step_boundary",
        lambda rank, global_step: captured.append((rank, global_step)),
    )

    for kind in collective_phase_diagnostics.CollectiveRegionKind:
        with collective_phase_diagnostics.region(
            _FakeMesh(),
            kind=kind,
            rank=3,
            metadata=collective_phase_diagnostics.CollectiveRegionMetadata(global_step=11, local_step=0),
        ):
            pass

    assert captured == [(3, 11)]


class _FakeGroup:
    def _get_sequence_number_for_group(self) -> int:
        return 0


class _FakeMesh:
    mesh_dim_names = ("fsdp",)
    shape = (1,)

    def get_coordinate(self) -> list[int]:
        return [0]

    def get_group(self, name: str) -> _FakeGroup:
        return _FakeGroup()
