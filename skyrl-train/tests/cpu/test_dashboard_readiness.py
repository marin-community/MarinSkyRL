"""The readiness report, which decides whether a finished run can be read on the dashboard."""

import json

import pytest

from ci.marin_nightly.dashboard_readiness import (
    MARKER,
    REQUIRED_SIGNALS,
    collect,
    readiness,
    report_line,
)


def _row(name: str, *, metric_source: str = "", work_kind: str = "", records: int = 1) -> dict[str, object]:
    return {"name": name, "metric_source": metric_source, "work_kind": work_kind, "records": records}


def _complete_rows() -> list[dict[str, object]]:
    return [
        _row(signal.name, metric_source=signal.metric_source, work_kind=signal.work_kind, records=7)
        for signal in REQUIRED_SIGNALS
    ]


class _FakeClient:
    """A log server that returns a scripted result per attempt."""

    def __init__(self, results: list[list[dict[str, object]]]) -> None:
        self._results = results
        self.queries: list[str] = []

    def query(self, sql: str, *, max_rows: int):
        self.queries.append(sql)
        rows = self._results[min(len(self.queries) - 1, len(self._results) - 1)]
        return _FakeTable(rows)


class _FakeTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return self._rows


def test_a_run_that_reported_everything_has_nothing_missing() -> None:
    result = readiness("run-1", _complete_rows())

    assert result.missing == ()
    assert all(signal.records == 7 for signal in result.signals)


def test_a_run_without_the_ray_collector_names_the_panels_that_go_blank() -> None:
    """The failure this exists for: the collector loses its startup race, training is unaffected,
    and two panels are empty with nothing in the log to explain them."""
    rows = [row for row in _complete_rows() if row["metric_source"] != "ray"]

    result = readiness("run-1", rows)

    assert set(result.missing) == {"ray_object_store", "ray_spill"}
    blank = {signal.panel for signal in result.signals if signal.records == 0}
    assert blank == {"Ray object store occupancy", "Ray spill manager bytes by state"}


def test_the_engine_signals_are_not_satisfied_by_the_trainers_own_rows() -> None:
    """`generation_tokens_total` arrives from the engine under metric_source='vllm'. A row with the
    same name and no metric source is a different measurement and must not count."""
    rows = [_row(signal.name, work_kind=signal.work_kind, records=3) for signal in REQUIRED_SIGNALS]

    result = readiness("run-1", rows)

    assert "engine_throughput" in result.missing


def test_work_kinds_sharing_a_metric_name_are_counted_apart() -> None:
    rows = [_row("work_completed", work_kind="rollout", records=30)]

    result = readiness("run-1", rows)

    assert "rollouts" not in result.missing
    assert {"samples", "generated_tokens"} <= set(result.missing)


def test_polling_stops_as_soon_as_the_rows_have_flushed() -> None:
    client = _FakeClient([[], _complete_rows()])

    result = collect(client, "run-1", attempts=6, interval=0.0)

    assert result.missing == ()
    assert len(client.queries) == 2, "polling continued after a complete result"


def test_polling_gives_up_after_its_budget_and_reports_what_it_saw() -> None:
    client = _FakeClient([[]])

    result = collect(client, "run-1", attempts=3, interval=0.0)

    assert len(client.queries) == 3
    assert set(result.missing) == {signal.key for signal in REQUIRED_SIGNALS}


def test_a_run_id_is_refused_rather_than_interpolated_into_the_query() -> None:
    with pytest.raises(ValueError):
        collect(_FakeClient([[]]), "run'; DROP TABLE x --", attempts=1, interval=0.0)


def test_the_report_line_survives_being_recovered_from_an_iris_log() -> None:
    """The workflow does not read this function's return value. It scans a streamed job log, whose
    every line carries an Iris prefix, for the marker, and splits the JSON off the right of it. So
    the line has to be one line, and it has to survive that prefix."""
    line = report_line(readiness("run-1", _complete_rows()).as_payload())
    as_logged = f"I20260910 09:38:05 1367 iris.cluster.client.remote_client task=/a/b/0:0 | {line}"

    assert "\n" not in line
    payload = json.loads(as_logged.split(MARKER, 1)[1])
    assert payload["run_id"] == "run-1"
    assert len(payload["signals"]) == len(REQUIRED_SIGNALS)
    assert payload["missing"] == []
