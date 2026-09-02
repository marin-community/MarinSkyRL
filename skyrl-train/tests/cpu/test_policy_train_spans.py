# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side decomposition of policy_train.

policy_train is a leaf on the driver -- a Ray dispatch plus a wait for the slowest policy worker --
and on a 67B-A2B MoE run it is ~90% of the step with nothing measured inside it. These tests cover
the span layer that decomposes it, and specifically the three ways it could look like it works while
reporting nothing.
"""

from __future__ import annotations

import pytest

from skyrl_train.timing_observability import (
    POLICY_TRAIN_SPANS,
    TIMING_PARENTS,
    WorkerSpanAccumulator,
    WorkerTimingSink,
    phase_timing_observations,
    publish_worker_spans,
)


def _stub_telemetry(monkeypatch, module, *, settled=True, lost_delta=0):
    """Stub the export surface: flush result plus the counters loss is actually detected from."""
    from types import SimpleNamespace

    calls = {"flush": [], "status": 0}

    def _status():
        calls["status"] += 1
        lost = lost_delta if calls["status"] > 1 else 0
        return SimpleNamespace(lost_records=lost, rejected_records=0)

    monkeypatch.setattr(module.telemetry, "flush", lambda timeout: calls["flush"].append(timeout) or settled)
    monkeypatch.setattr(module.telemetry, "runtime_status", _status)
    monkeypatch.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())
    return calls


def _accumulator(**kwargs) -> WorkerSpanAccumulator:
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("synchronize", False)
    return WorkerSpanAccumulator(**kwargs)


def test_every_span_is_registered_in_the_timing_tree():
    """An unregistered span is dropped in silence.

    phase_timing_observations filters on membership in TIMING_PARENTS, so a span missing from it
    publishes nothing at all -- which is indistinguishable from a phase that costs nothing.
    """
    for name in (*POLICY_TRAIN_SPANS, "policy_ppo_train", "policy_span_residual"):
        assert name in TIMING_PARENTS, f"{name} would be silently dropped"
    assert TIMING_PARENTS["policy_ppo_train"] == "policy_train"
    for name in (*POLICY_TRAIN_SPANS, "policy_span_residual"):
        assert TIMING_PARENTS[name] == "policy_ppo_train"


def test_disabled_accumulator_records_nothing_and_costs_nothing():
    accumulator = WorkerSpanAccumulator(enabled=False)
    with accumulator.span("policy_training_step"):
        pass
    assert accumulator.totals(total_seconds=1.0) == {}


def test_spans_accumulate_across_micro_steps():
    accumulator = _accumulator()
    for _ in range(3):
        with accumulator.span("policy_training_step"):
            pass
        with accumulator.span("policy_metric_allreduce"):
            pass
    totals = accumulator.totals()
    assert set(totals) == {"policy_training_step", "policy_metric_allreduce"}
    assert all(value >= 0.0 for value in totals.values())


def test_residual_closes_the_decomposition():
    """The residual is what makes a wrong decomposition visible rather than plausible."""
    accumulator = _accumulator()
    with accumulator.span("policy_training_step"):
        pass
    totals = accumulator.totals(total_seconds=10.0)
    assert totals["policy_ppo_train"] == 10.0
    covered = sum(totals.get(name, 0.0) for name in POLICY_TRAIN_SPANS)
    assert totals["policy_span_residual"] == pytest.approx(10.0 - covered)


def test_residual_never_goes_negative():
    accumulator = _accumulator()
    with accumulator.span("policy_final_barrier"):
        pass
    assert accumulator.totals(total_seconds=0.0)["policy_span_residual"] == 0.0


def test_published_rows_carry_worker_role_rank_and_an_exclusive_clock_domain():
    """Leaves must not share inclusive_wall with the driver's spans, or a consumer double-counts."""
    recorded: list[tuple[float, dict[str, str]]] = []

    class _Probe(WorkerTimingSink):
        pass

    sink = _Probe(rank=5)
    observations = phase_timing_observations(
        {"policy_ppo_train": 9.0, "policy_training_step": 6.0, "policy_final_barrier": 1.0}
    )
    import skyrl_train.timing_observability as module

    original = module.phase_duration

    class _Histogram:
        def record(self, value, attributes):
            recorded.append((value, attributes))

    module.phase_duration = _Histogram()
    try:
        sink.publish(observations, step=7)
    finally:
        module.phase_duration = original

    by_phase = {attributes["phase"]: attributes for _, attributes in recorded}
    assert set(by_phase) == {"policy_ppo_train", "policy_training_step", "policy_final_barrier"}
    for attributes in by_phase.values():
        assert attributes["role"] == "worker"
        assert attributes["rank"] == "5"
        assert attributes["step"] == "7"
    assert by_phase["policy_ppo_train"]["clock_domain"] == "inclusive_wall"
    assert by_phase["policy_training_step"]["clock_domain"] == "exclusive_wall"
    assert by_phase["policy_final_barrier"]["clock_domain"] == "exclusive_wall"
    # The parent is the declared one, not the nearest *recorded* one: policy_train is measured on the
    # driver and is never in this mapping, so nearest_recorded_parent would orphan every leaf.
    assert by_phase["policy_ppo_train"]["parent"] == "policy_train"
    assert by_phase["policy_training_step"]["parent"] == "policy_ppo_train"


def test_publishing_an_empty_mapping_is_a_no_op():
    publish_worker_spans({}, step=1, rank=0)


def test_every_child_of_policy_ppo_train_is_either_a_measured_span_or_the_residual():
    """Guards the two halves against drift.

    A name registered under policy_ppo_train but absent from POLICY_TRAIN_SPANS is excluded from the
    residual arithmetic, so its time would be counted twice -- once in its own row and again inside
    the residual.
    """
    children = {name for name, parent in TIMING_PARENTS.items() if parent == "policy_ppo_train"}
    assert children == set(POLICY_TRAIN_SPANS) | {"policy_span_residual"}


def test_presync_false_leaves_a_self_synchronising_region_measurable():
    """A leading sync would drain the queue and leave the wrapped sync timing nothing."""
    calls: list[str] = []

    accumulator = WorkerSpanAccumulator(enabled=True, synchronize=True)
    accumulator._sync = lambda: calls.append("sync")  # type: ignore[method-assign]

    with accumulator.span("policy_training_step"):
        pass
    assert calls == ["sync", "sync"], "a normal span brackets itself with synchronises"

    calls.clear()
    with accumulator.span("policy_entry_barrier", presync=False):
        pass
    assert calls == ["sync"], "presync=False must not synchronise before the region starts"


def test_publish_flushes_so_the_last_step_survives_ray_kill(monkeypatch):
    """Ray does not run atexit handlers on ray.kill, so durability has to be at publish time."""
    import skyrl_train.timing_observability as module

    calls = _stub_telemetry(monkeypatch, module)
    elapsed = publish_worker_spans({"policy_ppo_train": 1.0}, step=3, rank=0)
    assert calls["flush"] == [module.TELEMETRY_FLUSH_TIMEOUT_SECONDS]
    assert elapsed >= 0.0, "the publish cost is returned so it can be carried into the next step"

    calls["flush"].clear()
    assert publish_worker_spans({}, step=3, rank=0) == 0.0
    assert calls["flush"] == [], "nothing to publish means nothing to flush"


def test_dropped_records_are_detected_from_counters_not_from_the_flush_result(monkeypatch, caplog):
    """flush() returns True once dropped records have SETTLED, so True does not mean delivered.

    Detecting loss from the return value alone would let a short row set -- which understates max and
    p95 over ranks -- pass as a clean measurement.
    """
    import logging

    import skyrl_train.timing_observability as module

    _stub_telemetry(monkeypatch, module, settled=True, lost_delta=3)
    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        publish_worker_spans({"policy_ppo_train": 1.0}, step=9, rank=2)

    assert any("lost 3 record" in r.message for r in caplog.records), (
        "a True flush with dropped records must still warn"
    )


def test_a_flush_that_does_not_settle_is_logged(monkeypatch, caplog):
    import logging

    import skyrl_train.timing_observability as module

    _stub_telemetry(monkeypatch, module, settled=False)
    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        publish_worker_spans({"policy_ppo_train": 1.0}, step=9, rank=2)

    assert any("did not settle" in r.message for r in caplog.records)


def test_flush_timeout_stays_short_because_it_blocks_the_workers_return():
    import skyrl_train.timing_observability as module

    assert module.TELEMETRY_FLUSH_TIMEOUT_SECONDS <= 1.0


def test_publish_cost_has_a_span_so_it_is_attributable():
    """It happens after the window closes; carried forward one step beats being unmeasured."""
    assert TIMING_PARENTS["policy_span_publish"] == "policy_ppo_train"
    assert "policy_span_publish" in POLICY_TRAIN_SPANS


def test_previous_publish_is_emitted_under_its_own_step_not_this_one():
    """Labelling step n-1's publish as step n, and subtracting it from step n's residual, removes
    time that interval never contained."""
    import skyrl_train.timing_observability as module

    rows: list[tuple[str, int]] = []

    class _Histogram:
        def record(self, value, attributes):
            rows.append((attributes["phase"], int(attributes["step"])))

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(module, "phase_duration", _Histogram())
        monkey.setattr(module.telemetry, "flush", lambda timeout: True)
        monkey.setattr(
            module.telemetry,
            "runtime_status",
            lambda: type("S", (), {"lost_records": 0, "rejected_records": 0})(),
        )
        publish_worker_spans({"policy_ppo_train": 9.0}, step=5, rank=0, previous_publish=(4, 0.25))
    finally:
        monkey.undo()

    assert ("policy_ppo_train", 5) in rows
    assert ("policy_span_publish", 4) in rows, "the publish cost belongs to the step that incurred it"
    assert ("policy_span_publish", 5) not in rows


def test_totals_does_not_absorb_the_publish_cost():
    """It happens after the window closes, so it is not part of that interval's decomposition."""
    accumulator = _accumulator()
    with accumulator.span("policy_training_step"):
        pass
    totals = accumulator.totals(total_seconds=10.0)
    assert "policy_span_publish" not in totals


def test_dropped_records_are_not_double_counted():
    """Rigging already folds rejected records into lost_records; summing both reports 2N for N."""
    import logging

    import skyrl_train.timing_observability as module

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    seen = {"n": 0}

    def _status():
        seen["n"] += 1
        lost = 3 if seen["n"] > 1 else 0
        return type("S", (), {"lost_records": lost, "rejected_records": lost})()

    try:
        monkey.setattr(module.telemetry, "runtime_status", _status)
        monkey.setattr(module.telemetry, "flush", lambda timeout: True)
        monkey.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())
        import _pytest.logging  # noqa: F401

        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())  # type: ignore[method-assign]
        module.logger.addHandler(handler)
        try:
            publish_worker_spans({"policy_ppo_train": 1.0}, step=1, rank=0)
        finally:
            module.logger.removeHandler(handler)
    finally:
        monkey.undo()

    assert any("lost 3 record" in m for m in records), f"expected 3, got: {records}"


def test_worker_init_configures_telemetry_for_the_worker_role():
    """The wiring that stops the spans publishing into nothing.

    Rigging discards every record while unconfigured, and a Ray actor never enters
    ``process_telemetry`` on its own -- ``main_base`` does it for the trainer and driver roles only.
    Without this the spans emit silently into nothing, which is indistinguishable from a phase that
    costs nothing.

    Asserted on the source of ``Worker.__init__`` because the constructor cannot be exercised
    off-actor: it needs a live torch.distributed rendezvous. A structural check is weak, but it does
    catch the regression that matters -- someone deleting the wiring and leaving a green suite.
    """
    import inspect

    from skyrl_train.workers.worker import Worker

    source = inspect.getsource(Worker.__init__)
    assert "process_telemetry(WORKER_ROLE)" in source
    assert 'cfg.trainer.get("policy_train_spans", False)' in source
    # Ray kills actors rather than unwinding them, so the drain must be registered, not relied upon.
    assert "atexit.register" in source


def test_ppo_train_wires_the_span_layer():
    """The wiring itself, which no behavioural test can reach.

    ``ppo_train`` needs a live torch.distributed rendezvous and a Ray actor, so the seam it calls is
    tested directly (above) while its *use* is asserted structurally. Weak, but it catches the
    regressions that leave a green suite: timing that starts after the entry barrier, a publish that
    never carries the previous cost forward, or a return value that is dropped so every step reports
    a publish cost of zero.
    """
    import inspect

    from skyrl_train.workers.worker import PolicyWorkerBase

    # PolicyWorkerBase, not Worker: CriticWorkerBase has its own ppo_train and is not instrumented.
    source = inspect.getsource(PolicyWorkerBase.ppo_train)
    # The clock starts before the drain barrier, not after it.
    assert source.index("_policy_spans_started = time.perf_counter()") < source.index("WORKER_PPO_TRAIN_DRAIN_BARRIER")
    # The self-synchronising region opts out of the leading synchronise.
    assert 'span("policy_entry_barrier", presync=False)' in source
    # The previous step's publish cost is passed with its own step label, and this step's retained.
    assert 'getattr(self, "_policy_span_publish", None)' in source
    assert "previous_publish=_previous_publish" in source
    assert "self._policy_span_publish = (" in source
