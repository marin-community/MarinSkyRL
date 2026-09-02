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

    flushed: list[float] = []
    monkeypatch.setattr(module.telemetry, "flush", lambda timeout: flushed.append(timeout) or True)
    monkeypatch.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())

    publish_worker_spans({"policy_ppo_train": 1.0}, step=3, rank=0)
    assert flushed == [module.TELEMETRY_FLUSH_TIMEOUT_SECONDS]

    flushed.clear()
    publish_worker_spans({}, step=3, rank=0)
    assert flushed == [], "nothing to publish means nothing to flush"


def test_a_flush_that_does_not_settle_is_logged_not_swallowed(monkeypatch, caplog):
    """flush() settles the queue; it does not guarantee delivery.

    A silently short row set understates max and p95 in exactly the same direction as losing rows at
    shutdown, so the False return has to be loud.
    """
    import logging

    import skyrl_train.timing_observability as module

    monkeypatch.setattr(module.telemetry, "flush", lambda timeout: False)
    monkeypatch.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        publish_worker_spans({"policy_ppo_train": 1.0}, step=9, rank=2)

    assert any("did not settle" in record.message for record in caplog.records)


def test_flush_timeout_stays_off_the_critical_path():
    """It blocks the worker's return, so every second lands in driver policy_train unattributed."""
    import skyrl_train.timing_observability as module

    assert module.TELEMETRY_FLUSH_TIMEOUT_SECONDS <= 1.0


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
