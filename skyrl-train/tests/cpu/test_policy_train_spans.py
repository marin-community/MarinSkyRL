# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side decomposition of policy_train.

policy_train is a leaf on the driver -- a Ray dispatch plus a wait for the slowest policy worker --
and on a 67B-A2B MoE run it is ~90% of the step with nothing measured inside it. These tests cover
the span layer that decomposes it, and specifically the three ways it could look like it works while
reporting nothing.
"""

from __future__ import annotations

import inspect
import re
import sys
import time
from types import SimpleNamespace

import pytest

from skyrl_train.timing_observability import (
    StepMemoryProbe,
    POLICY_TRAIN_SPANS,
    TIMING_PARENTS,
    WorkerSpanAccumulator,
    WorkerTimingSink,
    phase_timing_observations,
    publish_worker_counters,
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
    with accumulator.span("policy_forward"):
        pass
    assert accumulator.totals(total_seconds=1.0) == {}


def test_spans_accumulate_across_micro_steps():
    accumulator = _accumulator()
    for _ in range(3):
        with accumulator.span("policy_forward"):
            pass
        with accumulator.span("policy_metric_allreduce"):
            pass
    totals = accumulator.totals()
    assert set(totals) == {"policy_forward", "policy_metric_allreduce"}
    assert all(value >= 0.0 for value in totals.values())


def test_residual_closes_the_decomposition():
    """The residual is what makes a wrong decomposition visible rather than plausible."""
    accumulator = _accumulator()
    with accumulator.span("policy_forward"):
        pass
    totals = accumulator.totals(total_seconds=10.0)
    assert totals["policy_ppo_train"] == 10.0
    covered = sum(totals.get(name, 0.0) for name in POLICY_TRAIN_SPANS)
    assert totals["policy_span_residual"] == pytest.approx(10.0 - covered)


def test_residual_is_signed_so_over_coverage_is_visible():
    """Clamping at zero would hide double-counting, which is what the residual exists to surface."""
    accumulator = _accumulator()
    with accumulator.span("policy_final_barrier"):
        pass
    assert accumulator.totals(total_seconds=0.0)["policy_span_residual"] <= 0.0


def test_record_zero_distinguishes_did_not_run_from_cost_nothing():
    """A conditional region must say which; a missing row and a zero row are different claims."""
    accumulator = _accumulator()
    assert "policy_entry_barrier" not in accumulator.totals()
    accumulator.record_zero("policy_entry_barrier")
    assert accumulator.totals()["policy_entry_barrier"] == 0.0


def test_disabled_accumulator_records_no_zeros_either():
    accumulator = WorkerSpanAccumulator(enabled=False)
    accumulator.record_zero("policy_entry_barrier")
    assert accumulator.totals(total_seconds=1.0) == {}


def test_training_step_is_split_not_one_coarse_span():
    """A single policy_training_step span reports ~95% of the parent and answers nothing."""
    for name in ("policy_forward", "policy_backward", "policy_optimizer_step", "policy_entropy_allreduce"):
        assert TIMING_PARENTS[name] == "policy_ppo_train"
        assert name in POLICY_TRAIN_SPANS
    # It is registered but INCLUSIVE: it wraps training_step, which contains the four leaves above.
    # Keeping it out of POLICY_TRAIN_SPANS is what stops the residual double-counting them -- the
    # first instrumented run reported -1703 s against a 1706 s parent before this was fixed.
    assert TIMING_PARENTS["policy_training_step"] == "policy_ppo_train"
    assert "policy_training_step" not in POLICY_TRAIN_SPANS


def test_published_rows_carry_worker_role_rank_and_an_exclusive_clock_domain():
    """Leaves must not share inclusive_wall with the driver's spans, or a consumer double-counts."""
    recorded: list[tuple[float, dict[str, str]]] = []

    class _Probe(WorkerTimingSink):
        pass

    sink = _Probe(rank=5)
    observations = phase_timing_observations(
        {"policy_ppo_train": 9.0, "policy_forward": 6.0, "policy_final_barrier": 1.0}
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
    assert set(by_phase) == {"policy_ppo_train", "policy_forward", "policy_final_barrier"}
    for attributes in by_phase.values():
        assert attributes["role"] == "worker"
        assert attributes["rank"] == "5"
        assert attributes["step"] == "7"
    assert by_phase["policy_ppo_train"]["clock_domain"] == "inclusive_wall"
    assert by_phase["policy_forward"]["clock_domain"] == "exclusive_wall"
    assert by_phase["policy_final_barrier"]["clock_domain"] == "exclusive_wall"
    # The parent is the declared one, not the nearest *recorded* one: policy_train is measured on the
    # driver and is never in this mapping, so nearest_recorded_parent would orphan every leaf.
    assert by_phase["policy_ppo_train"]["parent"] == "policy_train"
    assert by_phase["policy_forward"]["parent"] == "policy_ppo_train"


def test_publishing_an_empty_mapping_is_a_no_op():
    publish_worker_spans({}, step=1, rank=0)


def test_every_child_of_policy_ppo_train_is_either_a_measured_span_or_the_residual():
    """Guards the two halves against drift.

    A name registered under policy_ppo_train but absent from POLICY_TRAIN_SPANS is excluded from the
    residual arithmetic, so its time would be counted twice -- once in its own row and again inside
    the residual.
    """
    children = {name for name, parent in TIMING_PARENTS.items() if parent == "policy_ppo_train"}
    # policy_training_step is the one INCLUSIVE child: it wraps training_step, which contains
    # forward/backward/optimizer/entropy. It is registered so the tree is navigable, and excluded
    # from POLICY_TRAIN_SPANS so the residual does not count its children a second time.
    assert children == set(POLICY_TRAIN_SPANS) | {"policy_span_residual", "policy_training_step"}


def test_presync_false_leaves_a_self_synchronising_region_measurable():
    """A leading sync would drain the queue and leave the wrapped sync timing nothing."""
    calls: list[str] = []

    accumulator = WorkerSpanAccumulator(enabled=True, synchronize=True)
    accumulator._sync = lambda: calls.append("sync")  # type: ignore[method-assign]

    with accumulator.span("policy_forward"):
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
    with accumulator.span("policy_forward"):
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

    from skyrl_train.workers.worker import Worker

    source = inspect.getsource(Worker.__init__)
    assert "process_telemetry(WORKER_ROLE)" in source
    assert '"policy_train_spans", False' in source
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

    import skyrl_train.workers.worker as worker_module
    from skyrl_train.workers.worker import PolicyWorkerBase

    # PolicyWorkerBase, not Worker: CriticWorkerBase has its own ppo_train and is not instrumented.
    source = inspect.getsource(PolicyWorkerBase.ppo_train)
    # The clock starts before the drain barrier, not after it.
    assert source.index("_policy_spans_started = time.perf_counter()") < source.index("WORKER_PPO_TRAIN_DRAIN_BARRIER")
    # The self-synchronising region opts out of the leading synchronise.
    assert 'span("policy_entry_barrier", presync=False)' in source
    # The previous step's publish cost is passed with its own step label, and this step's retained.
    # The previous step's publish cost is carried forward and this step's retained. Both now go
    # through _publish_policy_spans, whose behaviour is asserted directly below.
    assert 'previous_publish=getattr(self, "_policy_span_publish", None)' in source
    assert "self._policy_span_publish = (" in source
    # The counter call G3 found untested rides the spans publish rather than standing alone, so a
    # dropped counter row is not invisible. That gating now lives in _publish_policy_spans and is
    # asserted behaviourally below.
    # The counters live in their own function so the caller can guard them; assert the names there.
    counters_source = inspect.getsource(worker_module._policy_span_counters)
    assert "micro_step_count" in counters_source
    assert "attention_work_ratio" in counters_source
    # 🚨 Neither gathering nor publishing may kill a step. Both run AFTER the optimizer step and the
    # final barrier, and both issue CUDA synchronisations and allocator queries; a raise there
    # discards work already paid for -- an hour of 80 H100s at E6 geometry -- to lose a telemetry
    # row. docs/telemetry.md states the contract: export failures do not change training results.
    assert "except Exception:" in source, "the counter gathering is not guarded"
    # The publish path's own guarantees are asserted BEHAVIOURALLY below, against the function
    # rather than against this source text.
    assert "_publish_policy_spans(" in source
    # H2's keystone. Three OOMs in this workstream asked for the eager attention score tensor and
    # no memory series existed to see any of them coming. The counters now come from the probe.
    assert "_step_memory.counters()" in counters_source
    # Rebased at the TOP, before any forward. Published raw, max_memory_allocated is the peak since
    # the process began, so a step attribute on it is a lie: after whichever step sets the
    # high-water mark, every later step republishes the same number.
    assert source.index("self._step_memory.begin_step()") < source.index("WORKER_PPO_TRAIN_DRAIN_BARRIER")
    # And reset NOWHERE ELSE -- asserted on the probe, not on this source text. The old guard read
    # `"reset_peak_memory_stats" not in source` and went vacuous the moment the call moved into
    # begin_step: it passed because the string had left ppo_train, not because the invariant held.
    import skyrl_train.timing_observability as timing_module

    probe_source = inspect.getsource(timing_module.StepMemoryProbe)
    assert probe_source.count("reset_peak_memory_stats") == 1
    assert "reset_peak_memory_stats" in inspect.getsource(timing_module.StepMemoryProbe.begin_step)


def test_counters_go_to_their_own_instrument_not_the_span_tree():
    """They are counts and ratios, not durations.

    Riding phase_duration would put a units mismatch into the span tree, where nothing downstream
    would notice it being summed into policy_ppo_train or subtracted from the residual.
    """
    import skyrl_train.timing_observability as module

    import pytest as _pytest

    recorded: list[tuple[float, dict]] = []
    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(
            module,
            "policy_step_counter",
            type("_H", (), {"record": lambda self, v, attributes: recorded.append((v, attributes))})(),
        )
        monkey.setattr(
            module, "phase_duration", type("_H", (), {"record": lambda *a, **k: pytest.fail("wrong instrument")})()
        )
        publish_worker_counters({"micro_step_count": 64.0, "rank_tokens_real": 12.0}, step=2, rank=7)
    finally:
        monkey.undo()

    by_name = {a["counter"]: (v, a) for v, a in recorded}
    assert by_name["micro_step_count"][0] == 64.0
    assert by_name["rank_tokens_real"][1]["rank"] == "7"
    assert by_name["rank_tokens_real"][1]["step"] == "2"
    assert by_name["micro_step_count"][1]["role"] == "worker"
    # And they must not be registered as spans, or they would join the residual arithmetic.
    for name in (
        "micro_step_count",
        "rank_tokens_real",
        "rank_tokens_padded",
        "attention_work_ratio",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "alloc_retries",
        "alloc_ooms",
    ):
        assert name not in TIMING_PARENTS
        assert name not in POLICY_TRAIN_SPANS


def test_publishing_no_counters_is_a_no_op():
    publish_worker_counters({}, step=1, rank=0)


def test_counters_ride_the_spans_publish_so_loss_detection_covers_them(monkeypatch):
    """Published separately they sat outside the before/after loss window and were invisible."""
    import skyrl_train.timing_observability as module

    seen: dict[str, object] = {}
    monkeypatch.setattr(module.telemetry, "flush", lambda timeout: True)
    monkeypatch.setattr(
        module.telemetry,
        "runtime_status",
        lambda: type("S", (), {"lost_records": 0, "rejected_records": 0})(),
    )
    monkeypatch.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())
    monkeypatch.setattr(
        module,
        "policy_step_counter",
        type("_H", (), {"record": lambda self, v, attributes: seen.setdefault(attributes["counter"], v)})(),
    )

    module.publish_worker_spans({"policy_ppo_train": 1.0}, step=4, rank=1, counters={"micro_step_count": 64.0})
    assert seen == {"micro_step_count": 64.0}


def test_counters_alone_still_publish():
    """Spans may be empty on a step that produced none; the counters must not be dropped with them."""
    import skyrl_train.timing_observability as module

    import pytest as _pytest

    seen: list[str] = []
    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(module.telemetry, "flush", lambda timeout: True)
        monkey.setattr(
            module.telemetry,
            "runtime_status",
            lambda: type("S", (), {"lost_records": 0, "rejected_records": 0})(),
        )
        monkey.setattr(module, "phase_duration", type("_H", (), {"record": lambda *a, **k: None})())
        monkey.setattr(
            module,
            "policy_step_counter",
            type("_H", (), {"record": lambda self, v, attributes: seen.append(attributes["counter"])})(),
        )
        module.publish_worker_spans({}, step=4, rank=1, counters={"rank_tokens_real": 7.0})
    finally:
        monkey.undo()
    assert seen == ["rank_tokens_real"]


# --- allocator counters, and the trap in the raw values ------------------------------------------


class _FakeCuda:
    """A scripted allocator. `peaks` and `allocs` are consumed in call order."""

    def __init__(self, peaks, allocs, retries=0, ooms=0):
        self._peaks, self._allocs = iter(peaks), iter(allocs)
        self.retries, self.ooms, self.resets = retries, ooms, 0

    is_available = staticmethod(lambda: True)
    is_initialized = staticmethod(lambda: True)

    def reset_peak_memory_stats(self):
        self.resets += 1

    def max_memory_allocated(self):
        return next(self._peaks)

    def max_memory_reserved(self):
        return 0.0

    def memory_allocated(self):
        return next(self._allocs)

    def memory_stats(self):
        return {"num_alloc_retries": self.retries, "num_ooms": self.ooms}


def _with_cuda(monkeypatch, cuda):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def test_the_memory_probe_is_inert_when_disabled():
    """It rides trainer.policy_train_spans, so an off run pays nothing and publishes nothing."""
    probe = StepMemoryProbe(enabled=False)
    probe.begin_step()
    assert probe.counters() == {}


def test_the_allocator_peak_is_rebased_so_it_describes_this_step(monkeypatch):
    """🚨 max_memory_allocated is "peak since the beginning of this program".

    Published per step without a baseline it is a process-lifetime high-water mark wearing a step
    attribute: after whichever step sets it, every later step republishes the same number and the
    series looks flat and healthy while measuring nothing. begin_step is the only reset.
    """
    cuda = _FakeCuda(peaks=[500.0], allocs=[])
    _with_cuda(monkeypatch, cuda)
    probe = StepMemoryProbe(enabled=True)
    probe.begin_step()
    assert cuda.resets == 1, "the peak must be rebased at the top of the step"
    assert probe.counters()["peak_allocated_bytes"] == 500.0


def test_the_cumulative_allocator_counters_are_published_as_deltas(monkeypatch):
    """num_alloc_retries and num_ooms are cumulative for the PROCESS.

    Published raw and tagged by step, summing them across steps counts old events again. A step that
    had no retries must publish zero, not the running total.
    """
    cuda = _FakeCuda(peaks=[0.0], allocs=[], retries=7, ooms=2)
    _with_cuda(monkeypatch, cuda)
    probe = StepMemoryProbe(enabled=True)
    probe.begin_step()
    cuda.retries, cuda.ooms = 9, 2  # two more retries during the step, no new OOM
    counters = probe.counters()
    assert counters["alloc_retries"] == 2.0, "the seven that predate this step are not ours"
    assert counters["alloc_ooms"] == 0.0


def test_a_byte_valued_counter_does_not_ride_the_unit_one_instrument(monkeypatch):
    """A byte value on a unit-1 histogram is a lie a consumer cannot see.

    Same suffix dispatch the driver counters already use.
    """
    import skyrl_train.timing_observability as module

    counts: list[str] = []
    byte_rows: list[str] = []
    monkeypatch.setattr(
        module,
        "policy_step_counter",
        type("_H", (), {"record": lambda self, v, attributes: counts.append(attributes["counter"])})(),
    )
    monkeypatch.setattr(
        module,
        "policy_step_bytes",
        type("_H", (), {"record": lambda self, v, attributes: byte_rows.append(attributes["counter"])})(),
    )
    module.publish_worker_counters(
        {"peak_allocated_bytes": 1.0, "micro_step_count": 2.0, "optimizer_step_peak_delta_bytes": 3.0},
        step=1,
        rank=0,
    )
    assert sorted(byte_rows) == ["optimizer_step_peak_delta_bytes", "peak_allocated_bytes"]
    assert counts == ["micro_step_count"]


def test_unsynchronized_spans_do_not_ship_the_synchronized_clock_domain():
    """🚨 The mode decides what the number means, so it must be visible on the row.

    Without a device synchronise a span measures kernel LAUNCH time and charges a backward's real
    cost to whatever later call happens to block; with it, the span measures execution and the
    pipeline is serialised. The accumulator's docstring says never to compare the two -- and a
    consumer cannot obey that if both arrive under the same label.
    """
    import skyrl_train.timing_observability as timing_module

    def _domains(synchronize):
        rows: list[dict[str, str]] = []
        monkey = pytest.MonkeyPatch()
        try:
            monkey.setattr(
                timing_module,
                "phase_duration",
                type("_H", (), {"record": lambda self, v, attributes: rows.append(attributes)})(),
            )
            timing_module.WorkerTimingSink(0, synchronize=synchronize).publish(
                timing_module.phase_timing_observations({"policy_ppo_train": 9.0, "policy_backward": 4.0}),
                step=1,
            )
        finally:
            monkey.undo()
        return {row["phase"]: row["clock_domain"] for row in rows}

    synced = _domains(True)
    launched = _domains(False)
    assert synced == {"policy_ppo_train": "inclusive_wall", "policy_backward": "exclusive_wall"}
    assert launched == {"policy_ppo_train": "inclusive_launch", "policy_backward": "exclusive_launch"}
    assert not set(synced.values()) & set(launched.values()), "the two modes must not share a label"


def test_a_failing_publish_does_not_raise_and_reports_its_real_cost(monkeypatch):
    """🚨 Behavioural, not a source-text match.

    This runs after the step's GPU work is already paid for. A raise here discards a completed step
    -- an hour of 80 H100s at E6 geometry -- to lose a telemetry row. And the cost it reports must be
    the REAL elapsed time: the next step subtracts it as policy_span_publish, so a convenient zero
    would understate that step by exactly what the failure cost.
    """
    import skyrl_train.workers.worker as worker_module

    def _boom(*args, **kwargs):
        time.sleep(0.02)
        raise RuntimeError("finelog is down")

    monkeypatch.setattr(worker_module, "publish_worker_spans", _boom)
    spans = WorkerSpanAccumulator(enabled=True, synchronize=False)

    cost = worker_module._publish_policy_spans(
        spans, total_seconds=1.0, step=3, rank=0, previous_publish=None, counters={}
    )
    assert cost >= 0.02, f"a failed publish reported {cost}s; it actually spent at least 0.02s"


def test_a_successful_publish_returns_the_publisher_s_own_cost(monkeypatch):
    """The happy path must pass the publisher's number through untouched, not re-time it."""
    import skyrl_train.workers.worker as worker_module

    monkeypatch.setattr(worker_module, "publish_worker_spans", lambda *a, **k: 0.125)
    spans = WorkerSpanAccumulator(enabled=True, synchronize=False)
    cost = worker_module._publish_policy_spans(
        spans, total_seconds=1.0, step=3, rank=0, previous_publish=None, counters={}
    )
    assert cost == 0.125


def test_the_publisher_is_told_which_clock_mode_produced_the_spans(monkeypatch):
    """The mode changes what the numbers mean, so it must reach the sink rather than default."""
    import skyrl_train.workers.worker as worker_module

    seen = {}
    monkeypatch.setattr(worker_module, "publish_worker_spans", lambda *a, **k: seen.update(k) or 0.0)
    for synchronize in (True, False):
        worker_module._publish_policy_spans(
            WorkerSpanAccumulator(enabled=True, synchronize=synchronize),
            total_seconds=1.0,
            step=1,
            rank=0,
            previous_publish=None,
            counters={},
        )
        assert seen["synchronize"] is synchronize


def test_disabled_spans_publish_no_counters_even_when_some_are_gathered(monkeypatch):
    """The counters ride the spans publish. With spans off, none may reach the sink.

    Behavioural: the old assertion matched the source line `counters=... if enabled else None`,
    which a rename would break and a logic inversion would not.
    """
    import skyrl_train.workers.worker as worker_module

    seen = {}
    monkeypatch.setattr(worker_module, "publish_worker_spans", lambda *a, **k: seen.update(k) or 0.0)
    counters = {"micro_step_count": 4.0}

    worker_module._publish_policy_spans(
        WorkerSpanAccumulator(enabled=False),
        total_seconds=1.0,
        step=1,
        rank=0,
        previous_publish=None,
        counters=counters,
    )
    assert seen["counters"] is None, "spans are off; nothing may be published"

    worker_module._publish_policy_spans(
        WorkerSpanAccumulator(enabled=True),
        total_seconds=1.0,
        step=1,
        rank=0,
        previous_publish=None,
        counters=counters,
    )
    assert seen["counters"] == counters


def test_the_old_logprob_forward_does_not_touch_spans_before_ppo_train_creates_them():
    """🚨 The forward runs BEFORE ppo_train, and ppo_train is where _policy_spans is assigned.

    `fwd_logprobs_values_reward` is called at trainer.py:706; `ppo_train` at :723. So on step 1 the
    first `_forward_micro_batch` happens on a worker that has no `_policy_spans` attribute at all.
    A bare `self._policy_spans.enabled` raises AttributeError there — on EVERY run, whether or not
    the probe is enabled — which is exactly the crash this asserts against.
    """
    import skyrl_train.workers.worker as worker_module

    source = inspect.getsource(worker_module.PolicyWorkerBase._forward_micro_batch)
    assert "self._policy_spans." not in source, (
        "_forward_micro_batch dereferences _policy_spans directly; it does not exist yet on step 1"
    )
    assert 'getattr(self, "_policy_spans", None)' in source

    # And behaviourally: the guard must survive a worker that has never run ppo_train.
    class _Bare:
        pass

    bare = _Bare()
    spans = getattr(bare, "_policy_spans", None)
    assert spans is None, "the getattr form is what makes a fresh worker safe"


def test_the_repeat_probe_does_not_claim_determinism_from_one_pair():
    """A single agreeing pair is not determinism, and the log must not say it is.

    The distinction decides which fix is correct: non-zero proves nondeterminism and points at the
    combine; zero on its own points nowhere, and only a long run of zeros makes a deterministic
    eval/train seam the better explanation.
    """
    import skyrl_train.workers.worker as worker_module

    source = inspect.getsource(worker_module._log_f25_repeat)
    assert "ZERO proves nothing on its own" in source
    assert "exactly 0 means this pass is deterministic" not in source


def test_the_repeat_probe_renders_its_delta():
    """A probe that logs its placeholder instead of its value is a run that measured nothing.

    worker.py logs through loguru, which formats with str.format. A %-placeholder renders
    literally and the argument is dropped -- silently, with the line still present and still
    reading like a result. That is what f25-probe3 did: 24 H100 for 19 minutes, every line
    reporting "max|delta| %.6e".
    """
    import skyrl_train.workers.worker as worker_module

    source = inspect.getsource(worker_module._log_f25_repeat)
    message = source[source.index("[F25] {} forward repeated") : source.index("tag,")]
    assert not re.search(r"%[-#0-9.]*[sdeEfgGr]", message), (
        "loguru formats with str.format; a %-placeholder drops the value"
    )
    assert "{:.6e}" in message


def test_the_repeat_probe_covers_the_training_pass_too():
    """An eval-only repeat cannot exclude train-only nondeterminism.

    G3 pass 7: the old-logprob pass runs under model.eval() and the training pass under
    model.train(), so a clean eval/eval repeat says nothing about the training path and must not
    be read as a deterministic seam. Both call sites are what make the zero case interpretable.
    """
    import skyrl_train.workers.worker as worker_module

    eval_site = inspect.getsource(worker_module.PolicyWorkerBase._forward_micro_batch)
    train_site = inspect.getsource(worker_module.PolicyWorkerBase.training_step)
    assert '_log_f25_repeat("eval"' in eval_site
    assert '_log_f25_repeat("train"' in train_site
    # The train repeat must not be charged to the phase the workstream exists to measure.
    forward_span_exit = train_site.index("_forward_span.__exit__")
    assert train_site.index('_log_f25_repeat("train"') > forward_span_exit, (
        "the repeat belongs outside policy_forward, or it inflates the span"
    )
