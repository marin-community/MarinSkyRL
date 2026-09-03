# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver-side decomposition of generate.

generate is 64% of an E6 step at PR488 geometry and has never been instrumented. These tests cover
the span layer that decomposes it, and specifically the ways it could look like it works while
reporting a number nobody should believe: a sum over 4,096 concurrent coroutines published as a
duration, a region that overlaps its siblings entering the residual arithmetic, two concurrent
run() calls accumulating into one dict, a per-region log record inside the phase being measured,
executor QUEUEING reported as environment runtime, and a mean standing in for a tail on a phase
that F20 showed is tail-latency-bound.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from skyrl_train.timing_observability import (
    GENERATE_NESTED_SPANS,
    GENERATE_SPANS,
    POLICY_TRAIN_SPANS,
    ROLLOUT_COUNTERS,
    ROLLOUT_ENGINE_AWAIT,
    ROLLOUT_ENV_AWAIT,
    ROLLOUT_ENV_EXEC,
    ROLLOUT_ENV_QUEUE,
    ROLLOUT_ENV_RESUME,
    ROLLOUT_TIMINGS,
    TIMING_PARENTS,
    RolloutTimings,
    phase_timing_observations,
    publish_driver_counters,
    record_generate_spans,
    rollout_span,
    rollout_timings_scope,
    rollout_trajectory,
    rollout_wait,
    timed_env_call,
)
import skyrl_train.timing_observability as timing_module
import skyrl_train.trajectory_runners.harbor.execution as harbor_execution
import skyrl_train.trajectory_runners.harbor.rollout_dispatcher as harbor_dispatcher
from loguru import logger
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector


# Every tokenizer call the whole-trajectory and step-wise collectors make per step. __init__'s
# base-conversation encode is excluded: it runs once at startup, not inside generate.
EXPECTED_TOKENIZE_REGIONS = {"skyrl_gym": 7, "step_wise": 3}

# Tokenizer calls that are deliberately NOT inside a rollout_tokenize region, by the function that
# holds them. Anything else is a leak, and the point of listing them is that adding one is a
# decision someone has to write down here.
TOKENIZE_EXEMPT_FUNCTIONS = {"__init__"}


class FakeClock:
    """A stand-in for the ``time`` timing_module, installed only on timing_observability.

    Follows tests/cpu/utils/test_timer.py. Durations here are exact rather than approximate, so
    every bound below is an equality and none of them can flake on a loaded runner --
    TESTING-core.md bans time.sleep() in tests for exactly that reason. ``advance`` is callable from
    the executor thread, which is what lets a test say "the environment took 0.1 s" without one.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self.reads = 0

    def perf_counter(self) -> float:
        self.reads += 1
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def clock(monkeypatch):
    """Bind a scripted clock to the module under test and nothing else.

    Patching time globally would also retime the asyncio event loop, which schedules off the same
    clock.
    """

    fake = FakeClock()
    monkeypatch.setattr(timing_module, "time", fake)
    return fake


class _InstrumentedRunner(TrajectoryRunner):
    """Stands in for SkyRLGymTrajectoryRunner: declares its call sites bracketed.

    Every subclass re-declares the flag because TrajectoryRunner.__init_subclass__ revokes it from
    anything that overrides _run without saying so -- which is the production semantics, and what
    keeps a new runner from publishing a seeded all-zero decomposition it never measured.
    """

    generate_spans_instrumented = True


def _tokenize_regions(module_name: str) -> list[str]:
    """The body of every ``with rollout_span("rollout_tokenize")`` block, as source text."""
    module = importlib.import_module(f"skyrl_train.trajectory_runners.{module_name}")

    lines = inspect.getsource(module).splitlines()
    regions: list[str] = []
    for index, line in enumerate(lines):
        if 'rollout_span("rollout_tokenize")' not in line:
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and (len(following) - len(following.lstrip())) <= indent:
                break
            body.append(following)
        regions.append("\n".join(body))
    return regions


def test_every_generate_span_is_registered_in_the_timing_tree():
    """An unregistered span is dropped in silence.

    phase_timing_observations filters on membership in TIMING_PARENTS, so a span missing from it
    publishes nothing at all -- indistinguishable from a phase that costs nothing.
    """
    for name in (*GENERATE_SPANS, *GENERATE_NESTED_SPANS, "generate_span_residual"):
        assert name in TIMING_PARENTS, f"{name} would be silently dropped"
    assert TIMING_PARENTS["generate"] == "step"
    for name in (*GENERATE_SPANS, "generate_span_residual"):
        assert TIMING_PARENTS[name] == "generate"


def test_every_child_of_generate_is_either_a_measured_span_or_the_residual():
    """Guards the two halves against drift.

    A name registered under generate but absent from GENERATE_SPANS is excluded from the residual
    arithmetic, so its time would be counted twice -- once in its own row and again inside the
    residual. That is the -1703.8 s bug from the policy tree, one phase over.
    """
    children = {name for name, parent in TIMING_PARENTS.items() if parent == "generate"}
    assert children == set(GENERATE_SPANS) | {"generate_span_residual"}


def test_the_nested_spans_hang_off_a_disjoint_one_and_stay_out_of_the_residual_set():
    """They are INCLUSIVE children: the parent wall already contains them."""
    assert TIMING_PARENTS["rollout_tokenize"] == "rollout_collect"
    assert TIMING_PARENTS["rollout_retain"] == "rollout_finalize"
    for name in GENERATE_NESTED_SPANS:
        assert name not in GENERATE_SPANS
        assert TIMING_PARENTS[name] in GENERATE_SPANS


def test_concurrent_await_sums_never_reach_the_span_tree():
    """rollout_engine_await is summed over up to 4,096 in-flight coroutines.

    At E6 geometry that is order 1e5 seconds against a ~98 s parent. Registered as a phase it would
    reach W&B (which carries no attributes, so nothing could mark it), every callback, and finelog
    under clock_domain="inclusive_wall" -- a containment claim that is false for a concurrent sum.
    """
    for counter in ROLLOUT_COUNTERS:
        assert counter not in TIMING_PARENTS
        assert counter not in GENERATE_SPANS
        assert counter not in GENERATE_NESTED_SPANS
        assert counter not in POLICY_TRAIN_SPANS
    # And behaviourally: handed to the phase publisher they produce no observation at all.
    assert phase_timing_observations({name: 1e5 for name in ROLLOUT_COUNTERS}) == ()


def test_no_bound_accumulator_records_nothing_and_costs_nothing():
    """Eval and the fully-async trainer run through the same regions with nothing bound."""
    assert ROLLOUT_TIMINGS.get() is None
    with rollout_span("rollout_collect"):
        pass
    with rollout_wait(ROLLOUT_ENGINE_AWAIT):
        pass
    with rollout_trajectory():
        pass
    assert ROLLOUT_TIMINGS.get() is None


def test_a_wait_records_an_exact_matched_triple_so_a_mean_and_a_tail_are_derivable():
    """Sum, count and tail -- not a mean: means do not compose, and one wait can issue several
    requests."""
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        for _ in range(3):
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                pass
    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 3.0
    assert timings.durations == {}, "a concurrent wait must never land on the duration path"


def test_spans_accumulate_across_regions_and_waits_stay_separate():
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        for _ in range(3):
            with rollout_span("rollout_tokenize"):
                pass
        with rollout_wait(ROLLOUT_ENV_AWAIT):
            pass
    assert set(timings.durations) == {"rollout_tokenize"}
    assert set(timings.counters) == {f"{ROLLOUT_ENV_AWAIT}_seconds_sum", f"{ROLLOUT_ENV_AWAIT}_count"}


# --- the tail, which is the whole point on a tail-latency-bound phase ----------------------------


def test_the_per_trajectory_max_is_a_tail_and_not_another_sum(clock):
    """F20: generate is tail-latency-bound -- the wall is set by the LAST trajectory to finish.

    A mean over 4,096 trajectories cannot tell a uniformly slow rollout from a fast one with three
    stragglers, and those two have opposite fixes ("buy more engines" vs "fix the tail"). This is
    the assertion that the max is reduced with max: given trajectories of 3 waits and 1 wait, a sum
    would report 4 and only a max reports 3.
    """
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        for waits in (3, 1):
            with rollout_trajectory():
                for _ in range(waits):
                    with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                        clock.advance(1.0)
    total = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_sum"]
    tail = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"]
    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 4.0
    assert total == 4.0, "every wait must reach the sum"
    assert tail == 3.0, "the tail is the LONGEST trajectory's total; folded with + it would be 4.0"


def test_a_wait_outside_any_trajectory_still_sums_but_contributes_no_tail():
    """The batched path and any future unscoped call site stay measurable rather than crash."""
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        with rollout_wait(ROLLOUT_ENGINE_AWAIT):
            pass
    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 1.0
    assert f"{ROLLOUT_ENGINE_AWAIT}_seconds_max" not in timings.counters


def test_concurrent_trajectories_do_not_pool_their_waits(clock):
    """Each trajectory is its own task and so its own context copy. Sharing one dict would make
    every trajectory's tail the whole rollout's sum."""

    async def _fanout():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):

            async def _trajectory(waits: int):
                with rollout_trajectory():
                    for _ in range(waits):
                        # Yield BETWEEN brackets, not inside one. The tasks still interleave --
                        # which is what makes this a context-isolation test -- but each bracket is
                        # atomic on the loop, so a shared clock cannot charge one wait for another's
                        # overlap and confound the question being asked.
                        await asyncio.sleep(0)
                        with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                            clock.advance(1.0)

            await asyncio.gather(_trajectory(3), _trajectory(1))
        return timings

    timings = asyncio.run(_fanout())
    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 4.0
    tail = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"]
    total = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_sum"]
    assert total == 4.0
    assert tail == 3.0, "pooled into one dict, every trajectory's tail would be the whole rollout"


# --- the env split, which is where a single bracket lies ----------------------------------------


def _gated(gate, work):
    """Hold the pool's single worker until every call has been submitted.

    A scripted clock removes timing nondeterminism but not SCHEDULING nondeterminism: without this
    the pool can finish call 1 before call 2 has been submitted, so call 2 records no queue and the
    test measures thread scheduling rather than the split under test.
    """
    gate.wait(5.0)
    work()


async def _submit_then_release(pool, gate, calls):
    """Start every call, let each reach its await, then release the worker."""
    pending = [asyncio.ensure_future(call(pool)) for call in calls]
    for _ in range(3):
        await asyncio.sleep(0)
    gate.set()
    return await asyncio.gather(*pending)


def test_env_queueing_is_not_reported_as_environment_runtime(clock):
    """🚨 The regression this split exists for.

    One bracket around run_in_executor measures submission-to-resumption. With N concurrent
    trajectories against W pool threads that sum is O(N^2/W): at E6 geometry, 4,096 coroutines and a
    32-worker pool produce tens of seconds of "env await" for an environment that did not change,
    and it moves with the batch size.

    Here one worker serialises 4 calls of 1.0 s, so the queue each sees is 0, 1, 2, 3 -- 6.0 s of
    backlog against 4.0 s of environment. A single bracket reports all 10.0 s as environment.
    """
    calls = 4
    gate = threading.Event()

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                await _submit_then_release(
                    pool,
                    gate,
                    [lambda p: timed_env_call(p, _gated, gate, lambda: clock.advance(1.0))] * calls,
                )
        return timings

    timings = asyncio.run(_drive())
    executed = timings.counters[f"{ROLLOUT_ENV_EXEC}_seconds_sum"]
    queued = timings.counters[f"{ROLLOUT_ENV_QUEUE}_seconds_sum"]
    resumed = timings.counters[f"{ROLLOUT_ENV_RESUME}_seconds_sum"]
    awaited = timings.counters[f"{ROLLOUT_ENV_AWAIT}_seconds_sum"]

    assert timings.counters[f"{ROLLOUT_ENV_AWAIT}_count"] == float(calls)
    # The environment, and only the environment: LINEAR in the number of calls.
    assert executed == float(calls), f"exec {executed} should be the {calls} x 1.0 s of env work"
    # The pool's backlog: 0 + 1 + 2 + 3, quadratic in the number of calls.
    assert queued == 6.0, f"queue {queued} should be the serialised backlog"
    # The three terms partition the caller-observed wait.
    assert awaited == queued + executed + resumed


def test_env_execution_is_stamped_on_the_pool_thread_not_the_loop_thread(clock):
    """The stamps have to be taken where the work runs, or exec absorbs the queue again.

    Two calls through one worker: the first advances 1.0 s, the second does nothing. Stamped on the
    pool thread, exec is 1.0 and the second call's whole wait is queue. Stamped on the loop thread
    exec would be 2.0, which is what this bound separates.
    """
    ran_on: list[str] = []
    gate = threading.Event()

    def _slow():
        ran_on.append(threading.current_thread().name)
        clock.advance(1.0)

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                await _submit_then_release(
                    pool,
                    gate,
                    [
                        lambda p: timed_env_call(p, _gated, gate, _slow),
                        lambda p: timed_env_call(p, _gated, gate, lambda: None),
                    ],
                )
        return timings

    timings = asyncio.run(_drive())
    assert ran_on and ran_on[0] != threading.main_thread().name, "the work did not run on the pool"
    assert timings.counters[f"{ROLLOUT_ENV_EXEC}_seconds_sum"] == 1.0
    assert timings.counters[f"{ROLLOUT_ENV_QUEUE}_seconds_sum"] == 1.0, "the second call waited out the first"


def test_with_no_executor_the_whole_env_wait_is_execution(clock):
    """Despite the `await`, this is a synchronous call on the loop thread: no queue, no resume."""

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            await timed_env_call(None, clock.advance, 2.0)
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters[f"{ROLLOUT_ENV_QUEUE}_seconds_sum"] == 0.0
    assert timings.counters[f"{ROLLOUT_ENV_RESUME}_seconds_sum"] == 0.0
    # A lower bound, so recording (0, 0, 0) fails. approx(0.02, abs=0.05) admitted zero.
    assert timings.counters[f"{ROLLOUT_ENV_EXEC}_seconds_sum"] == 2.0


def test_an_env_call_that_raises_still_records_its_wait(clock):
    """A failing environment is exactly when the number is wanted."""

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(ValueError):
                    await timed_env_call(pool, _raise_value_error, clock)
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters[f"{ROLLOUT_ENV_AWAIT}_count"] == 1.0
    assert timings.counters[f"{ROLLOUT_ENV_EXEC}_seconds_sum"] == 1.0, "a failing env still ran for 1 s"


def _raise_value_error(clock):
    clock.advance(1.0)
    raise ValueError("environment failed")


def _block_until(entered, release):
    """Signal that the pool thread is inside func, then wait to be let go."""
    entered.set()
    release.wait(5.0)


def test_a_cancelled_env_call_records_nothing_rather_than_inventing_a_queue():
    """A cancellation is not an environment call.

    Attributing its wait to rollout_env_queue -- documented as "pool too small / env slow" -- would
    read as executor undersizing, and counting it would inflate the denominator of every mean.
    """

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                entered, release = threading.Event(), threading.Event()
                task = asyncio.ensure_future(timed_env_call(pool, _block_until, entered, release))
                await asyncio.to_thread(entered.wait, 5.0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters == {}, f"a cancelled call recorded {timings.counters}"


def test_an_executor_that_cannot_run_the_work_records_nothing_and_still_raises():
    """A rejected submission has no execution time to report, and inventing one is worse than a gap."""

    async def _drive():
        timings = RolloutTimings()
        pool = ThreadPoolExecutor(max_workers=1)
        pool.shutdown()
        with rollout_timings_scope(timings):
            with pytest.raises(RuntimeError):
                await timed_env_call(pool, lambda: None)
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters == {}


def test_the_residual_is_published_as_an_exclusive_remainder():
    """It is what the parent's wall does NOT contain, and it may be negative.

    WorkerTimingSink already selects the domain per name so a consumer cannot sum a child into its
    parent twice; policy_span_residual ships exclusive_wall and this is the same shape.
    """

    rows: list[dict[str, str]] = []
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            timing_module,
            "phase_duration",
            type("_H", (), {"record": lambda self, v, attributes: rows.append(attributes)})(),
        )
        timing_module.FinelogTimingSink().publish(
            phase_timing_observations({"generate": 98.0, "rollout_collect": 90.0, "generate_span_residual": 8.0}),
            step=3,
        )
    finally:
        monkey.undo()

    domains = {row["phase"]: row["clock_domain"] for row in rows}
    assert domains["generate_span_residual"] == "exclusive_wall"
    assert domains["rollout_collect"] == "inclusive_wall"


def test_an_instrumented_flag_is_not_inherited_by_a_runner_that_replaces_the_body():
    """The flag certifies the call sites in _run. A subclass that overrides _run without saying so
    would inherit True and publish a seeded all-zero decomposition it never measured -- the measured
    zero lie, made indistinguishable from truth by the explicit zeros."""

    class _Replaces(_InstrumentedRunner):
        async def _run(self, input_batch, disable_tqdm: bool = False):
            return {}

    class _Reaffirms(_InstrumentedRunner):
        generate_spans_instrumented = True

        async def _run(self, input_batch, disable_tqdm: bool = False):
            return {}

    class _KeepsTheBody(_InstrumentedRunner):
        pass

    assert _Replaces.generate_spans_instrumented is False
    assert _Reaffirms.generate_spans_instrumented is True
    assert _KeepsTheBody.generate_spans_instrumented is True


def test_timed_env_call_passes_arguments_and_returns_the_result_unchanged():
    """The stamping closure must be transparent, or it is a behaviour change wearing a telemetry
    costume."""

    async def _drive(executor):
        with rollout_timings_scope(RolloutTimings()):
            return await timed_env_call(executor, lambda a, b, c=0: (a, b, c), 1, 2, c=3)

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert asyncio.run(_drive(pool)) == (1, 2, 3)
    assert asyncio.run(_drive(None)) == (1, 2, 3)


# --- measured zero vs never measured ------------------------------------------------------------


def test_a_supported_runner_seeds_explicit_zeros_for_every_leaf():
    """A leaf that is genuinely zero -- no retokenization configured, no environment executor --
    must publish 0.0, or a consumer cannot tell it from a call site nobody bracketed."""
    timings = RolloutTimings()
    timings.mark_supported()
    assert set(timings.durations) == set(GENERATE_SPANS) | set(GENERATE_NESTED_SPANS)
    assert set(timings.counters) == set(ROLLOUT_COUNTERS)
    assert set(timings.durations.values()) == {0.0}
    assert set(timings.counters.values()) == {0.0}


def test_an_uninstrumented_runner_publishes_nothing_not_a_full_residual():
    """🚨 Absence is the honest signal.

    Harbor and MiniSwe have not bracketed their call sites. Left to the generic path they would
    publish generate_span_residual == generate with every leaf MISSING, which reads as "generate is
    entirely unaccounted for" -- a claim about the rollout when it is a fact about the instrument.
    """

    class _Unbracketed(TrajectoryRunner):
        async def _run(self, input_batch, disable_tqdm: bool = False):
            with rollout_span("rollout_collect"):
                pass
            return {}

    assert _Unbracketed.generate_spans_instrumented is False
    timings = RolloutTimings()
    asyncio.run(_Unbracketed().run({}, phase_timings=timings))
    assert timings.supported is False
    assert timings.durations == {}, "nothing may be measured through an unbracketed runner"

    all_timings: dict[str, float] = {}
    counters: dict[str, float] = {}
    record_generate_spans(timings, 98.0, all_timings, counters)
    assert all_timings == {}, "no leaves AND no residual"
    assert counters == {}


def test_the_skyrl_gym_runner_declares_itself_instrumented():
    """The one runner whose every engine, environment and tokenizer call site is bracketed."""

    assert SkyRLGymTrajectoryRunner.generate_spans_instrumented is True


# --- the residual -------------------------------------------------------------------------------


def test_the_residual_closes_generate_and_is_signed():
    """The residual is the audit. Clamping would hide over-coverage, which is the one thing it is
    here to detect."""
    timings = RolloutTimings(
        durations={"rollout_collect": 6.0, "rollout_assemble": 1.0, "rollout_finalize": 0.5, "rollout_tokenize": 2.0},
        supported=True,
    )
    all_timings: dict[str, float] = {}
    record_generate_spans(timings, 10.0, all_timings, {})
    assert all_timings["generate_span_residual"] == pytest.approx(2.5)
    # rollout_tokenize is carried but not subtracted: it is inside rollout_collect.
    assert all_timings["rollout_tokenize"] == 2.0

    over = RolloutTimings(durations={"rollout_collect": 6.0, "rollout_assemble": 1.0}, supported=True)
    over_timings: dict[str, float] = {}
    record_generate_spans(over, 4.0, over_timings, {})
    assert over_timings["generate_span_residual"] == pytest.approx(-3.0)


def test_repeated_generate_calls_in_one_step_accumulate_but_tails_fold_with_max():
    """Group admission and dynamic sampling resample without closing the step, and Timer accumulates
    generate the same way. Assigning would report the last rollout against the summed wall -- and
    ADDING two tails would invent a third that no trajectory ever had."""
    all_timings: dict[str, float] = {}
    counters: dict[str, float] = {}
    for tail in (5.0, 3.0):
        timings = RolloutTimings(
            durations={"rollout_collect": 4.0, "rollout_assemble": 1.0},
            counters={
                f"{ROLLOUT_ENGINE_AWAIT}_count": 8.0,
                f"{ROLLOUT_ENGINE_AWAIT}_seconds_max": tail,
            },
            supported=True,
        )
        record_generate_spans(timings, 6.0, all_timings, counters)
    assert all_timings["rollout_collect"] == 8.0
    assert all_timings["generate_span_residual"] == pytest.approx(2.0)
    assert counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 16.0
    assert counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"] == 5.0, "tails fold with max, not with +"


# --- publishing ---------------------------------------------------------------------------------


def test_driver_counter_rows_carry_the_trainer_role_and_no_rank():
    """publish_worker_counters hardcodes role="worker" and requires a rank. The driver has neither,
    and a consumer that read rank off these rows raised KeyError on the first correct one."""

    waits: list[tuple[float, dict[str, str]]] = []
    counts: list[tuple[float, dict[str, str]]] = []
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            timing_module,
            "rollout_wait_seconds",
            type("_H", (), {"record": lambda self, v, attributes: waits.append((v, attributes))})(),
        )
        monkey.setattr(
            timing_module,
            "rollout_count",
            type("_H", (), {"record": lambda self, v, attributes: counts.append((v, attributes))})(),
        )
        monkey.setattr(
            timing_module,
            "phase_duration",
            type("_H", (), {"record": lambda *a, **k: pytest.fail("wrong instrument")})(),
        )
        monkey.setattr(
            timing_module,
            "policy_step_counter",
            type("_H", (), {"record": lambda *a, **k: pytest.fail("policy instrument is worker-only")})(),
        )
        publish_driver_counters(
            {
                f"{ROLLOUT_ENGINE_AWAIT}_seconds_sum": 1.5e5,
                f"{ROLLOUT_ENGINE_AWAIT}_seconds_max": 41.2,
                f"{ROLLOUT_ENGINE_AWAIT}_count": 4096.0,
            },
            step=11,
        )
    finally:
        monkey.undo()

    # A tail is seconds, so it rides the seconds instrument, not the unit-1 one.
    assert sorted(value for value, _ in waits) == [41.2, 1.5e5]
    assert [value for value, _ in counts] == [4096.0]
    for _, attributes in waits + counts:
        assert attributes["role"] == "trainer"
        assert attributes["step"] == "11"
        assert "rank" not in attributes
        # No phase, parent or clock_domain either: those invite a consumer to band a 1e5-second sum
        # into a 98 s parent.
        assert set(attributes) == {"counter", "role", "step"}


def test_an_unregistered_counter_is_dropped_and_warned_not_fatal(monkeypatch):
    """It must NOT raise: this runs in the trainer's step epilogue, after the step is paid for.

    Raising there turns a telemetry-naming typo into a killed training run. The name check is
    statically decidable and is asserted at import instead (see the ROLLOUT_COUNTERS loop at the
    bottom of timing_observability). Here we assert the runtime behaviour: the bad row is dropped,
    the good row still publishes, and a warning says so.
    """

    warned: list[str] = []
    # The publisher logs lazily (`logger.warning("%r ...", name)`), matching publish_worker_spans, so
    # the double has to render the args or every assertion below matches the format string instead.
    monkeypatch.setattr(timing_module.logger, "warning", lambda m, *a, **k: warned.append(str(m) % a if a else str(m)))
    published: list[tuple[float, dict]] = []
    monkeypatch.setattr(
        timing_module,
        "rollout_count",
        type("_H", (), {"record": lambda self, v, attributes: published.append((v, attributes))})(),
    )

    publish_driver_counters({"rollout_engine_await_requests": 1.0, f"{ROLLOUT_ENGINE_AWAIT}_count": 7.0}, step=1)

    assert [v for v, _ in published] == [7.0], "the VALID row must still publish"
    assert any("rollout_engine_await_requests" in w for w in warned), "the dropped row must warn"


def test_unconfigured_telemetry_is_announced_once_rather_than_publishing_in_silence(monkeypatch):
    """R10, on the driver.

    On an unconfigured runtime record() is a no-op, flush() still returns True and lost_records stays
    at 0: every signal reads healthy and the run produces no rows at all. That is a whole step spent
    to learn nothing, and the only defence is saying so out loud.
    """

    warned: list[str] = []
    # The publisher logs lazily (`logger.warning("%r ...", name)`), matching publish_worker_spans, so
    # the double has to render the args or every assertion below matches the format string instead.
    monkeypatch.setattr(timing_module.logger, "warning", lambda m, *a, **k: warned.append(str(m) % a if a else str(m)))
    monkeypatch.setattr(timing_module, "unconfigured_telemetry_reason", lambda: "endpoint is unset")
    monkeypatch.setattr(timing_module, "rollout_count", type("_H", (), {"record": lambda *a, **k: None})())
    monkeypatch.setattr(timing_module, "_driver_counter_check_done", False)

    for _ in range(3):
        publish_driver_counters({f"{ROLLOUT_ENGINE_AWAIT}_count": 1.0}, step=1)

    assert [w for w in warned if "endpoint is unset" in w], "the inert-telemetry case must warn"
    assert len([w for w in warned if "endpoint is unset" in w]) == 1, "once per process, not once per step"


def test_publishing_no_counters_is_a_no_op(monkeypatch):
    """Asserted, not merely executed: an empty step must reach no instrument at all."""

    monkeypatch.setattr(
        timing_module,
        "rollout_count",
        type("_H", (), {"record": lambda *a, **k: pytest.fail("published an empty step")})(),
    )
    monkeypatch.setattr(
        timing_module,
        "rollout_wait_seconds",
        type("_H", (), {"record": lambda *a, **k: pytest.fail("published an empty step")})(),
    )
    publish_driver_counters({}, step=1)


# --- call-site coverage -------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", sorted(EXPECTED_TOKENIZE_REGIONS))
def test_no_tokenize_region_contains_an_await(module_name):
    """Only a region that holds the event-loop thread throughout may be summed across coroutines.

    An await inside one would let it overlap its siblings, and the sum would exceed the wall it is
    supposed to partition -- the -1703.8 s bug in asyncio costume.
    """
    regions = _tokenize_regions(module_name)
    expected = EXPECTED_TOKENIZE_REGIONS[module_name]
    assert len(regions) == expected, f"expected {expected} wrapped sites in {module_name}"
    for region in regions:
        assert not re.search(r"\bawait\b", region), f"await inside a summed region:\n{region}"


@pytest.mark.parametrize("module_name", sorted(EXPECTED_TOKENIZE_REGIONS))
def test_every_tokenizer_call_is_inside_a_tokenize_region(module_name):
    """🚨 Counts the TOKENIZER CALLS, not the wrappers.

    Asserting the number of ``with rollout_span("rollout_tokenize")`` blocks passes unchanged when
    somebody adds an unwrapped ``self.tokenizer.encode`` beside them -- the leaf then under-reports
    and the suite stays green. This walks the AST instead: every ``self.tokenizer.<anything>()`` call
    must fall inside the line range of a tokenize region, or be named in TOKENIZE_EXEMPT_FUNCTIONS.
    """
    module = importlib.import_module(f"skyrl_train.trajectory_runners.{module_name}")
    source = inspect.getsource(module)
    tree = ast.parse(source)

    regions: list[range] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "rollout_span"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "rollout_tokenize"
            ):
                regions.append(range(node.lineno, node.end_lineno + 1))

    enclosing: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, node.end_lineno + 1):
                enclosing.setdefault(line, node.name)

    leaked: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "tokenizer"):
            continue
        if enclosing.get(node.lineno) in TOKENIZE_EXEMPT_FUNCTIONS:
            continue
        if not any(node.lineno in region for region in regions):
            leaked.append(f"{module_name}:{node.lineno} self.tokenizer.{node.func.attr}() outside a tokenize region")
    assert not leaked, "\n".join(leaked)


def test_the_env_call_is_bracketed_on_the_event_loop_thread_and_split_three_ways():
    """A ContextVar does not cross run_in_executor, so a span opened inside the executor thread
    finds no accumulator. And a single bracket around the whole submission reports the pool's
    backlog as environment runtime."""

    source = inspect.getsource(SkyRLGymTrajectoryRunner._run_in_executor_if_available)
    assert "timed_env_call(" in source
    # The method's own NAME contains run_in_executor, so match the call, not the substring.
    assert "loop.run_in_executor(" not in source, "the split lives in timed_env_call, not beside it"

    assert "loop.run_in_executor(" in inspect.getsource(timed_env_call)


# Functions whose body is one trajectory. Every engine and environment wait must be reachable only
# from inside one of these, or its time lands in the sums with no tail beside it.
TRAJECTORY_SCOPED_FUNCTIONS = {"agent_loop", "collect_batched"}


@pytest.mark.parametrize("module_name", ["skyrl_gym", "step_wise"])
def test_every_wait_site_is_inside_a_trajectory_scope(module_name):
    """🚨 The regression test for a max published smaller than its own mean.

    The batched path once scoped only its engine await, leaving env.init / env.step / env.close
    outside it. Those waits still reached rollout_env_await_seconds_sum and _count, but never the
    per-trajectory dict -- so _seconds_max stayed at the 0.0 that mark_supported seeds, and the row
    published as a MEASURED zero beside a non-zero mean. Counting decorators would not have caught
    it; this walks the call sites.
    """
    module = importlib.import_module(f"skyrl_train.trajectory_runners.{module_name}")
    tree = ast.parse(inspect.getsource(module))

    scoped: set[int] = set()
    holder: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for line in range(node.lineno, node.end_lineno + 1):
            holder.setdefault(line, node.name)
        if node.name not in TRAJECTORY_SCOPED_FUNCTIONS:
            continue
        assert any(isinstance(d, ast.Name) and d.id == "traced_trajectory" for d in node.decorator_list), (
            f"{module_name}.{node.name} holds waits but is not @traced_trajectory"
        )
        scoped.update(range(node.lineno, node.end_lineno + 1))

    unscoped: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in {"_run_in_executor_if_available", "rollout_wait"}:
            continue
        if node.lineno not in scoped:
            unscoped.append(
                f"{module_name}:{node.lineno} {name}() in {holder.get(node.lineno)!r}, outside a trajectory"
            )
    assert not unscoped, "\n".join(unscoped)


def test_every_agent_loop_scopes_its_own_trajectory():
    """Without it there is no tail, only a mean -- and F20 says the mean is the wrong statistic."""

    for owner in (SkyRLGymTrajectoryRunner, StepWiseRolloutCollector):
        assert getattr(owner.agent_loop, "__wrapped__", None) is not None, (
            f"{owner.__name__}.agent_loop is not scoped as a trajectory"
        )
    assert getattr(SkyRLGymTrajectoryRunner.collect_batched, "__wrapped__", None) is not None


def test_a_region_emits_no_log_record():
    """utils.utils.Timer logs two loguru records per region by default.

    At ~4e4 regions per step that is ~8e4 synchronous log records on the event-loop thread, inside
    the phase being measured. This asserts the observable consequence rather than the flag.
    """

    records: list[object] = []
    sink_id = logger.add(records.append, level="TRACE")
    try:
        with rollout_timings_scope(RolloutTimings()):
            for _ in range(5):
                with rollout_span("rollout_tokenize"):
                    pass
                with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                    pass
                with rollout_trajectory():
                    pass
    finally:
        logger.remove(sink_id)
    assert records == [], f"{len(records)} log record(s) emitted inside the measured phase"


# --- reentrancy ---------------------------------------------------------------------------------


class _CountingRunner(_InstrumentedRunner):
    generate_spans_instrumented = True

    """A runner whose only work is a fixed number of awaited engine calls."""

    def __init__(self, awaits: int) -> None:
        self._awaits = awaits

    async def _run(self, input_batch, disable_tqdm: bool = False):
        with rollout_span("rollout_collect"):
            pass
        for _ in range(self._awaits):
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                await asyncio.sleep(0)
        return {}


class _ParameterisedRunner(_InstrumentedRunner):
    generate_spans_instrumented = True

    """One instance, many concurrent run() calls, each asking for a different number of awaits."""

    async def _run(self, input_batch, disable_tqdm: bool = False):
        with rollout_span("rollout_collect"):
            pass
        for _ in range(input_batch["awaits"]):
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                await asyncio.sleep(0)
        return {}


def test_concurrent_run_calls_on_ONE_INSTANCE_never_share_an_accumulator():
    """🚨 One instance, three overlapping calls -- which is the actual shape in production.

    The fully-async trainer keeps up to 768 run() calls in flight on a SINGLE runner and awaits
    eval(), which calls run() on that same instance. Three separate instances would pass even if the
    accumulator were stashed on self, which is the defect this test exists to exclude.
    """
    runner = _ParameterisedRunner()

    async def _interleaved():
        first, second = RolloutTimings(), RolloutTimings()
        await asyncio.gather(
            runner.run({"awaits": 3}, phase_timings=first),
            runner.run({"awaits": 1}, phase_timings=second),
            runner.run({"awaits": 2}),
        )
        return first, second

    first, second = asyncio.run(_interleaved())
    assert first.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 3.0
    assert second.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 1.0
    assert first.durations["rollout_collect"] >= 0.0
    # ⚠️ Do NOT assert ROLLOUT_TIMINGS.get() is None out here: asyncio.run executes in a COPIED
    # context, so this reads None whether or not the scope leaked. The leak is only observable
    # INSIDE the loop, which is what _interleaved checks and returns.
    assert first is not second, "the two calls shared one accumulator"


def test_a_failing_run_on_a_shared_instance_leaves_the_next_one_clean():
    """The scope must be exception-safe on the instance, not merely on the happy path."""
    runner = _ParameterisedRunner()

    class _Failing(_InstrumentedRunner):
        generate_spans_instrumented = True

        async def _run(self, input_batch, disable_tqdm: bool = False):
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                await asyncio.sleep(0)
            raise RuntimeError("rollout failed")

    async def _sequence():
        broken = RolloutTimings()
        with pytest.raises(RuntimeError):
            await _Failing().run({}, phase_timings=broken)
        after = RolloutTimings()
        await runner.run({"awaits": 2}, phase_timings=after)
        return broken, after

    broken, after = asyncio.run(_sequence())
    assert broken.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 1.0
    assert after.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 2.0, "the failed call polluted the next one"


def test_an_unbound_run_inside_a_bound_one_records_nothing():
    """The harbor dispatcher calls run() on a sub-runner from inside a run(). Binding is per call,
    so the inner one accumulates into the dict it was handed -- here, none."""

    class _Nesting(_InstrumentedRunner):
        generate_spans_instrumented = True

        async def _run(self, input_batch, disable_tqdm: bool = False):
            await _CountingRunner(5).run({})
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                await asyncio.sleep(0)
            return {}

    outer = RolloutTimings()
    asyncio.run(_Nesting().run({}, phase_timings=outer))
    assert outer.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 1.0


def test_the_shared_epilogue_is_measured_and_retention_is_its_own_leaf(clock):
    """retain_trajectories is on by default and writes the whole batch -- ~8 MiB at E6 -- blocking,
    on the event-loop thread. Without its own leaf it lands in the residual, which already has
    several known occupants and so explains nothing.

    Driven through a real TrajectorySink rather than by monkeypatching the retention helper: the
    sink is the public seam, and proving wiring by patching an internal proves only that the patch
    took.
    """

    class _RecordingSink:
        """A fake standing in for TrajectorySink: the seam retain_trajectories actually uses.

        Duck-typed rather than a subclass, because TrajectorySink.__init__ starts a publication
        subprocess. retain_trajectories reads `.retain` and `.config.required`; the runner reads
        `.bind_runner`. Nothing else is touched.
        """

        config = SimpleNamespace(required=True)

        def __init__(self) -> None:
            self.runner: str | None = None
            self.batches = 0

        def bind_runner(self, runner_name: str) -> None:
            self.runner = runner_name

        def retain(self, input_batch, output):
            self.batches += 1
            clock.advance(1.0)
            return {}

    class _Sinking(_InstrumentedRunner):
        generate_spans_instrumented = True

        async def _run(self, input_batch, disable_tqdm: bool = False):
            return {"response_ids": [], "loss_masks": [], "rollout_metrics": None, "rewards": []}

    runner = _Sinking()
    sink = _RecordingSink()
    runner.set_trajectory_sink(sink)
    timings = RolloutTimings()
    asyncio.run(runner.run({"prompts": [], "env_classes": [], "env_extras": []}, phase_timings=timings))

    assert sink.batches == 1, "the fake sink was never actually written to"
    assert sink.runner == "_Sinking", "the sink was never bound to this runner"
    assert timings.durations["rollout_retain"] == 1.0
    assert timings.durations["rollout_finalize"] >= timings.durations["rollout_retain"], (
        "retain nests inside finalize; a finalize smaller than its child means the brackets crossed"
    )


# --- wiring -------------------------------------------------------------------------------------


def test_the_fully_async_trainer_binds_no_accumulator():
    """Up to 768 concurrent run() calls. Accumulating overlapping walls into one dict decomposes
    nothing, and the residual it produced would be an arbitrary number with a units label."""

    source = inspect.getsource(FullyAsyncRayPPOTrainer._run_generate_for_a_group_loop)
    assert "trajectory_runner.run(" in source
    assert "phase_timings=" not in source


def test_the_harbor_dispatcher_forwards_no_accumulator_to_its_shards():
    """It runs K coordinators concurrently. Handing one accumulator to all of them sums overlapping
    walls, which is the same defect one layer down from the fully-async trainer."""

    assert "phase_timings" in inspect.signature(harbor_dispatcher.RolloutDispatcher.run).parameters, (
        "it must still accept the call the trainer makes on whatever holds the runner slot"
    )
    assert "phase_timings=" not in inspect.getsource(harbor_dispatcher), "no shard may be handed the accumulator"


def test_the_harbor_runner_protocol_declares_the_argument_the_trainer_passes():
    """A third runner written to this Protocol without it dies with TypeError on the FIRST generate
    of step 1 -- after full model and engine bring-up."""

    assert "phase_timings" in inspect.getsource(harbor_execution.HarborRunner)


def test_the_trainer_wires_the_generate_span_layer():
    """The wiring itself, which no behavioural test reaches: _train_loop needs a live Ray cluster,
    inference engines and a dataloader. Structural, but it catches the regressions that leave a
    green suite -- a residual computed against a wall that has not closed, or spans collected and
    then dropped on the floor."""

    source = inspect.getsource(RayPPOTrainer._train_loop)
    # The residual is generate minus its children, so it is computed after the Timer closes.
    assert source.index('Timer("generate"') < source.index("record_generate_spans(")
    assert "generate_timer.duration" in source
    assert "self.all_rollout_counters" in source

    generate = inspect.getsource(RayPPOTrainer.generate)
    assert "phase_timings=phase_timings" in generate

    # The counters ride their own publisher, next to the phase publish rather than inside it.
    assert "publish_driver_counters(self.all_rollout_counters" in source
