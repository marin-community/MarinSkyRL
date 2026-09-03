"""Sink-neutral timing observations and their publishing adapters."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

# Via skyrl_train.telemetry, NOT `from rigging import telemetry`: that module guards the import
# and falls back to inert_telemetry, because an installed rigging without the telemetry
# submodule raises ImportError. This module is imported at the top of trainer.py and worker.py,
# so an unguarded import here would take the whole trainer down on that install shape.
from skyrl_train.telemetry import TRAINER_ROLE, WORKER_ROLE, phase_duration, telemetry


TIMING_PARENTS: dict[str, str | None] = {
    "step": None,
    "generate": "step",
    # Inside generate, measured on the driver's event loop. See the generate tree below.
    "rollout_collect": "generate",
    "rollout_assemble": "generate",
    "rollout_finalize": "generate",
    "rollout_tokenize": "rollout_collect",
    "rollout_retain": "rollout_finalize",
    "generate_span_residual": "generate",
    "wait_for_generation_buffer": "step",
    "postprocess_trajectory_batch": "step",
    "convert_to_training_input": "step",
    "run_training": "step",
    "fwd_logprobs_values_reward": "run_training",
    "apply_reward_kl_penalty": "run_training",
    "compute_advantages_and_returns": "run_training",
    "train_critic_and_policy": "run_training",
    "critic_train": "train_critic_and_policy",
    "policy_train": "train_critic_and_policy",
    # Inside policy_train, measured on the worker rather than the driver. policy_train itself is
    # only a Ray dispatch plus a wait for the slowest worker, so these are where its time actually
    # goes. Published with role=worker and clock_domain=exclusive_wall; see WorkerSpanAccumulator.
    "policy_ppo_train": "policy_train",
    "policy_entry_barrier": "policy_ppo_train",
    # 🔻 policy_train, NOT policy_ppo_train. This is the cost of publishing the PREVIOUS step's
    # spans, and it is measured AFTER policy_ppo_train's total_seconds has been taken -- so its
    # parent's wall does not contain it, while the driver's policy_train wait does. Parented to
    # policy_ppo_train it made the exclusive children sum to parent + publish (up to the 1.0 s flush
    # cap of over-coverage), and the signed residual could not see it, because the residual is
    # computed from POLICY_TRAIN_SPANS which this is no longer a member of. A reader auditing the
    # tree saw a positive discrepancy and was told by the design notes that it meant double-counting
    # inside a child. It did not; it was a row on the wrong parent.
    "policy_span_publish": "policy_train",
    # Split at the seams _phase_diagnostics already marks. A single policy_training_step span would
    # report ~95% of policy_ppo_train and reproduce the same black box one level down, at the cost of
    # a full step.
    "policy_forward": "policy_ppo_train",
    "policy_backward": "policy_ppo_train",
    "policy_optimizer_step": "policy_ppo_train",
    "policy_entropy_allreduce": "policy_ppo_train",
    # Inclusive of forward/backward/optimizer/entropy below it -- it wraps training_step,
    # which contains them. Deliberately NOT in POLICY_TRAIN_SPANS: including it would count
    # its children twice and drive the residual to roughly -parent, which is exactly what the
    # first instrumented run showed (-1703 s against a 1706 s parent).
    "policy_training_step": "policy_ppo_train",
    "policy_metric_allreduce": "policy_ppo_train",
    "policy_final_barrier": "policy_ppo_train",
    "policy_span_residual": "policy_ppo_train",
    "policy_critic_overlap_train": "train_critic_and_policy",
    "sync_weights": "step",
    "offload_policy_model_to_cpu": "step",
    "dump_data_batch": "run_training",
    "init_weight_sync_state": None,
    "save_checkpoints": None,
    "cleanup_old_checkpoints": "save_checkpoints",
    "save_hf_model": None,
    "queue_hf_export": None,
    "eval": None,
    "update_ref_with_policy": None,
}


@dataclass(frozen=True)
class PhaseTiming:
    name: str
    duration_seconds: float
    root: str
    parent: str | None


class TimingSink(Protocol):
    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None: ...


class Tracker(Protocol):
    def log(self, metrics: Mapping[str, float], *, step: int, commit: bool) -> None: ...


def nearest_recorded_parent(name: str, recorded: Mapping[str, object]) -> str | None:
    parent = TIMING_PARENTS.get(name)
    while parent is not None and parent not in recorded:
        parent = TIMING_PARENTS.get(parent)
    return parent


def declared_root(name: str) -> str:
    root = name
    while TIMING_PARENTS.get(root) is not None:
        root = TIMING_PARENTS[root]
    return root


def phase_timing_observations(timings: Mapping[str, float]) -> tuple[PhaseTiming, ...]:
    """Preserve measured wall durations; async spans may overlap and are not additive."""
    known = {name: float(duration) for name, duration in timings.items() if name in TIMING_PARENTS}
    return tuple(
        PhaseTiming(name, duration, declared_root(name), nearest_recorded_parent(name, known))
        for name, duration in known.items()
    )


# A residual is what its parent's wall does NOT contain, so it is exclusive by construction and may
# legitimately be negative. WorkerTimingSink already selects the domain per name for exactly this
# reason -- mixing the two under one domain lets a consumer sum a child into its parent twice -- and
# policy_span_residual ships as exclusive_wall. The driver's residual is the same shape.
EXCLUSIVE_DRIVER_SPANS = frozenset({"generate_span_residual"})


class FinelogTimingSink:
    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None:
        for observation in observations:
            phase_duration.record(
                observation.duration_seconds,
                attributes={
                    "phase": observation.name,
                    "root": observation.root,
                    "parent": observation.parent or "",
                    "clock_domain": (
                        "exclusive_wall" if observation.name in EXCLUSIVE_DRIVER_SPANS else "inclusive_wall"
                    ),
                    "role": TRAINER_ROLE,
                    "step": str(step),
                },
            )


def publish_step_timings(timings: Mapping[str, float], step: int, sinks: Sequence[TimingSink] | None = None) -> None:
    observations = phase_timing_observations(timings)
    for sink in (FinelogTimingSink(),) if sinks is None else sinks:
        sink.publish(observations, step)


def publish_startup_timings(
    startup_timings: MutableMapping[str, float],
    step_timings: MutableMapping[str, float],
    *,
    step: int,
    tracker: Tracker,
    console: Callable[..., None],
) -> None:
    """Move step timings into startup timings, clear them, then publish them."""
    startup_timings.update(step_timings)
    step_timings.clear()
    if not startup_timings:
        return
    payload = {f"startup/{name}": duration for name, duration in startup_timings.items()}
    console(payload, step=step, kind="startup")
    tracker.log(payload, step=step, commit=False)


# --- Driver-side spans inside generate ----------------------------------------------------------
#
# generate is 64% of an E6 step and is measured as one wall. The regions below decompose it on the
# driver's event loop, where the rollout actually runs: SkyRLGymTrajectoryRunner fans thousands of
# agent_loop coroutines out over one loop thread, so the interesting question is how much of the
# wall is the fan-out (rollout_collect), how much is projecting the results into a trainer batch
# (rollout_assemble), how much is the shared output finalization (rollout_finalize), and how much is
# neither.
#
# 🚨 A SUM OVER CONCURRENT COROUTINES IS NOT A DURATION. The wait counters below are summed over up
# to 4,096 in-flight trajectories, so at E6 geometry they are order 1e5 seconds against a ~98 s
# parent. They must never reach self.all_timings, which feeds W&B (no attributes, so nothing can
# mark such a row), every callback, finelog (which stamps clock_domain="inclusive_wall", asserting
# containment that is false here) and tools/spans.py. They go to their own instruments as exact
# sum/count/max triples instead, and the mean wait is a division the consumer does.
#
# Only regions with NO await inside may be summed across coroutines, because only those cannot
# overlap: rollout_tokenize holds the loop thread for its whole extent. rollout_collect,
# rollout_assemble and rollout_finalize are single walls on the loop, not sums, so they are additive
# by construction.

# Disjoint top-level regions. Their sum is subtracted from generate to form the residual.
GENERATE_SPANS = ("rollout_collect", "rollout_assemble", "rollout_finalize")

# Inclusive children of one of the above. Published so the tree is navigable, and deliberately NOT
# subtracted from the residual -- doing so would count them twice. rollout_tokenize nests inside
# rollout_collect; rollout_retain nests inside rollout_finalize.
GENERATE_NESTED_SPANS = ("rollout_tokenize", "rollout_retain")

GENERATE_LEAF_SPANS = GENERATE_SPANS + GENERATE_NESTED_SPANS

ROLLOUT_ENGINE_AWAIT = "rollout_engine_await"
ROLLOUT_ENV_AWAIT = "rollout_env_await"

# The waits that carry a per-trajectory tail. A trajectory that closes without ever waiting
# contributes a real 0.0 to these, which is why they are enumerated rather than discovered from
# whatever happened to fire: an agent loop can return before its first engine await (an overlong
# prompt is rejected up front), and a missing row there would read as "not measured" when the true
# answer is zero.
ROLLOUT_TAILED_WAITS = (ROLLOUT_ENGINE_AWAIT, ROLLOUT_ENV_AWAIT)

# How many trajectory scopes closed. THE DENOMINATOR THAT MAKES THE TAIL COMPARABLE: `_count` counts
# timed CALLS, of which one trajectory makes several, so `sum/_count` is a mean per call while
# `_seconds_max` is a max per trajectory -- two different populations presented side by side. Divide
# the sum by this instead to get a per-trajectory mean the tail can actually be read against.
ROLLOUT_TRAJECTORY_COUNT = "rollout_trajectory_count"

# 🚨 rollout_env_await is deliberately THREE numbers, not one.
#
# A single bracket around run_in_executor measures submission-to-resumption, and at E6 geometry that
# is dominated by neither the environment nor anything actionable: 4,096 coroutines queue against a
# 32-worker pool, so the SUM of that bracket grows as N^2/(2W) and reports tens of seconds of "env
# await" for an environment that did not change. It is the same misattribution as reading a queue
# depth as a service time. The three terms below are disjoint and sum to that bracket exactly:
#
#   rollout_env_queue    submitted -> the pool thread picks the work up   (pool too small / env slow)
#   rollout_env_exec     the pool thread runs func                        (the environment itself)
#   rollout_env_resume   func returns -> the coroutine runs again         (the EVENT LOOP is behind)
#
# The resume term is the one worth the plumbing. It is a direct measurement of event-loop backlog,
# and so a second, independent witness for the question rollout_tokenize answers: whether the
# driver's single thread, not the engines, is what generate is waiting on.
ROLLOUT_ENV_QUEUE = "rollout_env_queue"
ROLLOUT_ENV_EXEC = "rollout_env_exec"
ROLLOUT_ENV_RESUME = "rollout_env_resume"


_SUM_SUFFIX = "_seconds_sum"
_MAX_SUFFIX = "_seconds_max"
_COUNT_SUFFIX = "_count"

# Matched triples so both a mean and a tail are derivable. Deliberately not `..._requests`: one
# timed await can issue several engine requests, because InferenceEngineClient retries an aborted
# generation in place and fails over to another engine when one dies.
#
# The _seconds_max rows are per-TRAJECTORY cumulative waits, not per-await maxima. F20 established
# that generate is tail-latency-bound -- the wall is set by the last trajectory to finish -- and a
# mean over 4,096 trajectories cannot tell a uniformly slow rollout from a fast one with three
# stragglers. That distinction is the difference between "buy more engines" and "fix the tail", so
# the tail is measured rather than inferred.
ROLLOUT_COUNTERS = (
    f"{ROLLOUT_ENGINE_AWAIT}{_SUM_SUFFIX}",
    f"{ROLLOUT_ENGINE_AWAIT}{_COUNT_SUFFIX}",
    f"{ROLLOUT_ENGINE_AWAIT}{_MAX_SUFFIX}",
    f"{ROLLOUT_ENV_AWAIT}{_SUM_SUFFIX}",
    f"{ROLLOUT_ENV_AWAIT}{_COUNT_SUFFIX}",
    f"{ROLLOUT_ENV_AWAIT}{_MAX_SUFFIX}",
    f"{ROLLOUT_ENV_QUEUE}{_SUM_SUFFIX}",
    f"{ROLLOUT_ENV_EXEC}{_SUM_SUFFIX}",
    f"{ROLLOUT_ENV_RESUME}{_SUM_SUFFIX}",
    ROLLOUT_TRAJECTORY_COUNT,
)

# The names rollout_wait accepts, DERIVED rather than listed: it emits a sum, a count and a tail, so
# a name is only valid here if all three rows are declared above.
#
# ⚠️ Hand-listing this was wrong and the first version shipped the error. The three env terms carry a
# `_seconds_sum` ONLY -- they are written by _record_env_wait directly, not through rollout_wait --
# so blessing them let `rollout_wait(ROLLOUT_ENV_QUEUE)` emit an undeclared `rollout_env_queue_count`
# that publish_driver_counters then drops. A guard against dropped rows that permits dropped rows.
ROLLOUT_WAIT_NAMES = tuple(
    name.removesuffix(_SUM_SUFFIX)
    for name in ROLLOUT_COUNTERS
    if name.endswith(_SUM_SUFFIX)
    and f"{name.removesuffix(_SUM_SUFFIX)}{_COUNT_SUFFIX}" in ROLLOUT_COUNTERS
    and f"{name.removesuffix(_SUM_SUFFIX)}{_MAX_SUFFIX}" in ROLLOUT_COUNTERS
)

# Physical instruments with the units the values actually carry. Not policy_train_count: that name
# says policy_train, and these are trainer-role rows from the rollout.
rollout_wait_seconds = telemetry.histogram("rollout_wait_seconds", unit="s")
rollout_count = telemetry.histogram("rollout_count", unit="1")


@dataclass
class RolloutTimings:
    """One generate call's accumulator.

    ``durations`` are disjoint regions on the event-loop thread and become phase rows.
    ``counters`` are sums, counts and maxima over concurrent coroutines and must never become phase
    rows.

    ``supported`` is what separates "measured zero" from "not measured". A runner that has not
    instrumented its call sites leaves it False, and nothing at all is published for that generate --
    no leaves and, critically, no residual. Publishing a residual equal to the whole parent from an
    uninstrumented runner would read as "generate is entirely unaccounted for", which is a claim
    about the rollout rather than about the instrument.
    """

    durations: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    supported: bool = False
    # What record_generate_spans has already folded, so a second fold of the SAME accumulator adds
    # only what is new.
    #
    # ⚠️ The trainer does not currently need this. `phase_timings = RolloutTimings()` sits INSIDE the
    # dataloader loop, and both resample paths (group admission, dynamic sampling) `continue`, which
    # re-enters the loop and builds a fresh accumulator -- so today every fold sees an empty
    # `_folded` and this is inert. It is here because the docstring promises the function
    # "accumulates rather than assigns", and that promise is only true for one of the two shapes a
    # caller can use. Without it, folding one accumulator twice subtracts the running total from a
    # single call's wall: two 6 s calls with 5 s of leaves each publish a residual of -3 against a
    # true +2, and a negative residual is this tree's signal that a child is being counted inside
    # another. The function would accuse itself of the one defect the residual exists to catch.
    _folded: dict[str, float] = field(default_factory=dict)
    # The same bookkeeping for the additive counters. Maxima need none: re-folding a max is
    # idempotent, so they are deliberately absent from this dict rather than tracked and ignored.
    _folded_counters: dict[str, float] = field(default_factory=dict)

    def mark_supported(self) -> None:
        """Declare every leaf measured, seeding explicit zeros.

        A leaf that is genuinely zero -- no retokenization configured, no environment executor --
        must publish 0.0 rather than go missing, or a consumer cannot tell it apart from a call site
        someone forgot to bracket.

        The ``_seconds_max`` rows are the exception and are NOT seeded. They mean "the longest single
        trajectory", which is undefined on a path that has no per-trajectory scopes -- the batched
        collector issues one engine request for a whole batch. Seeded, they would publish 0.0 beside
        a non-zero mean; folded from a batch-wide scope they would publish a SUM under a max name.
        Absent, they say the only true thing.

        ``rollout_trajectory_count`` is excluded by the identical argument, and it is worse than the
        maxima because it is a DIVISOR: docs/telemetry.md tells the consumer to compute
        ``sum / rollout_trajectory_count`` for a per-trajectory mean. Seeded on the batched path it
        published a hard 0.0 next to non-zero wait sums, so that mean is a division by zero or an
        ``inf`` on a dashboard. How many trajectory scopes closed is undefined where none are opened.
        """
        self.supported = True
        for name in GENERATE_LEAF_SPANS:
            self.durations.setdefault(name, 0.0)
        for name in ROLLOUT_COUNTERS:
            if not name.endswith(_MAX_SUFFIX) and name != ROLLOUT_TRAJECTORY_COUNT:
                self.counters.setdefault(name, 0.0)


# Bound per TrajectoryRunner.run call rather than stashed on the runner, because run() is genuinely
# reentrant: the fully-async trainer keeps up to 768 background run() calls in flight and awaits
# eval(), which calls run() on the same instance. asyncio copies the context at task creation, so
# the coroutines a run() spawns see that call's accumulator and separate calls are isolated by
# construction -- no flag, no assertion, and nothing to get wrong in a third trainer.
ROLLOUT_TIMINGS: ContextVar[RolloutTimings | None] = ContextVar("rollout_timings", default=None)

# One trajectory's own cumulative waits, folded into the run's maxima when the trajectory ends.
# Isolated by the same mechanism: every trajectory is launched as a task (utils/progress.py gather
# -> ensure_future; asyncio.gather wraps coroutines the same way), and a task copies the context.
ROLLOUT_TRAJECTORY_WAITS: ContextVar[dict[str, float] | None] = ContextVar("rollout_trajectory_waits", default=None)


@contextlib.contextmanager
def rollout_timings_scope(timings: RolloutTimings | None) -> Iterator[None]:
    """Bind one accumulator for the extent of a ``TrajectoryRunner.run`` call.

    ``None`` binds nothing and makes every region below a no-op, which is how a nested run() -- the
    harbor dispatcher's sub-runner, or the fully-async trainer's concurrent calls -- is kept out of
    an enclosing call's totals rather than doubling them.

    ⚠️ The scope is the awaiting task's, not the fan-out's. A failed fan-out does not cancel its
    siblings, so a sibling task can outlive this ``with`` still holding the copied context and still
    writing into ``timings``. That is harmless -- the accumulator outlives the scope and the late
    write lands in a dict nobody reads again -- but it means the reset below bounds THIS task, not
    every task that inherited the binding.
    """
    token = ROLLOUT_TIMINGS.set(timings)
    try:
        yield
    finally:
        ROLLOUT_TIMINGS.reset(token)


@contextlib.contextmanager
def rollout_trajectory() -> Iterator[None]:
    """Scope one trajectory so its waits can be reduced with max, not only summed.

    The batched collector deliberately opens NO scope: it issues one engine request for a whole batch
    and loops the environment over every row, so a scope there would accumulate the batch into one
    "trajectory" and publish that sum under a ``_seconds_max`` name. Its tail rows are absent
    instead, which is the only true thing to say about a path with no trajectories.
    """
    timings = ROLLOUT_TIMINGS.get()
    if timings is None:
        yield
        return
    waits: dict[str, float] = {}
    token = ROLLOUT_TRAJECTORY_WAITS.set(waits)
    try:
        yield
    finally:
        ROLLOUT_TRAJECTORY_WAITS.reset(token)
        timings.counters[ROLLOUT_TRAJECTORY_COUNT] = timings.counters.get(ROLLOUT_TRAJECTORY_COUNT, 0.0) + 1.0
        for name in ROLLOUT_TAILED_WAITS:
            key = f"{name}{_MAX_SUFFIX}"
            # `waits.get(name, 0.0)`, not `waits.items()`: a trajectory that never waited still
            # closed, and its zero is a measurement.
            timings.counters[key] = max(timings.counters.get(key, 0.0), waits.get(name, 0.0))


def traced_trajectory(fn):
    """Scope one agent_loop coroutine as a trajectory.

    A decorator rather than an inline ``with`` so the two runners that own an agent_loop opt in the
    same way and a third cannot half-adopt it: the scope is the whole coroutine, by construction.
    """

    @functools.wraps(fn)
    async def _traced(*args, **kwargs):
        with rollout_trajectory():
            return await fn(*args, **kwargs)

    return _traced


@contextlib.contextmanager
def rollout_span(name: str) -> Iterator[None]:
    """Accumulate one disjoint generate-tree region.

    Only for regions that hold the event-loop thread throughout, or that the loop runs one at a time.
    A region containing an ``await`` that yields to concurrent siblings overlaps them, and its sum is
    not a partition of anything -- use :func:`rollout_wait` for those.
    """
    # 🚨 Validate the NAME, not just the constants. The import-time guard below checks that every
    # entry of GENERATE_LEAF_SPANS is registered; it cannot see what a call site actually passes, and
    # this function took any string. A typo therefore published a W&B series nobody expects, wrote
    # nothing to finelog, and moved its region silently into generate_span_residual -- so the tree
    # reported "generate is largely unaccounted for", which is the one claim the design says must
    # never be published. A mutation of "rollout_collect" to "rollout_collct" passed 407 CPU tests.
    #
    # Raising is right here, unlike in the publish epilogue where the policy is drop-and-warn: a
    # region name is a literal, so this is a programming error that is wrong on the FIRST rollout,
    # before a step has been paid for -- not a data-dependent condition discovered after the work.
    if name not in GENERATE_LEAF_SPANS:
        raise AssertionError(
            f"{name!r} is not a registered generate span; it would be dropped by "
            f"phase_timing_observations and absorbed by the residual. Expected one of {GENERATE_LEAF_SPANS}"
        )
    timings = ROLLOUT_TIMINGS.get()
    if timings is None:
        yield
        return
    # Not utils.utils.Timer: importing it here closes a cycle (utils.utils ->
    # trajectory_reward_shaping -> trajectory_runners -> base -> timing_observability), and Timer
    # logs two loguru records per region unless log_events=False -- tens of thousands of synchronous
    # log records per step, on the event-loop thread, inside the phase being measured.
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.durations[name] = timings.durations.get(name, 0.0) + (time.perf_counter() - started)


def _record_wait(timings: RolloutTimings, name: str, elapsed: float) -> None:
    """Fold one concurrent await into its run-level triple and its trajectory's total."""
    timings.counters[f"{name}{_SUM_SUFFIX}"] = timings.counters.get(f"{name}{_SUM_SUFFIX}", 0.0) + elapsed
    timings.counters[f"{name}{_COUNT_SUFFIX}"] = timings.counters.get(f"{name}{_COUNT_SUFFIX}", 0.0) + 1.0
    waits = ROLLOUT_TRAJECTORY_WAITS.get()
    if waits is not None:
        waits[name] = waits.get(name, 0.0) + elapsed


@contextlib.contextmanager
def rollout_wait(name: str) -> Iterator[None]:
    """Accumulate one concurrent await into its counter triple.

    Enter and exit run on the event-loop thread, which is what makes this usable around an await
    that hands work to a ThreadPoolExecutor: a ContextVar does not cross ``run_in_executor``, so
    reading the accumulator inside the executor thread would find nothing.
    """
    # The same validation rollout_span does, for the same reason. A wrong name here is NOT dropped
    # in silence by phase_timing_observations -- it reaches publish_driver_counters, which drops the
    # row with a warning. The engine-await series then simply vanishes and the run reads as an
    # environment that never waited. Statically decidable, so fail on the first rollout.
    if name not in ROLLOUT_WAIT_NAMES:
        raise AssertionError(f"{name!r} is not a registered rollout wait; expected one of {ROLLOUT_WAIT_NAMES}")
    timings = ROLLOUT_TIMINGS.get()
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        _record_wait(timings, name, time.perf_counter() - started)


async def timed_env_call(executor, func, /, *args, **kwargs):
    """Run one environment call, splitting the caller-observed wait into its three real terms.

    The stamps are taken on the pool thread, where the work actually starts and ends, and read back
    on the loop thread -- the only place the accumulator is reachable, because a ContextVar does not
    cross ``run_in_executor``.
    """
    timings = ROLLOUT_TIMINGS.get()
    if executor is None:
        # `await` notwithstanding, this is a synchronous call on the loop thread. There is no queue
        # and no resume delay, so the whole wait is execution -- which is also why the caller must
        # not read a large rollout_env_await here as executor contention.
        if timings is None:
            return func(*args, **kwargs)
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _record_env_wait(timings, queued=0.0, executed=time.perf_counter() - started, resumed=0.0)

    loop = asyncio.get_running_loop()
    # A closure rather than run_in_executor(executor, func, *args): the stamps have to be taken
    # inside the pool thread, on both sides of func. It also means kwargs survive, which the bare
    # run_in_executor form silently could not accept.
    if timings is None:
        return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))

    stamps: list[float] = []  # [picked_up, finished], both written on the pool thread

    def _stamped():
        stamps.append(time.perf_counter())
        try:
            return func(*args, **kwargs)
        finally:
            stamps.append(time.perf_counter())

    submitted = time.perf_counter()
    try:
        return await loop.run_in_executor(executor, _stamped)
    finally:
        resumed = time.perf_counter()
        # Read the list ONCE. The pool thread can append between a len() and an index, and a stamp
        # that arrives in that window would make resumed - stamps[1] negative.
        marks = tuple(stamps)
        if len(marks) == 2:
            _record_env_wait(
                timings,
                queued=marks[0] - submitted,
                executed=marks[1] - marks[0],
                resumed=max(0.0, resumed - marks[1]),
            )
        # Otherwise the call never completed: the pool rejected it, the executor shut down, or the
        # await was cancelled while func was still running. Record nothing. Attributing the wait to
        # queueing would report a cancellation as executor undersizing, and counting it would
        # inflate the denominator of every derived mean with a call that never happened.


def _record_env_wait(timings: RolloutTimings, *, queued: float, executed: float, resumed: float) -> None:
    counters = timings.counters
    for name, seconds in (
        (ROLLOUT_ENV_QUEUE, queued),
        (ROLLOUT_ENV_EXEC, executed),
        (ROLLOUT_ENV_RESUME, resumed),
    ):
        counters[f"{name}{_SUM_SUFFIX}"] = counters.get(f"{name}{_SUM_SUFFIX}", 0.0) + seconds
    _record_wait(timings, ROLLOUT_ENV_AWAIT, queued + executed + resumed)


def record_generate_spans(
    timings: RolloutTimings,
    generate_seconds: float,
    all_timings: MutableMapping[str, float],
    counters: MutableMapping[str, float],
) -> None:
    """Fold one generate call into the step's timings and counters.

    Accumulates rather than assigns, because a step can generate more than once: group admission
    and dynamic sampling both resample without closing the step, and Timer accumulates the same way.
    Maxima fold with max for the same reason -- adding two tails would invent a third.

    The residual is signed on purpose. It is the audit: a negative one means a child is being
    counted inside another, and clamping would hide the only automatic detector of that.
    """
    if not timings.supported:
        # An uninstrumented runner publishes nothing at all, not a residual equal to its parent.
        return
    # Fold the DELTA since the last fold, not the running total -- see RolloutTimings._folded.
    covered = 0.0
    for name, seconds in timings.durations.items():
        new_seconds = seconds - timings._folded.get(name, 0.0)
        timings._folded[name] = seconds
        all_timings[name] = all_timings.get(name, 0.0) + new_seconds
        if name in GENERATE_SPANS:
            covered += new_seconds
    all_timings["generate_span_residual"] = all_timings.get("generate_span_residual", 0.0) + (
        generate_seconds - covered
    )
    for name, value in timings.counters.items():
        if name.endswith(_MAX_SUFFIX):
            # A tail folds with max on both axes: the accumulator already holds the largest single
            # trajectory, and re-folding it is idempotent. Adding two tails would invent a third.
            counters[name] = max(counters.get(name, 0.0), value)
        else:
            # Sums and counts accumulate INSIDE the accumulator too, so fold the delta for the same
            # reason durations do -- folding one accumulator twice with counts 8 then 16 published
            # 24. The first version of this fix handled durations and left this loop cumulative.
            counters[name] = counters.get(name, 0.0) + (value - timings._folded_counters.get(name, 0.0))
            timings._folded_counters[name] = value


_driver_counter_check_done = False


def publish_driver_counters(counters: Mapping[str, float], *, step: int) -> None:
    """Publish the rollout's concurrent-wait sums, counts and tails under the trainer role.

    Deliberately not publish_worker_counters: that hardcodes role="worker" and requires a rank, and
    these rows come from the driver, which has neither. They carry no phase, parent or clock_domain
    either -- attributes that would invite a consumer to band a 1e5-second sum into a 98 s parent.
    """
    if not counters:
        return
    global _driver_counter_check_done
    if not _driver_counter_check_done:
        _driver_counter_check_done = True
        # R10, on the driver. Same failure as on the worker and just as invisible: record() is a
        # no-op, flush() still returns True, and the run finishes having published nothing.
        reason = unconfigured_telemetry_reason()
        if reason is not None:
            logger.warning("generate span tree will publish nothing: %s", reason)
    # Settle whatever is already in flight BEFORE opening the window. The trainer publishes the
    # step's phase rows one call earlier, so without this their rejections land inside our window and
    # get reported as "the engine and environment tails are understated" -- telling an operator the
    # tails are bad when the tails may be complete, and pointing them at the wrong producer.
    telemetry.flush(TELEMETRY_FLUSH_TIMEOUT_SECONDS)
    before = telemetry.runtime_status()
    for name, value in counters.items():
        if name not in ROLLOUT_COUNTERS:
            # Drop, do not raise. This runs in the trainer's step epilogue, AFTER the step's work
            # is paid for -- raising here converts a telemetry-naming mistake into a killed
            # training run. The condition is statically decidable and is asserted at import below.
            logger.warning("%r has no rollout counter instrument; dropping the row", name)
            continue
        instrument = rollout_count if name.endswith(_COUNT_SUFFIX) else rollout_wait_seconds
        instrument.record(float(value), attributes={"counter": name, "role": TRAINER_ROLE, "step": str(step)})
    # Same accounting publish_worker_spans does, and it matters more here: a dropped row understates
    # a MAX, and an understated tail is the one number this tree exists to report.
    #
    # FLUSH FIRST, exactly as the worker path does. record() only enqueues; a rejection can land
    # after an immediate runtime_status() and then be absorbed into the NEXT call's baseline, so the
    # step that actually lost the row reports zero and the warning never fires for it. Sampling
    # without flushing makes a lossy publish and a clean one look identical -- the failure this whole
    # file exists to make impossible.
    settled = telemetry.flush(TELEMETRY_FLUSH_TIMEOUT_SECONDS)
    dropped = telemetry.runtime_status().lost_records - before.lost_records
    if dropped > 0:
        logger.warning(
            "rollout counters lost %d record(s) at step %d; the engine and environment tails are "
            "understated and must not be quoted",
            dropped,
            step,
        )
    elif not settled:
        logger.warning(
            "rollout counter flush did not settle within %.1fs at step %d; rows may still be in flight",
            TELEMETRY_FLUSH_TIMEOUT_SECONDS,
            step,
        )


# --- Worker-side spans inside policy_train ------------------------------------------------------
#
# policy_train is a leaf on the driver: it dispatches ppo_train over Ray and waits for the slowest
# policy worker. Nothing inside it is measured, and on a 67B-A2B MoE run it is ~90% of the step.
# These spans decompose it from inside the worker.
#
# They do NOT travel back through the driver. trainer.py keeps only policy_statuses[0]'s
# "train_status", so a sibling key would be transported and then dropped, and rank 0 is the wrong
# rank anyway -- the driver waits for the slowest. Ray actors inherit the telemetry endpoint, run id
# and execution uid from the task runtime, so each worker publishes its own rows directly and the
# aggregation happens at query time over the rank attribute.
#
# ⚠️ Aggregate by PICKING A RANK, not by taking a per-phase max. The entry barrier and the compute
# are ANTI-CORRELATED across ranks -- the last rank to arrive waits ~0 in the barrier and then does
# full compute, while early ranks do the reverse -- so max_r(barrier) + max_r(training_step) comes
# from different ranks, can exceed max_r(policy_ppo_train), and attributes both the skew and the
# compute to their respective worst ranks. The correct read is
#
#     r* = argmax_r(policy_ppo_train);  then report THAT rank's row set.
#
# which is the rank the driver actually waited for. Use p95 across ranks only to describe spread.
#
# These do NOT close against the driver's policy_train, and the difference
#
#     policy_train - max_over_ranks(policy_ppo_train)
#
# does NOT measure driver-side overhead. Do not publish it as such. It is a lower bound on total
# UNMEASURED critical-path time, and that total is dominated by things which are not the driver:
# the worker epilogue after the span closes, the publish and flush below, and the fact that the
# driver waits for the LAST-RETURNING rank while this subtracts the LONGEST-RUNNING one, which need
# not be the same rank. A large value therefore does not demonstrate driver overhead, and a small
# one does not rule it out. It is not a useful quantity in either direction.
#
# Isolating driver-side overhead needs something this instrument does not have: a per-rank entry
# timestamp the driver can difference against its own dispatch timestamp, on a shared clock. That is
# worth building if dispatch is ever suspected; it is not built here, and nothing here substitutes
# for it.
#
# policy_ppo_train does start at true function entry, before the R3 co-arrival drain, so the arrival
# spread the driver pays for lands in policy_entry_barrier rather than vanishing.

logger = logging.getLogger(__name__)

# Deliberately well under Rigging's 5 s default: this blocks the worker's return to the driver, so
# every second here lands in driver policy_train. It is carried forward as policy_span_publish so it
# is at least attributable, but the cheapest version of that is a short timeout.
TELEMETRY_FLUSH_TIMEOUT_SECONDS = 1.0

POLICY_TRAIN_SPANS = (
    "policy_entry_barrier",
    "policy_forward",
    "policy_backward",
    "policy_optimizer_step",
    "policy_entropy_allreduce",
    "policy_metric_allreduce",
    "policy_final_barrier",
)


policy_step_counter = telemetry.histogram("policy_train_count", unit="1")
# Bytes do not belong on a unit-1 histogram. Same suffix dispatch the driver counters use.
policy_step_bytes = telemetry.histogram("policy_train_bytes", unit="By")


def publish_worker_counters(counters: Mapping[str, float], *, step: int, rank: int) -> None:
    """Publish per-step counts and token accounting.

    Separate from the spans on purpose. These are not durations and must never be summed into
    policy_ppo_train or subtracted from the residual, so they go to their own instrument rather than
    riding phase_duration with a units mismatch nobody would notice downstream.

    micro_step_count is the H3 multiplier: at micro_train_batch_size_per_gpu=1 the FSDP all-gather
    count is linear in it, so `policy_ppo_train / micro_step_count` is the number that says whether a
    micro-batch change bought anything. The token counts are the H7 keystone: packing is rejected for
    Grug, so batches are BSHD-padded and eager attention is quadratic on the PADDED shape -- which is
    why a linear padded-token fraction understates the cost and `attention_work_ratio` is carried
    alongside it.
    """
    if not counters:
        return
    for name, value in counters.items():
        instrument = policy_step_bytes if name.endswith("_bytes") else policy_step_counter
        instrument.record(
            float(value),
            attributes={"counter": name, "role": WORKER_ROLE, "rank": str(rank), "step": str(step)},
        )


def unconfigured_telemetry_reason() -> str | None:
    """Why spans would publish nothing, or None if they will publish.

    Used by both the worker spans and the driver's rollout counters. Checked once per process,
    because the failure is otherwise invisible: on an unconfigured
    runtime ``record`` is discarded, ``flush`` returns **True**, and the loss counters stay at zero.
    Every signal reads healthy and the run produces no rows at all -- a full step spent to learn
    nothing. Verified empirically, not assumed.
    """
    status = telemetry.runtime_status()
    if getattr(status, "configured", False):
        return None
    return (
        "rigging telemetry is not configured in this process, so every span and counter will be "
        "discarded silently: record() is a no-op, flush() still returns True, and lost_records stays "
        "at 0. Check that the telemetry endpoint, run id and execution uid reached this Ray actor "
        "(cloud/iris/telemetry_env.py scopes SKYRL_EXECUTION_UID to TASK_RUNTIME and DRIVER, not "
        "RAY_WORKER, so inheritance is what carries it)."
    )


class WorkerSpanAccumulator:
    """Accumulate exclusive wall time per span across a worker's micro-steps.

    Disabled by default and a no-op when disabled, so the on/off pair is a config flip rather than
    two revisions.

    ``synchronize`` decides what the numbers mean, and it is not a free choice. CUDA kernels are
    launched asynchronously, so without a device synchronise these spans measure *launch* time and
    attribute a backward's real cost to whatever later call happens to block. With it they measure
    execution, at the cost of serialising the pipeline the run is otherwise trying to overlap.
    Enable it to attribute, disable it to measure end-to-end -- and never compare the two.
    """

    def __init__(self, *, enabled: bool = False, synchronize: bool = True) -> None:
        self.enabled = enabled
        self.synchronize = synchronize
        self._totals: dict[str, float] = {}

    @contextlib.contextmanager
    def span(self, name: str, *, presync: bool = True) -> Iterator[None]:
        """Time one span.

        ``presync=False`` for a region that performs its own device synchronise: the leading sync
        would drain the queue first and leave the wrapped one measuring nothing, pushing the real
        drain cost into the residual and making the region look free.
        """
        if not self.enabled:
            yield
            return
        if presync:
            self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._totals[name] = self._totals.get(name, 0.0) + (time.perf_counter() - started)

    def record_zero(self, name: str) -> None:
        """Record an explicit zero for a span whose region did not execute.

        A missing row and a zero row are not the same claim: the first says nothing was measured,
        the second says the region cost nothing. Conditional regions must say which.
        """
        if self.enabled:
            self._totals.setdefault(name, 0.0)

    def _sync(self) -> None:
        if not self.synchronize:
            return
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def totals(self, *, total_seconds: float | None = None) -> dict[str, float]:
        """Return the accumulated spans, plus the residual when the enclosing wall is known.

        The residual is what makes the decomposition auditable: a large one means the spans are
        missing where the time goes, which is the failure this instrumentation exists to detect.
        """
        if not self.enabled:
            return {}
        totals = dict(self._totals)
        if total_seconds is not None:
            totals["policy_ppo_train"] = total_seconds
            covered = sum(totals.get(name, 0.0) for name in POLICY_TRAIN_SPANS)
            # Signed on purpose. Clamping at zero hides over-coverage, which is exactly the
            # double-counting an auditable residual is for.
            totals["policy_span_residual"] = total_seconds - covered
        return totals


class StepMemoryProbe:
    """Allocator counters scoped to ONE policy_train call.

    ``torch.cuda.max_memory_allocated`` is documented as "peak allocated memory since the beginning
    of this program", and ``num_alloc_retries`` / ``num_ooms`` are cumulative for the process.
    Published per step without a baseline they read as per-step values and are not: after whichever
    step sets the high-water mark, every later step republishes it unchanged and the series looks
    flat and healthy while measuring nothing, and summing retries counts old events again. A number
    that looks valid and is wrong is worse than a missing one.

    :meth:`begin_step` takes the baseline and is the ONLY place the peak is reset. Resetting it
    later in the step would clear the forward that actually sets it.

    ⚠️ **This does not decompose the peak, and deliberately so.** A per-window probe was built here
    and removed: it was meant to decide whether resident optimizer state could grow -- an fp32
    ``exp_avg_sq`` is +2 bytes per parameter per shard -- by checking whether the optimizer window
    was where the peak occurred. That reasoning is wrong. ``backload_to_gpu`` runs immediately
    before ``ppo_train`` with ``backload_optimizer=True``, and the offload is after training, so the
    optimizer state is GPU-resident throughout forward and backward and the peak grows by that
    amount wherever it falls. The question is plain headroom against the per-step peak below, and it
    needs no extra instrument.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled
        self._baseline_retries = 0.0
        self._baseline_ooms = 0.0

    @staticmethod
    def _cuda():
        # Local, matching WorkerSpanAccumulator.span in this file: torch is heavy and this module is
        # imported by the driver, which has no CUDA context.
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda
        return None

    def begin_step(self) -> None:
        """Rebase the peak and the cumulative counters. Call once, at the top of the step."""
        if not self.enabled:
            return
        cuda = self._cuda()
        if cuda is None:
            return
        cuda.reset_peak_memory_stats()
        stats = cuda.memory_stats()
        self._baseline_retries = float(stats.get("num_alloc_retries", 0))
        self._baseline_ooms = float(stats.get("num_ooms", 0))

    def counters(self) -> dict[str, float]:
        """Per-step allocator counters, or nothing if disabled or off-device."""
        if not self.enabled:
            return {}
        cuda = self._cuda()
        if cuda is None:
            return {}
        stats = cuda.memory_stats()
        return {
            # Peaks since begin_step, not since process start. H2's keystone and the gap three OOMs
            # went through: every crash in this workstream asked for the eager attention score
            # tensor (num_heads * L^2 * 4 B ~= 3.4-3.6 GiB) and we had no memory series to see it
            # coming. Peak is also what binds micro_train_batch_size_per_gpu to 1.
            "peak_allocated_bytes": float(cuda.max_memory_allocated()),
            "peak_reserved_bytes": float(cuda.max_memory_reserved()),
            # Deltas since begin_step. Fragmentation, not capacity, is what killed step 2 before
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. A retry means the allocator had the
            # bytes but not a contiguous block; it is the leading indicator of the OOM, and it is
            # invisible in a peak-bytes series alone.
            "alloc_retries": float(stats.get("num_alloc_retries", 0)) - self._baseline_retries,
            "alloc_ooms": float(stats.get("num_ooms", 0)) - self._baseline_ooms,
        }


class WorkerTimingSink:
    """Publish worker spans under their own role and clock domain.

    ``clock_domain`` is deliberately not ``inclusive_wall``: these are exclusive durations, and
    mixing the two under one domain would let a consumer sum a child into its parent twice.

    It also carries the SYNCHRONIZE mode, because that decides what the numbers mean. CUDA kernels
    launch asynchronously, so without a device synchronise a span measures *launch* time and charges
    a backward's real cost to whatever later call happens to block; with it, the span measures
    execution and the pipeline is serialised. The accumulator's own docstring says never to compare
    the two -- and a consumer cannot obey that if both ship under the same label. `_launch` rows are
    end-to-end-honest and attribution-poor; `_wall` rows are the reverse.
    """

    def __init__(self, rank: int, *, synchronize: bool = True) -> None:
        self.rank = rank
        self.synchronize = synchronize

    def _clock_domain(self, name: str) -> str:
        containment = "inclusive" if name in ("policy_ppo_train", "policy_training_step") else "exclusive"
        return f"{containment}_{'wall' if self.synchronize else 'launch'}"

    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None:
        for observation in observations:
            phase_duration.record(
                observation.duration_seconds,
                attributes={
                    "phase": observation.name,
                    "root": observation.root,
                    "parent": TIMING_PARENTS.get(observation.name) or "",
                    "clock_domain": self._clock_domain(observation.name),
                    "role": WORKER_ROLE,
                    "rank": str(self.rank),
                    "step": str(step),
                },
            )


def publish_worker_spans(
    timings: Mapping[str, float],
    *,
    step: int,
    rank: int,
    previous_publish: tuple[int, float] | None = None,
    counters: Mapping[str, float] | None = None,
    synchronize: bool = True,
) -> float:
    """Publish one worker's policy_train decomposition, settle the queue, and report what it cost.

    Returns the wall seconds spent publishing and flushing, so the caller can fold it into the NEXT
    step's spans as ``policy_span_publish``. Without that it is unmeasured time on the critical path
    and nothing downstream can subtract it.

    Loss is detected from Rigging's own counters rather than from ``flush``'s return value.
    ``flush`` reports only whether the queue *settled*: it returns ``True`` once rejected and dropped
    records have settled too, so a ``True`` return does **not** mean every rank's rows arrived.
    Treating it as if it did would let a short row set -- which understates max and p95 -- pass as a
    clean measurement.
    """
    if not timings and not counters:
        return 0.0
    started = time.perf_counter()
    before = telemetry.runtime_status()
    sink = WorkerTimingSink(rank, synchronize=synchronize)
    sink.publish(phase_timing_observations(timings), step)
    # Inside the same before/after window as the spans, so the loss check below covers them too.
    # Published separately they were invisible: emission exceptions are swallowed by Rigging and the
    # counters went out before any baseline was taken, so a dropped counter row looked like a phase
    # that simply was not measured.
    if counters:
        publish_worker_counters(counters, step=step, rank=rank)
    if previous_publish is not None:
        # Emitted under the step it actually belongs to, and deliberately NOT folded into that step's
        # totals: it happened after that interval closed, so subtracting it from that step's residual
        # would remove time the interval never contained. The last step's publish is never emitted --
        # there is no later call to carry it -- and that is the honest cost of measuring it at all.
        previous_step, seconds = previous_publish
        sink.publish(phase_timing_observations({"policy_span_publish": seconds}), previous_step)
    settled = telemetry.flush(TELEMETRY_FLUSH_TIMEOUT_SECONDS)
    after = telemetry.runtime_status()

    # lost_records already includes rejected ones; adding both deltas reported 2N for N losses.
    dropped = after.lost_records - before.lost_records
    if dropped > 0:
        logger.warning(
            "policy_train spans lost %d record(s) at step %d rank %d; max and p95 over ranks are "
            "understated and must not be quoted",
            dropped,
            step,
            rank,
        )
    elif not settled:
        logger.warning(
            "policy_train span flush did not settle within %.1fs at step %d rank %d; rows may still be in flight",
            TELEMETRY_FLUSH_TIMEOUT_SECONDS,
            step,
            rank,
        )
    return time.perf_counter() - started


# Statically decidable, so check it once at import rather than in the step epilogue (see
# publish_driver_counters). A mistake here fails the process at start, not after a paid-for step.
# publish_driver_counters dispatches on the name suffix, so every declared counter must carry one
# of the two recognised suffixes or it would silently land on the wrong instrument (a seconds value
# on a unit-1 histogram, which nothing downstream would notice).
for _name in ROLLOUT_COUNTERS:
    if not _name.endswith((_SUM_SUFFIX, _MAX_SUFFIX, _COUNT_SUFFIX)):
        raise AssertionError(
            f"ROLLOUT_COUNTERS names {_name!r}, which ends in none of {_SUM_SUFFIX!r}, {_MAX_SUFFIX!r} "
            f"or {_COUNT_SUFFIX!r}; publish_driver_counters would route it to the wrong instrument"
        )

# The same guard for the span half, which had only a test. An unregistered span name is dropped by
# phase_timing_observations -- but trainer.py splats all_timings into W&B unfiltered, so a typo
# publishes a W&B series nobody expects, publishes nothing to finelog, and (being outside
# GENERATE_SPANS) is absorbed by the residual with no row explaining it. Every signal reads healthy.
for _name in (*GENERATE_LEAF_SPANS, "generate_span_residual"):
    if _name not in TIMING_PARENTS:
        raise AssertionError(f"{_name!r} is not in TIMING_PARENTS; phase_timing_observations would drop it")

# A _seconds_max row is folded with max across generate calls and a _seconds_sum row with addition,
# so a name that reads as one and is treated as the other is a silent arithmetic error rather than a
# missing row. record_generate_spans dispatches on the same suffix; keep the two in step.
for _name in ROLLOUT_COUNTERS:
    if _name.endswith(_MAX_SUFFIX) and f"{_name[: -len(_MAX_SUFFIX)]}{_SUM_SUFFIX}" not in ROLLOUT_COUNTERS:
        raise AssertionError(f"{_name!r} has no matching {_SUM_SUFFIX!r} row, so no mean is derivable beside its tail")

del _name
