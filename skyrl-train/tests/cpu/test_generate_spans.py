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
    ROLLOUT_TRAJECTORY_COUNT,
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
# ⚠️ Keys are module paths under `skyrl_train`, NOT bare names under `trajectory_runners`.
#
# The earlier form hardcoded the `trajectory_runners.` prefix, so the guard could not see a
# tokenizer call anywhere else -- and there was one: InferenceEngineClient.generate templates on the
# DRIVER thread whenever a caller passes `prompts=`, inside the rollout_wait that is supposed to be
# engine time. Inserting `self.tokenizer.encode("leak")` there left the whole suite green.
#
# This is the same scope error that was fixed for the all_reduce(status) walk one commit earlier and
# left standing here. Every one of these lists is a hypothesis about where the defect can live, so
# adding a module is cheap and omitting one is how the guard goes quiet.
EXPECTED_TOKENIZE_REGIONS = {
    "trajectory_runners.skyrl_gym": 7,
    "trajectory_runners.step_wise": 3,
    "inference_engines.inference_engine_client": 2,
    "inference_engines.remote_inference_engine": 2,
    "inference_engines.sglang.sglang_engine": 1,
}

# The packages the rollout actually runs through. Scanning DIRECTORIES rather than enumerating
# modules is what makes the too-narrow version unrepresentable: dropping a module from the dict above
# used to just delete a parametrized case, and a test that does not run is indistinguishable from one
# that passes.
TOKENIZE_SCANNED_PACKAGES = ("trajectory_runners", "inference_engines")

# Rollout-path modules that tokenize and are deliberately NOT covered. Each needs a reason, because
# the whole point of deriving the list is that adding one is a decision somebody writes down.
TOKENIZE_UNCOVERED_MODULES = {
    # ⚠️ These three were INVISIBLE to the previous guard, which matched the literal text
    # `"self.tokenizer."`. They reach a tokenizer through a parameter or a local, and the
    # receiver-resolving walk found them the moment it replaced the spelling match. Nine call sites
    # between them.
    #
    # ✅ VERIFIED ATTRIBUTED, not lost: `_decode` is reached from `build_trajectory_records` ->
    # `retain_trajectories`, which `base.py:116` calls inside `rollout_span("rollout_retain")`. Its
    # cost lands on a declared leaf; it is simply not `rollout_tokenize`. That is a docs problem, not
    # a measurement one, and docs/telemetry.md no longer claims the leaf covers every tokenizer call.
    "trajectory_runners.trajectory_retention": "attributed to rollout_retain (verified)",
    # ⚠️ NOT TRACED. Helper modules whose callers are probably inside bracketed regions, but I did
    # not follow all nine sites, and the last two modules I marked "unaudited" both turned out to
    # carry a real defect. Treat as an open question, not a clean bill.
    "trajectory_runners.trajectory_processing": "helper; call sites NOT traced -- see note",
    "inference_engines.teacher_engine_client": "teacher-scoring path; NOT traced -- see note",
    # Uninstrumented runners. They have not bracketed their call sites at all, and
    # `generate_spans_instrumented` is False for both, so they publish NOTHING rather than a seeded
    # all-zero tree -- absence is the honest signal (see the uninstrumented-runner test below).
    # Bracketing their tokenizer calls alone would be worse than leaving them: a lone leaf under a
    # runner that publishes no parent.
    "trajectory_runners.harbor.runner": "uninstrumented runner; publishes nothing by design",
    # ⚠️ Found only once the walk resolved ATTRIBUTE-bound receivers: it holds `self._tokenizer` and
    # re-encodes every response in the batch on the driver event-loop thread, inside the caller's
    # rollout_wait -- byte-for-byte the defect bracketed in remote_inference_engine.
    #
    # ✅ VERIFIED INERT TODAY, not assumed: `OpenAIHTTPModelClient` is constructed only at
    # `entrypoints/fully_async.py:49`, and the fully-async generate loop deliberately passes NO
    # phase_timings (`fully_async_trainer.py:1126` says why -- 768 concurrent run() calls cannot be
    # summed into one accumulator). With no accumulator in scope every span there is a no-op, so
    # nothing wrong is published. Bracketing it would add a leaf under a runner that publishes no
    # parent, which is worse than leaving it.
    #
    # ⚠️ **This stops being inert the moment the fully-async path gains per-call accumulators.**
    # Whoever does that must bracket this call site in the same change.
    "trajectory_runners.model_clients": "fully-async only; no accumulator in scope (verified inert)",
    "trajectory_runners.mini_swe.runner": "uninstrumented runner; publishes nothing by design",
    # ⚠️ Both alternative backends were audited after the fact and BOTH carried the defect, so they
    # are covered above rather than exempted. remote_inference_engine re-encodes every generated
    # response in the batch on the driver thread (its own comment explains why: vLLM cannot return
    # token IDs over HTTP), and sglang detokenizes there too. Passing `prompt_token_ids=` does not
    # protect against either -- they are on the RESPONSE side.
}

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


# Tokenizer METHODS we care about. Matching on the method name alone would fire on `str.encode`
# and `bytes.decode` everywhere and immediately need an exemption list -- and an exemption list is
# where the next leak hides. So the method name narrows, and the RECEIVER decides.
_TOKENIZER_METHODS = {"encode", "decode", "apply_chat_template", "batch_decode", "convert_ids_to_tokens"}


def _tokenizer_receivers(tree: ast.AST) -> set[str]:
    """Identifiers bound to a tokenizer anywhere in this module.

    ⚠️ Receiver-resolving, not spelling-matching. The old check keyed on the literal text
    `self.tokenizer.`, so a routine local alias -- `tok = self.tokenizer; tok.apply_chat_template(...)`
    -- was invisible to it, and so was every function that takes the tokenizer as a PARAMETER.
    `trajectory_retention._decode(tokenizer, ...)` is the second kind and was not even discovered.
    """
    names = {"tokenizer", "self.tokenizer"}
    for node in ast.walk(tree):
        # `x = self.tokenizer` / `x = tokenizer` / `self._tok = tokenizer`
        if isinstance(node, ast.Assign):
            value = node.value
            bound = (isinstance(value, ast.Attribute) and "tokenizer" in value.attr) or (
                isinstance(value, ast.Name) and value.id in names
            )
            if bound:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        # ⚠️ Attribute targets too. `self._tokenizer = tokenizer` is the ordinary
                        # way to hold one, and a Name-only walk could not see the field it created
                        # -- which is how model_clients stayed undiscovered while re-encoding every
                        # response in the batch on the driver thread.
                        names.add(target.attr)
        # `def f(tokenizer: PreTrainedTokenizerBase)` -- the parameter form
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in [*node.args.args, *node.args.kwonlyargs]:
                if "tokenizer" in arg.arg:
                    names.add(arg.arg)
    return names


def _tokenizer_calls(tree: ast.AST) -> list[ast.Call]:
    """Every call that reaches a tokenizer method through a resolved receiver."""
    receivers = _tokenizer_receivers(tree)
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _TOKENIZER_METHODS:
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in receivers:
            found.append(node)
        elif isinstance(owner, ast.Attribute) and ("tokenizer" in owner.attr or owner.attr in receivers):
            # `"tokenizer" in owner.attr` rather than equality: `self._tokenizer` is the same thing
            # spelled differently. The _TOKENIZER_METHODS narrowing above is what keeps this from
            # firing on `str.encode`, so widening the receiver here is safe.
            found.append(node)
    return found


def _module_source(module_name: str) -> str:
    """Read a module's source from DISK, never by importing it.

    `importlib.import_module` pulled in the real backend package, and `sglang` is not installed in
    the CPU environment -- so covering that module by import turned its guard into a collection
    error. These walks only ever needed the text.
    """
    import pathlib

    import skyrl_train

    path = pathlib.Path(skyrl_train.__file__).parent / (module_name.replace(".", "/") + ".py")
    assert path.exists(), f"{module_name} does not resolve to a file at {path}"
    return path.read_text()


def _tokenize_regions(module_name: str) -> list[str]:
    """The body of every ``with rollout_span("rollout_tokenize")`` block, as source text."""
    lines = _module_source(module_name).splitlines()
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


def test_a_trajectory_that_never_waited_still_publishes_a_zero_tail(clock):
    """🚨 A closed trajectory that never waited is a MEASURED zero, not a gap.

    An agent loop can return before its first engine await — an overlong prompt is rejected up
    front. Folding only the waits that fired leaves that trajectory's tail absent, which reads as
    "not measured" when the true answer is zero, and it silently shrinks the population the max is
    taken over.
    """
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        with rollout_trajectory():  # returns having awaited nothing
            pass
        with rollout_trajectory():
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                clock.advance(2.0)

    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"] == 2.0
    assert timings.counters[f"{ROLLOUT_ENV_AWAIT}_seconds_max"] == 0.0, (
        "neither trajectory touched the environment, and that zero is a measurement"
    )
    assert timings.counters[ROLLOUT_TRAJECTORY_COUNT] == 2.0


def test_the_tail_and_the_mean_describe_the_same_population(clock):
    """`_count` counts timed CALLS; `_seconds_max` is a max per TRAJECTORY.

    Dividing the sum by `_count` gives a mean per call and presenting it beside a per-trajectory max
    compares two different populations. ROLLOUT_TRAJECTORY_COUNT is the denominator that makes them
    comparable: here two trajectories make three engine calls totalling 6 s, so the per-call mean is
    2.0 and the per-trajectory mean is 3.0 against a tail of 4.0.
    """
    timings = RolloutTimings()
    with rollout_timings_scope(timings):
        with rollout_trajectory():
            for _ in range(2):
                with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                    clock.advance(2.0)
        with rollout_trajectory():
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                clock.advance(2.0)

    total = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_sum"]
    calls = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_count"]
    trajectories = timings.counters[ROLLOUT_TRAJECTORY_COUNT]
    tail = timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"]

    assert (total, calls, trajectories, tail) == (6.0, 3.0, 2.0, 4.0)
    assert total / calls == 2.0, "the per-CALL mean, which the tail must not be read against"
    assert total / trajectories == 3.0, "the per-TRAJECTORY mean, which it must"
    assert tail > total / trajectories, "the straggler is above the population the tail is drawn from"


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


def test_a_cancellation_that_completes_its_executor_call_still_records_nothing():
    """🚨 The real race, driven through a real cancellation -- not a clock made to run backwards.

    Counting the stamps was never enough. The pool thread can FINISH -- appending both stamps --
    while the coroutine is resumed with CancelledError, so `len(marks) == 2` held on a call that
    returned a value to nobody, and the split published queue/exec/resume/await and a count of 1 for
    it. An earlier version of this file drove a SUCCESSFUL call with a scripted clock instead, which
    proved a clamp and left the bogus row untouched.

    The pool has one worker, so submitting a second job and blocking on it guarantees `_stamped` ran
    to completion, including the finally that appends the second stamp. The loop thread never yields
    in that window, so the wrapper future is still PENDING and cancel() takes.
    """

    started = threading.Event()

    def _work():
        started.set()
        return "value nobody receives"

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                task = asyncio.ensure_future(timed_env_call(pool, _work))
                # ⚠️ ensure_future only SCHEDULES the coroutine. Without yielding here the task
                # never reaches its await, nothing is submitted to the executor, and cancelling it
                # proves nothing -- an earlier version of this test did exactly that and the
                # revert-the-guard mutation survived it.
                while not started.is_set():
                    await asyncio.sleep(0)
                # One worker, so this queues BEHIND _stamped and returning proves _stamped ran to
                # completion -- including the finally that appends the second stamp. It blocks the
                # loop thread, so the wrapper's result cannot be delivered while we hold it.
                pool.submit(lambda: None).result(timeout=5.0)
                assert task.cancel(), "the wrapper had already resumed; the race was not reproduced"
                with pytest.raises(asyncio.CancelledError):
                    await task
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters == {}, f"a cancelled call recorded {timings.counters}"
    assert timings.durations == {}, f"a cancelled call recorded {timings.durations}"


def test_a_callee_that_raises_still_records_its_split():
    """The other half of the same condition, and the reason it is `not cancelled` rather than a
    bare except. An environment that fails after two seconds of real work queued and executed; its
    time is a fact about the rollout, and dropping it would understate every derived mean."""

    def _boom():
        raise ValueError("the environment rejected the action")

    async def _drive():
        timings = RolloutTimings()
        with rollout_timings_scope(timings):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(ValueError):
                    await timed_env_call(pool, _boom)
        return timings

    timings = asyncio.run(_drive())
    assert timings.counters[f"{ROLLOUT_ENV_AWAIT}_count"] == 1.0, "a failed call is still a call"
    assert timings.counters[f"{ROLLOUT_ENV_EXEC}_seconds_sum"] >= 0.0


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
    assert set(timings.durations.values()) == {0.0}
    # Two exceptions, both undefined on a path with no per-trajectory scopes and so ABSENT rather
    # than seeded to a false zero: the tails ("the longest single trajectory") and the trajectory
    # COUNT, which is the divisor docs/telemetry.md tells the consumer to use for a per-trajectory
    # mean -- seeded, it makes that mean a division by zero.
    seeded = {
        name
        for name in ROLLOUT_COUNTERS
        if not name.endswith("_seconds_max") and name != timing_module.ROLLOUT_TRAJECTORY_COUNT
    }
    assert set(timings.counters) == seeded
    assert set(timings.counters.values()) == {0.0}
    assert not any(name.endswith("_seconds_max") for name in timings.counters)


def test_a_certified_runner_actually_publishes_a_populated_tree():
    """🚨 The positive direction. Without it the whole feature is one deletable line.

    `TrajectoryRunner.run` calls `phase_timings.mark_supported()` on a certified runner, and
    everything downstream is gated on the flag it sets: `record_generate_spans` returns at its first
    line, no leaf and no residual reach `all_timings`, and `publish_driver_counters` has nothing to
    publish. Replacing that one call with `pass` left the entire suite green -- a default-on feature
    publishing NOTHING, with the negative test (below) still passing because it asserts absence.

    So drive a certified runner end to end and assert the tree came out populated.
    """

    class _Bracketed(TrajectoryRunner):
        generate_spans_instrumented = True

        async def _run(self, input_batch, disable_tqdm: bool = False):
            with rollout_span("rollout_collect"):
                with rollout_span("rollout_tokenize"):
                    pass
            with rollout_span("rollout_assemble"):
                pass
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                pass
            return {}

    timings = RolloutTimings()
    asyncio.run(_Bracketed().run({}, phase_timings=timings))
    assert timings.supported is True, "a certified runner must mark the accumulator supported"

    all_timings: dict[str, float] = {}
    counters: dict[str, float] = {}
    record_generate_spans(timings, generate_seconds=10.0, all_timings=all_timings, counters=counters)

    # Every declared leaf reaches the step's timings, and so does the residual that closes them.
    for name in (*GENERATE_SPANS, *GENERATE_NESTED_SPANS):
        assert name in all_timings, f"{name} never reached the published tree"
    assert "generate_span_residual" in all_timings
    covered = sum(all_timings[name] for name in GENERATE_SPANS)
    assert all_timings["generate_span_residual"] == pytest.approx(10.0 - covered)
    assert counters[f"{ROLLOUT_ENGINE_AWAIT}_count"] == 1.0, "the wait counters must publish too"


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
    ADDING two tails would invent a third that no trajectory ever had.

    ⚠️ This builds a FRESH accumulator per call, which is one of the two shapes a caller can use. The
    trainer uses the other -- one step-scoped RolloutTimings folded repeatedly -- and that shape is
    covered by test_a_step_that_generates_twice_reuses_one_accumulator_without_double_counting.
    Passing here said nothing about it, which is how the double-count survived.
    """
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


def test_the_unconfigured_detector_reads_the_runtime_rather_than_always_saying_yes(monkeypatch):
    """🚨 The detector itself, not its callers.

    Both call sites' tests monkeypatch `unconfigured_telemetry_reason` wholesale, so NOTHING
    exercised the function: `if True or getattr(status, "configured", False)` left the whole suite
    green. If the check inverts or the attribute name drifts, the branch's headline safety property
    -- that an unconfigured runtime is announced rather than silently producing no rows while every
    signal reads healthy -- stops holding, and that is a full multi-hour run spent to learn nothing.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(timing_module.telemetry, "runtime_status", lambda: SimpleNamespace(configured=True))
    assert timing_module.unconfigured_telemetry_reason() is None, "a configured runtime has no reason"

    monkeypatch.setattr(timing_module.telemetry, "runtime_status", lambda: SimpleNamespace(configured=False))
    reason = timing_module.unconfigured_telemetry_reason()
    assert reason is not None and "not configured" in reason

    # An old Rigging without the attribute must read as UNCONFIGURED, not as configured: the
    # getattr default is what decides which way an unknown runtime falls, and falling the other way
    # would suppress the warning on exactly the runtimes that need it.
    monkeypatch.setattr(timing_module.telemetry, "runtime_status", lambda: SimpleNamespace())
    assert timing_module.unconfigured_telemetry_reason() is not None


def test_a_step_that_generates_twice_reuses_one_accumulator_without_double_counting():
    """Folding one accumulator twice must add only what is new -- on BOTH dicts.

    ⚠️ The trainer does NOT do this: `phase_timings = RolloutTimings()` is inside the dataloader loop
    and both resample paths `continue`, so production builds a fresh accumulator every time. This is
    a contract test for the documented "accumulates rather than assigns" behaviour, not a regression
    test for a live bug. Saying otherwise -- as an earlier version of this docstring did -- tells the
    next reader the arithmetic is load-bearing where it is defensive.

    Under that shape the earlier fold recomputed `covered` from the running total while
    generate_seconds stayed per-call, so the second fold subtracted the accumulated leaves from a
    single call's wall: two 6 s calls with 5 s of leaves each give a residual of -3 against a true
    +2, and a negative residual is this tree's signal that a child is being counted inside another.
    Counts had the same shape and published 24 for a true 16.
    """
    timings = RolloutTimings()
    timings.mark_supported()
    all_timings: dict[str, float] = {}
    counters: dict[str, float] = {}

    count_key = f"{ROLLOUT_ENGINE_AWAIT}_count"
    tail_key = f"{ROLLOUT_ENGINE_AWAIT}_seconds_max"

    timings.durations["rollout_collect"] = 5.0
    timings.counters[count_key] = 8.0
    timings.counters[tail_key] = 3.0
    record_generate_spans(timings, 6.0, all_timings, counters)
    assert all_timings["rollout_collect"] == pytest.approx(5.0)
    assert all_timings["generate_span_residual"] == pytest.approx(1.0)
    assert counters[count_key] == pytest.approx(8.0)

    # Second generate in the same step. Durations and additive counters BOTH accumulate inside the
    # accumulator, so these are the running totals, not this call's contribution.
    timings.durations["rollout_collect"] = 10.0
    timings.counters[count_key] = 16.0
    timings.counters[tail_key] = 5.0
    record_generate_spans(timings, 6.0, all_timings, counters)
    assert all_timings["rollout_collect"] == pytest.approx(10.0), "the leaf must total both calls"
    assert all_timings["generate_span_residual"] == pytest.approx(2.0), (
        "12 s of generate against 10 s of leaves is +2; the pre-fix arithmetic published -3"
    )
    assert counters[count_key] == pytest.approx(16.0), (
        "counts accumulate in the accumulator too; folding cumulatively published 24"
    )
    assert counters[tail_key] == pytest.approx(5.0), "a tail folds with max and re-folding is idempotent"


def test_the_certificate_follows_the_collector_and_is_not_inherited():
    """Exercises the exact rule the runner applies, on real collector classes.

    The bracketed call sites live in the collector, and the collector is INJECTED -- main_base
    already injects one when step_wise_training is set, and an adopting team is expected to.
    __init_subclass__ guards the class and cannot see that. An uncertified collector inheriting True
    would make mark_supported() seed 0.0 for every leaf and publish residual == generate: a
    measured-zero lie that the explicit seeds make indistinguishable from a real all-zero rollout.
    """
    from skyrl_train.trajectory_runners.skyrl_gym import (
        BatchedTrajectoryCollector,
        WholeTrajectoryCollector,
        collector_is_instrumented,
    )
    from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector

    # All three shipped collectors bracket their call sites and must be certified. StepWise is the
    # one a central allowlist silently revoked -- it lives in another module.
    for collector_type in (WholeTrajectoryCollector, BatchedTrajectoryCollector, StepWiseRolloutCollector):
        assert collector_is_instrumented(collector_type.__new__(collector_type)), collector_type.__name__

    class UnbracketedCollector:
        def __init__(self, runner):
            self._runner = runner

    assert not collector_is_instrumented(UnbracketedCollector(None)), (
        "a collector that brackets nothing must publish nothing, not a seeded zero"
    )

    # A SUBCLASS of a certified collector is NOT certified: it may override agent_loop or
    # collect_batched without the brackets, and those methods are what the certificate is about.
    class SneakySubclass(WholeTrajectoryCollector):
        pass

    assert not collector_is_instrumented(SneakySubclass.__new__(SneakySubclass)), (
        "getattr would inherit the flag here; the rule reads the class's own __dict__ for this reason"
    )


def test_a_subclass_that_replaces_run_is_not_recertified_by_its_collector():
    """The two certificates cover different halves and neither implies the other.

    __init_subclass__ revokes the CLASS certificate from a subclass that replaces _run. The runner
    then assigns an INSTANCE attribute, which shadows it -- so a subclass with its own unbracketed
    _run, inheriting __init__ and getting the default (certified) collector, would be re-certified by
    the very check meant to tighten certification. It would then seed 0.0 for every leaf and publish
    residual == generate.
    """

    class ReplacesRun(SkyRLGymTrajectoryRunner):
        async def _run(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("not invoked")

    assert SkyRLGymTrajectoryRunner.generate_spans_instrumented is True
    assert ReplacesRun.generate_spans_instrumented is False, "__init_subclass__ must revoke it"

    # The expression __init__ evaluates, on a certified collector. The class certificate has to
    # survive into it, or the instance attribute silently re-grants what __init_subclass__ removed.
    from skyrl_train.trajectory_runners.skyrl_gym import WholeTrajectoryCollector, collector_is_instrumented

    certified_collector = WholeTrajectoryCollector.__new__(WholeTrajectoryCollector)
    assert collector_is_instrumented(certified_collector) is True
    assert (ReplacesRun.generate_spans_instrumented and collector_is_instrumented(certified_collector)) is False


def test_an_unsettled_flush_is_reported_even_when_rows_were_also_lost(monkeypatch):
    """The two conditions are independent, and together they are the worst case.

    Under `elif`, a degraded endpoint that rejected two rows and still held forty at the one-second
    cap reported only the two -- and an operator reasonably read the other forty as delivered. The
    branch was also dead to the suite: deleting it entirely left the whole suite green.
    """
    from types import SimpleNamespace

    warned: list[str] = []
    # Key the loss on the FLUSH, not on a call count: the R10 unconfigured check reads
    # runtime_status before the baseline does, so counting calls made the baseline itself see the
    # loss and `dropped` came out zero. A fake that keys on the wrong event tests the wrong thing.
    state = {"flushed": False}

    def _flush(timeout):
        state["flushed"] = True
        return False  # timed out AND, below, rows were lost -- the combination `elif` used to hide

    monkeypatch.setattr(timing_module.telemetry, "flush", _flush)
    monkeypatch.setattr(
        timing_module.telemetry,
        "runtime_status",
        # Cumulative, and non-zero before this publish: the delta is 2, the absolute value is 9.
        lambda: SimpleNamespace(lost_records=9 if state["flushed"] else 7, rejected_records=0),
    )
    monkeypatch.setattr(timing_module.logger, "warning", lambda m, *a, **k: warned.append(str(m) % a if a else str(m)))
    monkeypatch.setattr(timing_module, "rollout_count", type("_H", (), {"record": lambda *a, **k: None})())

    publish_driver_counters({f"{ROLLOUT_ENGINE_AWAIT}_count": 1.0}, step=4)

    assert [w for w in warned if "2 telemetry record(s) were lost" in w], (
        "the DELTA must be reported, not the process-cumulative absolute value"
    )
    assert [w for w in warned if "did not settle" in w], (
        "and so must the rows still in flight; under elif they were silent whenever anything was lost"
    )


def test_a_loss_warning_does_not_claim_more_than_lost_records_can_support(monkeypatch):
    """`lost_records` is process-wide, so this publisher cannot prove the lost rows were its own.

    An earlier version flushed before sampling the baseline to clean the window. That bought
    precision for a second blocking 1 s flush per step on the critical path, by default -- and lost
    the attribution anyway whenever the flush timed out. The warning now says rows were lost AROUND
    this publish and that they may be another producer's, which is what the counter supports.
    """
    from types import SimpleNamespace

    warned: list[str] = []
    state = {"flushes": 0}
    timeouts: list[float] = []

    def _flush(timeout):
        state["flushes"] += 1
        timeouts.append(timeout)
        return True

    monkeypatch.setattr(timing_module.telemetry, "flush", _flush)
    monkeypatch.setattr(
        timing_module.telemetry,
        "runtime_status",
        lambda: SimpleNamespace(lost_records=3 if state["flushes"] else 0, rejected_records=0),
    )
    monkeypatch.setattr(timing_module.logger, "warning", lambda m, *a, **k: warned.append(str(m) % a if a else str(m)))
    monkeypatch.setattr(timing_module, "rollout_count", type("_H", (), {"record": lambda *a, **k: None})())

    publish_driver_counters({f"{ROLLOUT_ENGINE_AWAIT}_count": 1.0}, step=4)

    assert state["flushes"] == 1, "one flush per publish; a second was added once and cost 1 s/step"
    assert timeouts == [timing_module.TELEMETRY_FLUSH_TIMEOUT_SECONDS], (
        "the driver's flush must use the documented cap; raising it to 60 s was green, and this runs "
        "in the step epilogue on the default path"
    )
    # Pin the CONSTANT too, not only that the call uses it. Comparing the call to the same constant
    # passes for any value, including 0.0 -- which turns the flush into a no-op and makes every loss
    # invisible -- and 60.0, which is what the doc says it is not.
    assert timing_module.TELEMETRY_FLUSH_TIMEOUT_SECONDS == 1.0, (
        "docs/telemetry.md quotes a one-second cap and derives the per-step cost from it"
    )
    losses = [w for w in warned if "record(s) were lost" in w]
    assert losses, "a loss inside the window must still be reported"
    assert "may be these counters or another producer" in losses[0], (
        "the warning must not assert the rows were ours; lost_records cannot show that"
    )


def test_the_trajectory_count_is_absent_where_no_trajectory_scope_closes():
    """It is a DIVISOR, so a seeded zero is worse than a missing row.

    docs/telemetry.md tells the consumer to compute sum / rollout_trajectory_count for a
    per-trajectory mean. The batched collector opens no trajectory scope, so seeding published a
    hard 0.0 beside non-zero wait sums -- a division by zero, or an inf on a dashboard. Absent, the
    consumer can tell that the mean is not derivable there.
    """
    timings = timing_module.RolloutTimings()
    timings.mark_supported()
    assert timing_module.ROLLOUT_TRAJECTORY_COUNT not in timings.counters
    # The seeding it must NOT break: the sum/count pair still says "bracketed and never waited".
    assert timings.counters[f"{timing_module.ROLLOUT_ENGINE_AWAIT}_count"] == 0.0
    assert timings.counters[f"{timing_module.ROLLOUT_ENGINE_AWAIT}_seconds_sum"] == 0.0


def test_an_unregistered_name_raises_at_both_entry_points():
    """Behavioural coverage for the guards themselves, not for the call sites they protect.

    The static walk below validates the literals at every call site; it says nothing about whether
    the guard exists. Deleting the raise from BOTH rollout_span and rollout_wait left the entire
    whole suite green -- so the headline fix of one review round and its mirror in the next were
    each unprotected against simply being removed.
    """
    with pytest.raises(AssertionError, match="not a registered generate span"):
        with rollout_span("rollout_collct"):
            pass
    with pytest.raises(AssertionError, match="not a registered rollout wait"):
        with rollout_wait("rollout_engine_awiat"):
            pass
    # And the guards must not fire on the real names, with or without an accumulator bound.
    for name in GENERATE_SPANS:
        with rollout_span(name):
            pass
    for name in timing_module.ROLLOUT_WAIT_NAMES:
        with rollout_wait(name):
            pass


def test_every_accepted_wait_name_has_all_three_rows_declared():
    """rollout_wait emits a sum, a count AND a tail, so accepting a name without all three is a
    guard that permits the failure it exists to prevent.

    The first version of ROLLOUT_WAIT_NAMES was hand-listed and included the three env terms, which
    carry a _seconds_sum only -- they are written by _record_env_wait directly. rollout_wait on one
    of them emitted an undeclared `rollout_env_queue_count` that publish_driver_counters then
    dropped with a warning. Deriving the set from the declared rows makes that unrepresentable.
    """
    # Pin the ANSWER, not the derivation. Re-deriving the same predicate here was a tautology: it
    # could only fail if someone replaced the derivation with a hand-list, which is a much narrower
    # guarantee than "these are the names rollout_wait accepts".
    assert timing_module.ROLLOUT_WAIT_NAMES == (
        timing_module.ROLLOUT_ENGINE_AWAIT,
        timing_module.ROLLOUT_ENV_AWAIT,
    ), "exactly the two waits that carry a full sum/count/tail triple"

    # And the property that answer has to satisfy, checked against the publisher's own registry.
    for name in timing_module.ROLLOUT_WAIT_NAMES:
        for suffix in ("_seconds_sum", "_count", "_seconds_max"):
            assert f"{name}{suffix}" in timing_module.ROLLOUT_COUNTERS, (
                f"rollout_wait({name!r}) would emit {name}{suffix}, which nothing declares"
            )


def test_every_rollout_span_call_site_names_a_registered_span():
    """A typo in a call site is invisible to every other test, and it corrupts the decomposition.

    The import-time guard checks that the CONSTANTS are registered; it cannot see what a call site
    passes. Mutating "rollout_collect" to "rollout_collct" in skyrl_gym.py passed the whole suite --
    while on a real run that leaf vanishes, its ~60 s moves into generate_span_residual, and the tree
    reports that generate is largely unaccounted for.

    An AST walk rather than a substring search: it sees the literal actually passed as the argument,
    so a name in a comment or docstring cannot satisfy it.
    """
    import ast
    import pathlib

    runners = pathlib.Path(timing_module.__file__).parent / "trajectory_runners"
    seen: list[tuple[str, int, str, str]] = []
    for path in sorted(runners.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if fname not in ("rollout_span", "rollout_wait"):
                continue
            # Keyword form too. `rollout_span(name="rollout_collct")` has EMPTY node.args, so an
            # args-only walk skipped it in silence -- a hole in the check that exists to close a hole.
            arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "name"), None)
            assert arg is not None, f"{path.name}:{node.lineno} calls {fname} with no name argument"
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                value = arg.value
            elif isinstance(arg, ast.Name):
                # A module constant is SAFER than a literal -- a typo in the identifier is a
                # NameError at import, before anything runs. Resolve it and check what it holds.
                assert hasattr(timing_module, arg.id), (
                    f"{path.name}:{node.lineno} passes {arg.id!r} to {fname}, which is not a "
                    "timing_observability constant; this check cannot resolve it"
                )
                value = getattr(timing_module, arg.id)
            else:
                raise AssertionError(
                    f"{path.name}:{node.lineno} passes a {type(arg).__name__} to {fname}; the name "
                    "would be unvalidated until it ran"
                )
            seen.append((path.name, node.lineno, value, fname))

    assert seen, "found no rollout_span/rollout_wait call sites at all; the walk is looking in the wrong place"
    assert {fname for *_, fname in seen} == {"rollout_span", "rollout_wait"}, (
        "both name-taking entry points must be covered; a wait name that vanishes is worse than a "
        "span name that lands in the residual"
    )
    registered = {
        "rollout_span": timing_module.GENERATE_LEAF_SPANS,
        "rollout_wait": timing_module.ROLLOUT_WAIT_NAMES,
    }
    for filename, lineno, value, fname in seen:
        assert value in registered[fname], (
            f"{filename}:{lineno} calls {fname}({value!r}), which is not registered; a span name is "
            f"absorbed by the residual and a wait name is dropped outright"
        )


def test_driver_loss_detection_flushes_before_it_samples(monkeypatch):
    """A rejection that lands after the sample is invisible, and looks exactly like a clean publish.

    record() only enqueues. The worker path flushes before reading runtime_status a second time
    (publish_worker_spans); the driver path did not, so an asynchronous rejection was absorbed into
    the NEXT call's baseline and the step that actually lost the row warned about nothing -- while
    the tail it understates is the one number this tree exists to report.

    The double makes the loss visible ONLY after the flush, so the pre-fix ordering cannot pass it.
    """

    from types import SimpleNamespace

    state = {"flushed": False}
    warned: list[str] = []

    def _flush(timeout):
        state["flushed"] = True
        return True

    monkeypatch.setattr(timing_module.telemetry, "flush", _flush)
    monkeypatch.setattr(
        timing_module.telemetry,
        "runtime_status",
        lambda: SimpleNamespace(lost_records=3 if state["flushed"] else 0, rejected_records=0),
    )
    monkeypatch.setattr(timing_module.logger, "warning", lambda m, *a, **k: warned.append(str(m) % a if a else str(m)))
    monkeypatch.setattr(timing_module, "rollout_count", type("_H", (), {"record": lambda *a, **k: None})())

    publish_driver_counters({f"{ROLLOUT_ENGINE_AWAIT}_count": 1.0}, step=4)

    assert [w for w in warned if "3 telemetry record(s) were lost around the step 4" in w], (
        "a loss visible only after the flush must still be reported against its own step"
    )


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


def test_the_tokenize_walk_names_every_rollout_module_that_tokenizes():
    """🚨 The list of modules IS the hypothesis, so derive it rather than trusting it.

    The walk used to hardcode the `trajectory_runners.` prefix, which is how a full-batch
    `apply_chat_template` on the driver thread -- inside the caller's engine wait -- went unseen in
    `inference_engine_client`. Adding that module fixed the instance. This fixes the SHAPE: dropping
    a module from EXPECTED_TOKENIZE_REGIONS merely deleted a parametrized case and left the suite
    green, so the too-narrow version was invisible.

    Every module under the rollout's packages that calls a tokenizer must be either covered or
    explicitly uncovered with a reason.
    """
    import pathlib

    import skyrl_train

    root = pathlib.Path(skyrl_train.__file__).parent
    found: set[str] = set()
    for package in TOKENIZE_SCANNED_PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            # Parsed, not substring-matched: `trajectory_retention` reaches its tokenizer through a
            # PARAMETER, so a `"self.tokenizer."` scan never discovered the file at all.
            if _tokenizer_calls(ast.parse(path.read_text())):
                found.add(str(path.relative_to(root).with_suffix("")).replace("/", "."))

    assert found, "the scan found no tokenizer calls at all -- the call shape changed and this is inert"
    declared = set(EXPECTED_TOKENIZE_REGIONS) | set(TOKENIZE_UNCOVERED_MODULES)
    assert found == declared, (
        f"undeclared modules tokenize on the rollout path: {sorted(found - declared)}; "
        f"declared but no longer tokenizing: {sorted(declared - found)}"
    )


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
    source = _module_source(module_name)
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
    for node in _tokenizer_calls(tree):
        if enclosing.get(node.lineno) in TOKENIZE_EXEMPT_FUNCTIONS:
            continue
        if not any(node.lineno in region for region in regions):
            leaked.append(f"{module_name}:{node.lineno} tokenizer.{node.func.attr}() outside a tokenize region")
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


# Functions whose body is exactly one trajectory. These MUST carry @traced_trajectory.
TRAJECTORY_SCOPED_FUNCTIONS = {"agent_loop"}

# Functions that hold waits but are NOT one trajectory, so no tail is defined for them. These must
# NOT carry @traced_trajectory: collect_batched issues one engine request for a whole batch and
# loops the environment over every row, so a scope there would publish a batch-wide SUM under a
# _seconds_max name. Absence is the only true answer.
TAIL_FREE_FUNCTIONS = {"collect_batched"}


# Deliberately the trajectory runners ONLY: a trajectory scope is a runner concept, and the engine
# client has no trajectory to scope. Unlike EXPECTED_TOKENIZE_REGIONS, widening this list would be
# wrong rather than merely cheap.
@pytest.mark.parametrize("module_name", ["trajectory_runners.skyrl_gym", "trajectory_runners.step_wise"])
def test_every_wait_site_is_inside_a_trajectory_scope(module_name):
    """🚨 The regression test for a max published smaller than its own mean.

    The batched path once scoped only its engine await, leaving env.init / env.step / env.close
    outside it. Those waits still reached rollout_env_await_seconds_sum and _count, but never the
    per-trajectory dict -- so _seconds_max stayed at the 0.0 that mark_supported seeds, and the row
    published as a MEASURED zero beside a non-zero mean. Counting decorators would not have caught
    it; this walks the call sites.
    """
    tree = ast.parse(_module_source(module_name))

    scoped: set[int] = set()
    holder: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for line in range(node.lineno, node.end_lineno + 1):
            holder.setdefault(line, node.name)
        decorated = any(isinstance(d, ast.Name) and d.id == "traced_trajectory" for d in node.decorator_list)
        if node.name in TAIL_FREE_FUNCTIONS:
            assert not decorated, (
                f"{module_name}.{node.name} is not one trajectory; a scope here would publish a "
                "batch-wide sum under a _seconds_max name"
            )
            scoped.update(range(node.lineno, node.end_lineno + 1))
            continue
        if node.name not in TRAJECTORY_SCOPED_FUNCTIONS:
            continue
        assert decorated, f"{module_name}.{node.name} holds waits but is not @traced_trajectory"
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
                f"{module_name}:{node.lineno} {name}() in {holder.get(node.lineno)!r}, in neither "
                "TRAJECTORY_SCOPED_FUNCTIONS nor TAIL_FREE_FUNCTIONS"
            )
    assert not unscoped, "\n".join(unscoped)


def test_every_agent_loop_scopes_its_own_trajectory():
    """Without it there is no tail, only a mean -- and F20 says the mean is the wrong statistic."""

    for owner in (SkyRLGymTrajectoryRunner, StepWiseRolloutCollector):
        assert getattr(owner.agent_loop, "__wrapped__", None) is not None, (
            f"{owner.__name__}.agent_loop is not scoped as a trajectory"
        )
    # And the batched collector deliberately is NOT: one batch is not one trajectory.
    assert getattr(SkyRLGymTrajectoryRunner.collect_batched, "__wrapped__", None) is None


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


def test_the_SHIPPED_runner_puts_its_own_cost_under_its_own_leaf(monkeypatch):
    """🚨 The shipped brackets, not a private stand-in.

    The positive tree test above drives a `_Bracketed` runner defined in this file, so relabelling
    the REAL `rollout_span("rollout_collect")` in skyrl_gym.py to `rollout_finalize` left the whole
    suite green -- `mark_supported()` seeds every declared leaf at 0.0, so `rollout_collect` was
    still PRESENT, still a key, just permanently zero while collection time piled into the wrong
    leaf. Asserting a key exists is not asserting it measured anything.

    `_run` takes its collector and its projection as injected seams, so the shipped method runs here
    with fakes that cost a known, DISTINCT number of seconds each.
    """
    from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner

    clock = {"now": 0.0}

    class _Collector:
        async def collect(self, input_batch, disable_tqdm=False):
            clock["now"] += 2.0
            return {"rows": []}

    class _Projection:
        def project(self, outputs, input_batch):
            clock["now"] += 3.0
            return {"response_ids": [], "loss_masks": [], "rollout_metrics": None, "rewards": []}

    runner = object.__new__(SkyRLGymTrajectoryRunner)
    runner.collector = _Collector()
    runner.projection = _Projection()
    runner.trajectory_sink = None
    runner.trajectory_runner_cfg = {}

    monkeypatch.setattr(timing_module, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    timings = RolloutTimings()
    asyncio.run(runner.run({}, phase_timings=timings))

    assert timings.supported is True, "the shipped runner is certified and must mark the accumulator"
    # ⚠️ The exact seconds, not mere presence. Every declared leaf is seeded at 0.0, so a relabelled
    # bracket leaves this key in place reading zero -- which is what let the mutation survive.
    assert timings.durations["rollout_collect"] == pytest.approx(2.0), (
        f"collection cost landed as {timings.durations.get('rollout_collect')}; a relabelled bracket "
        "leaves the seeded 0.0 here and moves the real seconds to another leaf"
    )
    assert timings.durations["rollout_assemble"] == pytest.approx(3.0)
    # rollout_finalize wraps the shared epilogue, which does no work on this input.
    assert timings.durations["rollout_finalize"] == pytest.approx(0.0)


def test_the_shipped_client_charges_its_templating_to_rollout_tokenize(monkeypatch):
    """🚨 Behavioural, because both walks are evadable and one aliasing mutation proved it.

    `tokenizer = self.tokenizer; tokenizer.apply_chat_template(...)` keeps the wrapper count right,
    keeps the module discoverable (it still contains another `self.tokenizer.`), and is invisible to
    an AST walk that only matches calls whose immediate owner is an attribute named `tokenizer`.
    Production then charges a full-batch templating to `rollout_engine_await` again.

    So drive the shipped `InferenceEngineClient.generate` on the `prompts=` form with a tokenizer
    that costs a known number of seconds, and assert those seconds land on `rollout_tokenize`.
    """
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

    clock = {"now": 0.0}

    class _Tokenizer:
        def apply_chat_template(self, prompts, **kwargs):
            clock["now"] += 4.0
            return {"input_ids": [[1, 2, 3]]}

    class _Engine:
        async def generate(self, engine_input):
            clock["now"] += 11.0
            return {
                "responses": ["r"],
                "stop_reasons": ["stop"],
                "response_ids": [[4, 5]],
                "response_logprobs": None,
            }

    client = object.__new__(InferenceEngineClient)
    client.tokenizer = _Tokenizer()
    client.engines = [_Engine()]
    # The client checks these before dispatching; neither is part of what this test measures.
    client.generation_paused_event = SimpleNamespace(is_set=lambda: False)
    client.enable_http_endpoint = False
    client._dead_engines = set()

    monkeypatch.setattr(timing_module, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    async def _drive():
        timings = RolloutTimings()
        timings.mark_supported()
        with rollout_timings_scope(timings):
            with rollout_wait(ROLLOUT_ENGINE_AWAIT):
                await client.generate({"prompts": [[{"role": "user", "content": "hi"}]]})
        return timings

    # ⚠️ NO try/skip here, deliberately. An earlier version wrapped this in a catch-all that skipped
    # on any exception, so a harness broken by an unrelated change would silently retire the ONLY
    # behavioural guard on this path -- and a test that does not run is indistinguishable from one
    # that passes. If the client needs more scaffolding, this must fail and say so.
    timings = asyncio.run(_drive())

    assert timings.durations.get("rollout_tokenize") == pytest.approx(4.0), (
        f"templating cost landed as {timings.durations.get('rollout_tokenize')}; an aliased "
        "tokenizer call is invisible to the AST walk and charges this to the engine wait"
    )
    # 🚨 Bound the WAIT as well as the leaf, which is what closes the CLASS rather than a spelling.
    # The wait is 4.0 of templating plus 11.0 of engine; any additional tokenizer call inside it --
    # aliased, bound-method, attribute-held, however spelled -- moves this number, whether or not a
    # source walk can see how it was written.
    assert timings.counters[f"{ROLLOUT_ENGINE_AWAIT}_seconds_sum"] == pytest.approx(15.0), (
        "the observed client wait changed, so something else ran inside it"
    )


def test_the_trainer_hands_its_accumulator_to_the_runner(monkeypatch):
    """🚨 The forwarding boundary, driven -- the one link a source search cannot hold.

    The guard for this searched `trainer.py` for the text `phase_timings=phase_timings`. Passing
    `phase_timings and None` keeps that text intact, hands the runner nothing, and training carries
    on: every generate leaf, the residual, and every wait counter vanish together, with a green
    suite and no warning. Absence is indistinguishable from an uninstrumented runner, which the tree
    deliberately renders as silence.

    So drive the real `RayPPOTrainer.generate` with a runner that measures a known cost, and assert
    that cost arrives in the accumulator the trainer was given.
    """
    from skyrl_train.trainer import RayPPOTrainer

    clock = {"now": 0.0}

    class _Runner:
        async def run(self, input_batch, phase_timings=None):
            # A certified runner does exactly this, and does nothing when handed None.
            if phase_timings is not None:
                phase_timings.mark_supported()
            with rollout_timings_scope(phase_timings):
                with rollout_span("rollout_collect"):
                    clock["now"] += 7.0
            return {"response_ids": [[1]], "rollout_metrics": None}

    trainer = object.__new__(RayPPOTrainer)
    trainer.trajectory_runner = _Runner()
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(step_wise_training=True))

    monkeypatch.setattr(timing_module, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    timings = RolloutTimings()
    asyncio.run(trainer.generate({"prompts": [[{"role": "user", "content": "hi"}]]}, phase_timings=timings))

    assert timings.supported is True, "the trainer handed the runner no accumulator to mark"
    assert timings.durations.get("rollout_collect") == pytest.approx(7.0), (
        f"the runner measured {timings.durations.get('rollout_collect')} into the trainer's "
        "accumulator; handing it None makes the whole generate tree disappear silently"
    )
