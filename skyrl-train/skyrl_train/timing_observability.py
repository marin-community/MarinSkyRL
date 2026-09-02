"""Sink-neutral timing observations and their publishing adapters."""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# Via skyrl_train.telemetry, NOT `from rigging import telemetry`: that module guards the import
# and falls back to inert_telemetry, because an installed rigging without the telemetry
# submodule raises ImportError. This module is imported at the top of trainer.py and worker.py,
# so an unguarded import here would take the whole trainer down on that install shape.
from skyrl_train.telemetry import TRAINER_ROLE, WORKER_ROLE, phase_duration, telemetry


TIMING_PARENTS: dict[str, str | None] = {
    "step": None,
    "generate": "step",
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
    "policy_span_publish": "policy_ppo_train",
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


class FinelogTimingSink:
    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None:
        for observation in observations:
            phase_duration.record(
                observation.duration_seconds,
                attributes={
                    "phase": observation.name,
                    "root": observation.root,
                    "parent": observation.parent or "",
                    "clock_domain": "inclusive_wall",
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
    "policy_span_publish",
    "policy_forward",
    "policy_backward",
    "policy_optimizer_step",
    "policy_entropy_allreduce",
    "policy_metric_allreduce",
    "policy_final_barrier",
)


policy_step_counter = telemetry.histogram("policy_train_count", unit="1")


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
        policy_step_counter.record(
            float(value),
            attributes={"counter": name, "role": WORKER_ROLE, "rank": str(rank), "step": str(step)},
        )


def unconfigured_telemetry_reason() -> str | None:
    """Why these spans would publish nothing, or None if they will publish.

    Checked once at worker start, because the failure is otherwise invisible: on an unconfigured
    runtime ``record`` is discarded, ``flush`` returns **True**, and the loss counters stay at zero.
    Every signal reads healthy and the run produces no rows at all -- a full step spent to learn
    nothing. Verified empirically, not assumed.
    """
    status = telemetry.runtime_status()
    if getattr(status, "configured", False):
        return None
    return (
        "rigging telemetry is not configured in this process, so every policy_train span will be "
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


class WorkerTimingSink:
    """Publish worker spans under their own role and clock domain.

    ``clock_domain`` is deliberately not ``inclusive_wall``: these are exclusive durations, and
    mixing the two under one domain would let a consumer sum a child into its parent twice.
    """

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None:
        for observation in observations:
            phase_duration.record(
                observation.duration_seconds,
                attributes={
                    "phase": observation.name,
                    "root": observation.root,
                    "parent": TIMING_PARENTS.get(observation.name) or "",
                    "clock_domain": (
                        "inclusive_wall"
                        if observation.name in ("policy_ppo_train", "policy_training_step")
                        else "exclusive_wall"
                    ),
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
    sink = WorkerTimingSink(rank)
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
