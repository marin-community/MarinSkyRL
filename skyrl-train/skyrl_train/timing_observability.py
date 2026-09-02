"""Sink-neutral timing observations and their publishing adapters."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rigging import telemetry

from skyrl_train.telemetry import TRAINER_ROLE, WORKER_ROLE, phase_duration


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
# aggregation (max, p95) happens at query time over the rank attribute.
#
# These do NOT close against the driver's policy_train.
#
#     policy_train - max_over_ranks(policy_ppo_train)
#
# is an UPPER BOUND on driver-side overhead, not an isolation of it. It also contains the skew
# between ranks' start times -- the worker intervals are measured on different hosts against
# unsynchronised clocks -- plus the worker epilogue after the span closes and the transport of the
# result back. Quote it as a bound and never as "dispatch cost".
#
# policy_ppo_train does start at true function entry, before the R3 co-arrival drain, so the arrival
# spread the driver pays for lands in policy_entry_barrier rather than vanishing.

TELEMETRY_FLUSH_TIMEOUT_SECONDS = 5.0

POLICY_TRAIN_SPANS = (
    "policy_entry_barrier",
    "policy_training_step",
    "policy_metric_allreduce",
    "policy_final_barrier",
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
            totals["policy_span_residual"] = max(total_seconds - covered, 0.0)
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
                    "clock_domain": ("inclusive_wall" if observation.name == "policy_ppo_train" else "exclusive_wall"),
                    "role": WORKER_ROLE,
                    "rank": str(self.rank),
                    "step": str(step),
                },
            )


def publish_worker_spans(timings: Mapping[str, float], *, step: int, rank: int) -> None:
    """Publish one worker's policy_train decomposition, then settle the queue.

    The flush is the durability mechanism, and it has to be here rather than at shutdown: Ray tears
    actors down with ``ray.kill``, which does not run ``atexit`` handlers, and even the graceful path
    allows less than Rigging's request timeout. Relying on shutdown loses the last step's rows from
    exactly the slowest ranks -- which would understate max and p95 and inflate the apparent driver
    gap, i.e. bias every number in the direction that makes the instrumentation look successful.
    """
    if not timings:
        return
    WorkerTimingSink(rank).publish(phase_timing_observations(timings), step)
    telemetry.flush(TELEMETRY_FLUSH_TIMEOUT_SECONDS)
