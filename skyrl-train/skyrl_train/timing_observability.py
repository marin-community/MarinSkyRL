"""Sink-neutral timing observations and their publishing adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from skyrl_train.telemetry import TRAINER_ROLE, phase_duration


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
    "policy_critic_overlap_train": "train_critic_and_policy",
    "sync_weights": "step",
    "offload_policy_model_to_cpu": "step",
    "dump_data_batch": "run_training",
    "init_weight_sync_state": None,
    "save_checkpoints": "step",
    "cleanup_old_checkpoints": "save_checkpoints",
    "save_hf_model": "step",
    "queue_hf_export": "step",
    "eval": "step",
    "update_ref_with_policy": "step",
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
