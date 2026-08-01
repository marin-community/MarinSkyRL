"""Cases for the opt-in EP/FSDP collective schedule matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CheckpointMode(StrEnum):
    NONE = "none"
    REENTRANT = "reentrant"


class RoutingMode(StrEnum):
    LIVE = "live"
    REPLAY_SPREAD = "replay-spread"
    REPLAY_HOT = "replay-hot"


@dataclass(frozen=True)
class CollectiveScheduleCase:
    name: str
    checkpoint_mode: CheckpointMode
    routing_mode: RoutingMode
    delayed_rank: int | None = None


COLLECTIVE_SCHEDULE_CASES = (
    CollectiveScheduleCase("live", CheckpointMode.NONE, RoutingMode.LIVE),
    CollectiveScheduleCase("replay-spread", CheckpointMode.NONE, RoutingMode.REPLAY_SPREAD),
    CollectiveScheduleCase("checkpoint-live", CheckpointMode.REENTRANT, RoutingMode.LIVE),
    CollectiveScheduleCase("checkpoint-replay-spread", CheckpointMode.REENTRANT, RoutingMode.REPLAY_SPREAD),
    CollectiveScheduleCase("checkpoint-replay-hot", CheckpointMode.REENTRANT, RoutingMode.REPLAY_HOT),
    CollectiveScheduleCase("checkpoint-replay-hot-delay", CheckpointMode.REENTRANT, RoutingMode.REPLAY_HOT, 0),
)

DEFAULT_COLLECTIVE_SCHEDULE_CASE = "checkpoint-replay-hot"


def collective_schedule_case(name: str) -> CollectiveScheduleCase:
    """Resolve a named schedule case or fail with the accepted names."""

    for case in COLLECTIVE_SCHEDULE_CASES:
        if case.name == name:
            return case
    accepted = ", ".join(case.name for case in COLLECTIVE_SCHEDULE_CASES)
    raise ValueError(f"unknown collective schedule case {name!r}; expected one of: {accepted}")
