"""Publication stage clocks; CUDA events are resolved only after the transfer."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter

import torch


@dataclass(frozen=True)
class PublicationStage:
    wall_seconds: float
    gpu_ms: float | None
    calls: int


class PublicationStageTimer:
    """Accumulate repeated stages without synchronizing between tensors.

    GPU elapsed times belong to the stream executing the span. They can include
    stream idle time during blocking host work and are not kernel utilization.
    """

    def __init__(self, *, enabled: bool, now: Callable[[], float] = perf_counter):
        self.enabled = enabled
        self.now = now
        self.cuda = enabled and torch.cuda.is_available()
        self.active = {}
        self.walls = defaultdict(float)
        self.events = defaultdict(list)
        self.calls = defaultdict(int)

    def start(self, stage: str) -> None:
        if not self.enabled:
            return
        if stage in self.active:
            raise RuntimeError(f"publication stage already started: {stage}")
        event = torch.cuda.Event(enable_timing=True) if self.cuda else None
        if event is not None:
            event.record()
        self.active[stage] = (self.now(), event)

    def stop(self, stage: str) -> None:
        if not self.enabled:
            return
        if stage not in self.active:
            raise RuntimeError(f"publication stage not started: {stage}")
        started, first = self.active.pop(stage)
        self.walls[stage] += self.now() - started
        self.calls[stage] += 1
        if first is not None:
            last = torch.cuda.Event(enable_timing=True)
            last.record()
            self.events[stage].append((first, last))

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)

    def finish(self, *, synchronized: bool = False) -> dict[str, dict]:
        """Return a serializable receipt, after one final CUDA synchronization."""
        if self.active:
            raise RuntimeError(f"unfinished publication stages: {sorted(self.active)}")
        if self.cuda and not synchronized:
            torch.cuda.synchronize()
        result = {
            stage: asdict(
                PublicationStage(
                    wall_seconds=wall,
                    gpu_ms=sum(first.elapsed_time(last) for first, last in self.events[stage]) if self.cuda else None,
                    calls=self.calls[stage],
                )
            )
            for stage, wall in self.walls.items()
        }
        self.walls.clear()
        self.events.clear()
        self.calls.clear()
        return result


def publication_stage_walls(receipt: dict) -> dict[str, float]:
    """Fold ranks by maximum wall time; concurrent stages remain non-additive."""
    walls: dict[str, float] = {}
    ranks = receipt["trainer"] + [worker for engine in receipt["receiver"] for worker in engine]
    for rank in ranks:
        for stage, values in rank["stages"].items():
            key = f"weight_broadcast/{stage}"
            walls[key] = max(walls.get(key, 0.0), values["wall_seconds"])
    return walls
