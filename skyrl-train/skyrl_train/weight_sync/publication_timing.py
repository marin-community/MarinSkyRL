"""Publication stage clocks; CUDA events are resolved only after the transfer."""

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
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


def record_receiver_publication_stages(receivers: list[list[dict]], *, step: int) -> None:
    """Export native receiver clocks through the trainer's telemetry lifecycle.

    vLLM worker processes return receipts but do not own an exporter. Keep the
    measured worker identity distinct from the trainer that transports the event.
    Validate the whole batch before emitting any event; do not remeasure clocks.
    """
    from skyrl_train.telemetry import record_event

    events = []
    for engine_index, workers in enumerate(receivers):
        if not workers:
            raise ValueError("missing weight-sync receiver workers")
        ranks = set()
        for worker in workers:
            rank = worker["rank"]
            if rank in ranks or worker["step"] != step:
                raise ValueError("duplicate receiver rank or wrong weight-sync step")
            ranks.add(rank)
            if not worker["hostname"] or worker["pid"] <= 0:
                raise ValueError("missing weight-sync receiver origin")
            if not worker["stages"]:
                raise ValueError("missing weight-sync receiver stages")
            for stage, values in worker["stages"].items():
                if values["calls"] <= 0 or not math.isfinite(values["wall_seconds"]) or values["wall_seconds"] < 0:
                    raise ValueError("invalid weight-sync receiver wall clock")
                gpu_ms = values["gpu_ms"]
                if gpu_ms is not None and (not math.isfinite(gpu_ms) or gpu_ms < 0):
                    raise ValueError("invalid weight-sync receiver CUDA clock")
                events.append(
                    (
                        values,
                        {
                            "role": "inference",
                            "exporter_role": "trainer",
                            "engine_index": str(engine_index),
                            "origin_host": worker["hostname"],
                            "origin_pid": str(worker["pid"]),
                            "rank": str(rank),
                            "step": str(step),
                            "stage": stage,
                        },
                    )
                )
    if not events:
        raise ValueError("missing weight-sync receiver receipts")
    for values, attributes in events:
        record_event("publication_stage", values, attributes=attributes)
