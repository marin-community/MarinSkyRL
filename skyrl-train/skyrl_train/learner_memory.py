"""Worker-local CUDA allocator observations for bounded learner phase scopes.

Enabled by policy_train_spans, using the worker's existing telemetry lifecycle.
No CUDA synchronization, cache eviction, NVML sampling, or exporter flush is added.
Allocator peaks include the resident baseline and only cover PyTorch allocations
on this process/device, not whole-device usage. CUDA free/total are instantaneous
whole-device samples. Model-ready is before any lazily initialized Adam state;
the first successful update's exit supplies the corresponding warm baseline.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

import torch
from loguru import logger

from skyrl_train.telemetry import WORKER_ROLE, record_event


# CUDA peak counters belong to the process/device, not a Python scope. Never
# let nested or concurrent observations reset an active interval's counters.
_peak_locks: dict[int, Lock] = {}


class LearnerMemory:
    """Emit entry/exit events and native allocator interval peaks for one worker.

    Observation errors disable subsequent collection and warn once, without
    replacing training exceptions. Overlapping scopes are skipped with a warning;
    the enclosing interval still includes all allocations on its device.
    """

    def __init__(self, *, enabled: bool, rank: int) -> None:
        self.enabled = enabled
        self._rank = rank
        self._device: int | None = None
        self._identity: dict[str, str] = {}
        self._warned_overlap = False

    def _initialize_identity(self) -> int:
        if self._device is None:
            device = torch.cuda.current_device()
            allocator_backend = torch.cuda.get_allocator_backend()
            if allocator_backend != "native":
                raise RuntimeError(f"learner allocator peaks require native CUDA allocator, got {allocator_backend}")
            self._identity = {
                "backend": "megatron",
                "role": WORKER_ROLE,
                "worker_role": "policy",
                "rank": str(self._rank),
                "cuda_device": str(device),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "allocator_backend": allocator_backend,
            }
            self._device = device
        return self._device

    def _record(self, *, phase: str, boundary: str, outcome: str, step: int | None, step_kind: str) -> None:
        device = self._initialize_identity()
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        fields = {
            "allocated_bytes": stats["allocated_bytes.all.current"],
            "reserved_bytes": stats["reserved_bytes.all.current"],
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
        if boundary == "exit":
            fields.update(
                peak_allocated_bytes=stats["allocated_bytes.all.peak"],
                peak_reserved_bytes=stats["reserved_bytes.all.peak"],
            )
        attributes = {
            **self._identity,
            "phase": phase,
            "boundary": boundary,
            "outcome": outcome,
            "step_kind": step_kind if step is not None else "unknown",
        }
        if step is not None:
            attributes["step"] = str(step)
        record_event("cuda_memory_observation", fields, attributes=attributes)

    def _disable(self, phase: str, error: Exception) -> None:
        self.enabled = False
        logger.warning("Disabling learner CUDA memory observations after phase {} failed: {}", phase, error)

    def snapshot(self, phase: str, *, step: int | None = None, step_kind: str = "completed_update") -> None:
        """Sample current memory without resetting or publishing interval peaks."""
        if not self.enabled:
            return
        try:
            self._record(phase=phase, boundary="snapshot", outcome="success", step=step, step_kind=step_kind)
        except Exception as error:
            self._disable(phase, error)

    @contextmanager
    def span(self, phase: str, *, step: int | None, step_kind: str) -> Iterator[None]:
        """Measure one phase; exceptions retain their identity and a failure exit."""
        if not self.enabled:
            yield
            return

        acquired = False
        lock = None
        try:
            device = self._initialize_identity()
            lock = _peak_locks.setdefault(device, Lock())
            acquired = lock.acquire(blocking=False)
            if acquired:
                torch.cuda.reset_peak_memory_stats(device)
                self._record(phase=phase, boundary="enter", outcome="started", step=step, step_kind=step_kind)
            elif not self._warned_overlap:
                self._warned_overlap = True
                logger.warning("Skipping overlapping learner CUDA memory phase {} on device {}", phase, device)
        except Exception as error:
            self._disable(phase, error)

        outcome = "success"
        try:
            yield
        except BaseException:
            outcome = "failure"
            raise
        finally:
            if acquired:
                try:
                    if self.enabled:
                        self._record(phase=phase, boundary="exit", outcome=outcome, step=step, step_kind=step_kind)
                except Exception as error:
                    self._disable(phase, error)
                finally:
                    assert lock is not None
                    lock.release()
