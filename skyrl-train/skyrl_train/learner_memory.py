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
from dataclasses import dataclass
from threading import Lock

import torch
from loguru import logger

from skyrl_train.telemetry import WORKER_ROLE, record_event


@dataclass
class _PeakScope:
    participants: int = 1
    overlapping: bool = False


# CUDA peak counters belong to the process/device. An overlapping interval
# remains occupied until every participant exits, even if its first owner exits
# early. Its peak cannot be attributed to one phase and must not be published.
_peak_scopes: dict[int, _PeakScope] = {}
_peak_scope_lock = Lock()


class LearnerMemory:
    """Emit entry/exit events and native allocator interval peaks for one worker.

    Observation errors disable subsequent collection and warn once, without
    replacing training exceptions. Overlapping scopes are skipped with a warning;
    the enclosing exit omits peaks and marks scope_overlap=true.
    """

    def __init__(self, *, enabled: bool, rank: int, backend: str = "megatron") -> None:
        self.enabled = enabled
        self._rank = rank
        self._backend = backend
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
                "backend": self._backend,
                "role": WORKER_ROLE,
                "worker_role": "policy",
                "rank": str(self._rank),
                "cuda_device": str(device),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "allocator_backend": allocator_backend,
            }
            self._device = device
        return self._device

    def _record(
        self, *, phase: str, boundary: str, outcome: str, step: int | None, step_kind: str, overlapping: bool = False
    ) -> None:
        device = self._initialize_identity()
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        fields = {
            "allocated_bytes": stats["allocated_bytes.all.current"],
            "reserved_bytes": stats["reserved_bytes.all.current"],
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
        if boundary == "exit" and not overlapping:
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
        if overlapping:
            attributes["scope_overlap"] = "true"
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
        scope = None
        try:
            device = self._initialize_identity()
            with _peak_scope_lock:
                scope = _peak_scopes.get(device)
                if scope is None:
                    scope = _PeakScope()
                    _peak_scopes[device] = scope
                    acquired = True
                else:
                    scope.participants += 1
                    scope.overlapping = True
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
            if scope is not None:
                # Serialize the last sample against a new entrant. The guard is
                # never held while model work or awaited publication runs.
                with _peak_scope_lock:
                    try:
                        if acquired and self.enabled:
                            self._record(
                                phase=phase,
                                boundary="exit",
                                outcome=outcome,
                                step=step,
                                step_kind=step_kind,
                                overlapping=scope.overlapping,
                            )
                    except Exception as error:
                        self._disable(phase, error)
                    finally:
                        scope.participants -= 1
                        if scope.participants == 0:
                            del _peak_scopes[device]
