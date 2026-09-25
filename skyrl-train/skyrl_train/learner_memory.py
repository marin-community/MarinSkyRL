"""CUDA allocator observations for bounded phases of a learner worker.

Enabled by policy_train_spans. The recorder adds no CUDA synchronization, cache
eviction, NVML sampling or exporter flush. Allocator peaks include the resident
baseline and cover only this process's PyTorch allocations on its device; CUDA
free and total are instantaneous whole-device samples. The model-ready sample
precedes lazily initialized Adam state, so the first successful update's exit is
the warm baseline.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import torch
from loguru import logger

from skyrl_train.telemetry import WORKER_ROLE, StepKind, record_event


@dataclass
class _PeakScope:
    participants: int = 1
    overlapping: bool = False


# CUDA peak counters belong to the process/device. An overlapping interval
# remains occupied until every participant exits, even if its first owner exits
# early. Its peak cannot be attributed to one phase and must not be published.
_peak_scopes: dict[int, _PeakScope] = {}
_peak_scope_lock = Lock()


class LearnerCudaMetrics:
    """Record CUDA allocator samples and interval peaks around worker phases."""

    def __init__(self, *, enabled: bool, rank: int, backend: str = "megatron") -> None:
        self.enabled = enabled
        self._rank = rank
        self._backend = backend
        self._device: int | None = None
        self._identity: dict[str, str] = {}
        self._warned_overlap = False

    def _identified_device(self) -> int:
        """Return this worker's CUDA device index, building the identity attributes on first use."""
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
        self,
        *,
        phase: str,
        boundary: str,
        outcome: str,
        step: int | None,
        step_kind: StepKind,
        overlapping: bool = False,
    ) -> None:
        device = self._identified_device()
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
            # str() keeps the exported value byte-identical to the literal the
            # dashboards were built against, whatever a serializer does with enums.
            "step_kind": str(step_kind if step is not None else StepKind.UNKNOWN),
        }
        if step is not None:
            attributes["step"] = str(step)
        if overlapping:
            attributes["scope_overlap"] = "true"
        record_event("cuda_memory_observation", fields, attributes=attributes)

    def _disable(self, phase: str, error: Exception) -> None:
        self.enabled = False
        logger.warning("Disabling learner CUDA memory observations after phase {} failed: {}", phase, error)

    def snapshot(
        self, phase: str, *, step: int | None = None, step_kind: StepKind = StepKind.MODEL_VERSION_STEP
    ) -> None:
        """Sample current memory without resetting or publishing interval peaks."""
        if not self.enabled:
            return
        try:
            self._record(phase=phase, boundary="snapshot", outcome="success", step=step, step_kind=step_kind)
        except Exception as error:
            self._disable(phase, error)

    @contextmanager
    def span(self, phase: str, *, step: int | None, step_kind: StepKind) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        acquired = False
        scope = None
        try:
            device = self._identified_device()
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


# A disabled recorder does nothing and holds no per-worker state, so one shared
# instance serves as the class-level default for workers built without a config.
INERT_LEARNER_CUDA_METRICS = LearnerCudaMetrics(enabled=False, rank=0, backend="unknown")
