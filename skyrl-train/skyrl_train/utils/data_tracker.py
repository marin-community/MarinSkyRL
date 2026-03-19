"""
Data consumption tracking for SkyRL training.

Provides a reusable, trainer-agnostic tracker for which training data has been
consumed. Designed to be persisted via the callback system (DataTrackingCallback)
rather than ad-hoc inline checkpoint code.

Key properties:
- Tracks consumed UIDs per-epoch (for skip-on-resume in async training)
- Tracks monotonic total_samples_consumed across all epochs (for validation)
- Thread-safe via asyncio.Lock
- Clean serialization via get_state() / load_state()
"""

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

from loguru import logger


@dataclass
class DataConsumptionState:
    """Persistable snapshot of data consumption state."""

    global_step: int
    epoch: int
    consumed_uids_in_epoch: List[str]
    total_samples_consumed: int


class DataConsumptionTracker:
    """
    Tracks which training data has been consumed across steps and epochs.

    Unlike the previous approach (an in-memory Set cleared at epoch boundaries
    with blind counting assertions), this tracker:
    - Maintains both epoch-scoped UIDs (for skip-on-resume) and a monotonic
      total count (for validation without fragile assertions)
    - Owns epoch transitions explicitly via on_epoch_end()
    - Serializes cleanly via get_state() / load_state()

    The epoch-scoped UID clearing is driven by the DataTrackingCallback's
    on_epoch_end hook, which fires AFTER any checkpoint save — eliminating
    the race condition where UIDs were cleared before the checkpoint captured them.
    """

    def __init__(self, mini_batch_size: int, num_steps_per_epoch: int):
        self._mini_batch_size = mini_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._consumed_uids_in_epoch: Set[str] = set()
        self._total_samples_consumed: int = 0
        self._current_epoch: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def mark_consumed(self, uids: Iterable[str]) -> None:
        """Record UIDs as consumed after a training step."""
        async with self._lock:
            for uid in uids:
                if uid in self._consumed_uids_in_epoch:
                    logger.warning(
                        f"Duplicate UID {uid} in epoch {self._current_epoch}, skipping"
                    )
                    continue
                self._consumed_uids_in_epoch.add(uid)
                self._total_samples_consumed += 1

    async def on_epoch_end(self) -> None:
        """Clear epoch-scoped UIDs and advance epoch counter.

        Called by DataTrackingCallback.on_epoch_end_async, which fires AFTER
        any checkpoint save at the last step — so the checkpoint always
        captures the full epoch's UIDs before this clears them.
        """
        async with self._lock:
            logger.info(
                f"Epoch {self._current_epoch} end: clearing {len(self._consumed_uids_in_epoch)} "
                f"consumed UIDs (total consumed: {self._total_samples_consumed})"
            )
            self._consumed_uids_in_epoch.clear()
            self._current_epoch += 1

    def get_state(self) -> DataConsumptionState:
        """Snapshot current state for checkpointing."""
        return DataConsumptionState(
            global_step=-1,  # filled by the callback at save time
            epoch=self._current_epoch,
            consumed_uids_in_epoch=list(self._consumed_uids_in_epoch),
            total_samples_consumed=self._total_samples_consumed,
        )

    def load_state(self, state: DataConsumptionState) -> None:
        """Restore from a checkpoint snapshot."""
        self._consumed_uids_in_epoch = set(state.consumed_uids_in_epoch)
        self._total_samples_consumed = state.total_samples_consumed
        self._current_epoch = state.epoch
        logger.info(
            f"Loaded data tracker state: epoch={state.epoch}, "
            f"consumed_in_epoch={len(state.consumed_uids_in_epoch)}, "
            f"total_consumed={state.total_samples_consumed}"
        )

    def get_consumed_uids_in_epoch(self) -> Set[str]:
        """Return the set of UIDs consumed in the current epoch (for skip-on-resume)."""
        return set(self._consumed_uids_in_epoch)

    @property
    def total_samples_consumed(self) -> int:
        return self._total_samples_consumed

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def consumed_in_epoch_count(self) -> int:
        return len(self._consumed_uids_in_epoch)
