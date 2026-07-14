"""
Trial archiving callback and path tracking utilities for terminal bench.

This module provides:
- TrialPathTracker: Helper class to track active and completed trial paths
- TrialArchiveCallback: Async callback for archiving completed trials to tar archives
"""

import asyncio
import os
import shutil
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from loguru import logger

from skyrl_train.callbacks import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AtomicStepCounter,
)


@dataclass
class TrialPathInfo:
    """Information about a trial's paths and status."""
    trial_id: str
    trial_path: Path
    created_at: datetime
    global_step_created: int
    is_completed: bool = False
    completed_at: Optional[datetime] = None
    global_step_completed: Optional[int] = None
    is_archived: bool = False
    archived_at: Optional[datetime] = None


class TrialPathTracker:
    """
    Thread-safe helper for tracking trial paths during terminal bench training.

    This class maintains a registry of all trial directories in trace_jobs,
    tracking which trials are active (still being processed) vs completed
    (finished batch processing and ready for archiving).

    Thread-safety is ensured via a threading.RLock for all operations, making
    it safe to use from multiple async workers.

    Attributes:
        trials_dir: Base directory containing trial subdirectories
        active_trials: Set of trial IDs currently being processed
        completed_trials: Set of trial IDs that have finished processing
        archived_trials: Set of trial IDs that have been archived

    Example:
        ```python
        tracker = TrialPathTracker("/path/to/trace_jobs")

        # When a trial starts
        tracker.register_trial("trial_001", global_step=5)

        # When a trial completes
        tracker.mark_completed("trial_001", global_step=5)

        # Get completed trials ready for archiving
        ready = tracker.get_completed_unarchived()

        # After archiving
        tracker.mark_archived(["trial_001"])
        ```
    """

    def __init__(self, trials_dir: str):
        """
        Initialize the trial path tracker.

        Args:
            trials_dir: Base directory containing trial subdirectories
        """
        self.trials_dir = Path(trials_dir)
        self._trials: Dict[str, TrialPathInfo] = {}
        self._lock = threading.RLock()

        # Track step counts for determining archivability
        self._step_counter = AtomicStepCounter()

        logger.info(f"TrialPathTracker initialized with trials_dir={trials_dir}")

    def set_current_step(self, global_step: int) -> None:
        """Update the current global step (thread-safe)."""
        self._step_counter.set(global_step)

    def get_current_step(self) -> int:
        """Get the current global step (thread-safe)."""
        return self._step_counter.get()

    def register_trial(
        self,
        trial_id: str,
        trial_path: Optional[Path] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """
        Register a new trial as active.

        Args:
            trial_id: Unique identifier for the trial
            trial_path: Path to trial directory (defaults to trials_dir/trial_id)
            global_step: The global step when this trial was created
        """
        with self._lock:
            if trial_id in self._trials:
                logger.debug(f"Trial {trial_id} already registered, skipping")
                return

            if trial_path is None:
                trial_path = self.trials_dir / trial_id
            else:
                trial_path = Path(trial_path)

            step = global_step if global_step is not None else self._step_counter.get()

            self._trials[trial_id] = TrialPathInfo(
                trial_id=trial_id,
                trial_path=trial_path,
                created_at=datetime.now(),
                global_step_created=step,
            )
            logger.debug(f"Registered trial {trial_id} at step {step}")

    def mark_completed(
        self,
        trial_id: str,
        global_step: Optional[int] = None,
    ) -> bool:
        """
        Mark a trial as completed (ready for archiving after batch processed).

        Args:
            trial_id: The trial to mark as completed
            global_step: The global step when this trial was completed

        Returns:
            True if trial was found and marked, False otherwise
        """
        with self._lock:
            if trial_id not in self._trials:
                logger.warning(f"Cannot mark unknown trial {trial_id} as completed")
                return False

            trial = self._trials[trial_id]
            if trial.is_completed:
                logger.debug(f"Trial {trial_id} already marked completed")
                return True

            step = global_step if global_step is not None else self._step_counter.get()
            trial.is_completed = True
            trial.completed_at = datetime.now()
            trial.global_step_completed = step

            logger.debug(f"Marked trial {trial_id} as completed at step {step}")
            return True

    def mark_trials_completed_for_step(self, global_step: int) -> int:
        """
        Mark all active trials created before or at this step as completed.

        This is useful for marking all trials from a batch as completed once
        the training step has finished processing that batch.

        Args:
            global_step: The global step to use for completion cutoff

        Returns:
            Number of trials marked as completed
        """
        count = 0
        with self._lock:
            for trial in self._trials.values():
                if (
                    not trial.is_completed
                    and trial.global_step_created <= global_step
                ):
                    trial.is_completed = True
                    trial.completed_at = datetime.now()
                    trial.global_step_completed = global_step
                    count += 1

        if count > 0:
            logger.debug(f"Marked {count} trials as completed for step {global_step}")
        return count

    def mark_archived(self, trial_ids: List[str]) -> None:
        """
        Mark trials as archived.

        Args:
            trial_ids: List of trial IDs that have been archived
        """
        with self._lock:
            for trial_id in trial_ids:
                if trial_id in self._trials:
                    self._trials[trial_id].is_archived = True
                    self._trials[trial_id].archived_at = datetime.now()
            logger.debug(f"Marked {len(trial_ids)} trials as archived")

    def get_active_trials(self) -> List[TrialPathInfo]:
        """Get all active (non-completed) trials."""
        with self._lock:
            return [t for t in self._trials.values() if not t.is_completed]

    def get_completed_trials(self) -> List[TrialPathInfo]:
        """Get all completed trials (including archived)."""
        with self._lock:
            return [t for t in self._trials.values() if t.is_completed]

    def get_completed_unarchived(self) -> List[TrialPathInfo]:
        """Get completed trials that haven't been archived yet."""
        with self._lock:
            return [
                t for t in self._trials.values()
                if t.is_completed and not t.is_archived
            ]

    def get_archived_trials(self) -> List[TrialPathInfo]:
        """Get all archived trials."""
        with self._lock:
            return [t for t in self._trials.values() if t.is_archived]

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about tracked trials."""
        with self._lock:
            active = sum(1 for t in self._trials.values() if not t.is_completed)
            completed = sum(
                1 for t in self._trials.values()
                if t.is_completed and not t.is_archived
            )
            archived = sum(1 for t in self._trials.values() if t.is_archived)
            return {
                "total": len(self._trials),
                "active": active,
                "completed_unarchived": completed,
                "archived": archived,
            }

    def scan_trials_dir(self) -> int:
        """
        Scan the trials directory and register any untracked trial directories.

        Returns:
            Number of new trials registered
        """
        if not self.trials_dir.exists():
            logger.warning(f"Trials directory does not exist: {self.trials_dir}")
            return 0

        count = 0
        with self._lock:
            for item in self.trials_dir.iterdir():
                if item.is_dir() and item.name not in self._trials:
                    self.register_trial(item.name, trial_path=item)
                    count += 1

        if count > 0:
            logger.info(f"Scanned and registered {count} new trials from {self.trials_dir}")
        return count


class TrialArchiveCallback(TrainerCallback):
    """
    Async callback for archiving completed trials to tar archives.

    This callback:
    1. Tracks completed trials using a TrialPathTracker
    2. At specified intervals, archives completed trials to tar.gz files
    3. Removes the original directories after successful archiving

    The archiving happens asynchronously to not block the training loop.

    Args:
        tracker: TrialPathTracker instance for tracking trial paths
        archive_every_steps: Archive completed trials every N steps
        archive_dir: Directory to store archives (defaults to trials_dir/archives)
        compression: Compression type for tarfile ("gz", "bz2", "xz", or "")
        delete_after_archive: Whether to delete original dirs after archiving
        min_trials_to_archive: Minimum number of completed trials before archiving

    Example:
        ```python
        tracker = TrialPathTracker("/path/to/trace_jobs")
        callback = TrialArchiveCallback(
            tracker=tracker,
            archive_every_steps=5,
            delete_after_archive=True,
        )

        # Add to trainer callbacks
        trainer.callback_handler.add_callback(callback)
        ```
    """

    def __init__(
        self,
        tracker: TrialPathTracker,
        archive_every_steps: int = 5,
        archive_dir: Optional[str] = None,
        compression: str = "gz",
        delete_after_archive: bool = True,
        min_trials_to_archive: int = 1,
    ):
        self.tracker = tracker
        self.archive_every_steps = archive_every_steps
        self.compression = compression
        self.delete_after_archive = delete_after_archive
        self.min_trials_to_archive = min_trials_to_archive

        # Set up archive directory
        if archive_dir:
            self.archive_dir = Path(archive_dir)
        else:
            self.archive_dir = tracker.trials_dir / "archives"

        # Track archiving stats
        self._total_archived = 0
        self._total_bytes_archived = 0
        self._archive_lock = asyncio.Lock()

        logger.info(
            f"TrialArchiveCallback initialized: archive_every_steps={archive_every_steps}, "
            f"archive_dir={self.archive_dir}, compression={compression}, "
            f"delete_after_archive={delete_after_archive}"
        )

    async def on_step_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """
        Called at the end of each training step.

        Marks trials from the completed batch as ready for archiving,
        and triggers archiving if at the configured interval.
        """
        # Update the tracker's step counter
        self.tracker.set_current_step(state.global_step)

        # Mark all trials from this step's batch as completed
        self.tracker.mark_trials_completed_for_step(state.global_step)

        # Check if we should archive
        if (
            self.archive_every_steps > 0
            and state.global_step % self.archive_every_steps == 0
        ):
            await self._archive_completed_trials(state.global_step)

        return control

    async def on_train_end_async(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        """Archive any remaining completed trials at the end of training."""
        # Mark all remaining trials as completed
        self.tracker.mark_trials_completed_for_step(state.global_step)

        # Archive everything
        await self._archive_completed_trials(state.global_step, force=True)

        return control

    async def _archive_completed_trials(
        self,
        global_step: int,
        force: bool = False,
    ) -> int:
        """
        Archive completed trials to tar files.

        Args:
            global_step: Current global step (used in archive name)
            force: If True, archive even if below min_trials_to_archive

        Returns:
            Number of trials archived
        """
        async with self._archive_lock:
            # Get completed trials ready for archiving
            to_archive = self.tracker.get_completed_unarchived()

            if not to_archive:
                logger.debug("No completed trials to archive")
                return 0

            if not force and len(to_archive) < self.min_trials_to_archive:
                logger.debug(
                    f"Only {len(to_archive)} trials ready, need {self.min_trials_to_archive}"
                )
                return 0

            # Ensure archive directory exists
            self.archive_dir.mkdir(parents=True, exist_ok=True)

            # Create archive filename with timestamp and step
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = f".tar.{self.compression}" if self.compression else ".tar"
            archive_name = f"trials_step{global_step}_{timestamp}{ext}"
            archive_path = self.archive_dir / archive_name

            logger.info(
                f"Archiving {len(to_archive)} trials to {archive_path}"
            )

            # Run archiving in thread pool to not block event loop
            archived_trial_ids, bytes_archived = await asyncio.to_thread(
                self._create_archive,
                archive_path,
                to_archive,
            )

            if archived_trial_ids:
                # Mark trials as archived
                self.tracker.mark_archived(archived_trial_ids)
                self._total_archived += len(archived_trial_ids)
                self._total_bytes_archived += bytes_archived

                logger.info(
                    f"Archived {len(archived_trial_ids)} trials "
                    f"({bytes_archived / (1024*1024):.2f} MB) to {archive_path}. "
                    f"Total archived: {self._total_archived} trials, "
                    f"{self._total_bytes_archived / (1024*1024):.2f} MB"
                )

                # Delete original directories if configured
                if self.delete_after_archive:
                    await asyncio.to_thread(
                        self._delete_archived_trials,
                        to_archive,
                        archived_trial_ids,
                    )

            return len(archived_trial_ids)

    def _create_archive(
        self,
        archive_path: Path,
        trials: List[TrialPathInfo],
    ) -> tuple[List[str], int]:
        """
        Create a tar archive of the given trials.

        Args:
            archive_path: Path to create the archive
            trials: List of trial info to archive

        Returns:
            Tuple of (archived trial IDs, bytes archived)
        """
        mode = f"w:{self.compression}" if self.compression else "w"
        archived_ids = []
        bytes_archived = 0

        try:
            with tarfile.open(archive_path, mode) as tar:
                for trial in trials:
                    if trial.trial_path.exists():
                        try:
                            # Add directory to archive
                            tar.add(
                                trial.trial_path,
                                arcname=trial.trial_id,
                            )
                            archived_ids.append(trial.trial_id)

                            # Calculate size
                            for root, dirs, files in os.walk(trial.trial_path):
                                for f in files:
                                    bytes_archived += os.path.getsize(
                                        os.path.join(root, f)
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Failed to add trial {trial.trial_id} to archive: {e}"
                            )
                    else:
                        logger.warning(
                            f"Trial path does not exist: {trial.trial_path}"
                        )

        except Exception as e:
            logger.error(f"Failed to create archive {archive_path}: {e}")
            # Clean up partial archive
            if archive_path.exists():
                archive_path.unlink()
            return [], 0

        return archived_ids, bytes_archived

    def _delete_archived_trials(
        self,
        trials: List[TrialPathInfo],
        archived_ids: List[str],
    ) -> int:
        """
        Delete the original trial directories after archiving.

        Args:
            trials: All trials that were candidates for archiving
            archived_ids: IDs of trials that were successfully archived

        Returns:
            Number of directories deleted
        """
        archived_set = set(archived_ids)
        deleted = 0

        for trial in trials:
            if trial.trial_id in archived_set and trial.trial_path.exists():
                try:
                    shutil.rmtree(trial.trial_path)
                    deleted += 1
                    logger.debug(f"Deleted archived trial directory: {trial.trial_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete trial directory {trial.trial_path}: {e}"
                    )

        if deleted > 0:
            logger.info(f"Deleted {deleted} archived trial directories")

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get archiving statistics."""
        tracker_stats = self.tracker.get_stats()
        return {
            **tracker_stats,
            "total_archived_by_callback": self._total_archived,
            "total_bytes_archived": self._total_bytes_archived,
            "archive_dir": str(self.archive_dir),
        }


def create_archive_callback_from_config(
    trials_dir: str,
    archive_every_steps: int = 5,
    archive_dir: Optional[str] = None,
    compression: str = "gz",
    delete_after_archive: bool = True,
    min_trials_to_archive: int = 1,
) -> tuple[TrialPathTracker, TrialArchiveCallback]:
    """
    Factory function to create a trial tracker and archive callback.

    This is a convenience function for creating both the tracker and callback
    from configuration values, typically loaded from YAML.

    Args:
        trials_dir: Base directory containing trial subdirectories
        archive_every_steps: Archive completed trials every N steps
        archive_dir: Directory to store archives (defaults to trials_dir/archives)
        compression: Compression type ("gz", "bz2", "xz", or "")
        delete_after_archive: Whether to delete original dirs after archiving
        min_trials_to_archive: Minimum number of trials before triggering archive

    Returns:
        Tuple of (TrialPathTracker, TrialArchiveCallback)

    Example:
        ```python
        tracker, callback = create_archive_callback_from_config(
            trials_dir="/path/to/trace_jobs",
            archive_every_steps=5,
        )
        trainer.callback_handler.add_callback(callback)
        ```
    """
    tracker = TrialPathTracker(trials_dir)
    callback = TrialArchiveCallback(
        tracker=tracker,
        archive_every_steps=archive_every_steps,
        archive_dir=archive_dir,
        compression=compression,
        delete_after_archive=delete_after_archive,
        min_trials_to_archive=min_trials_to_archive,
    )
    return tracker, callback
