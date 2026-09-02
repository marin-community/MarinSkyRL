"""
Implementation of fully async training in SkyRL.

For details, see https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html.

High-level notes:
- The global_step in each training loop iteration denotes the "current step being worked on", so
`global_step - 1` is the number of steps the model has finished training.
- We do not do any cross-epoch asynchrony here, so all the control logics like
  generation workers and data buffer are initialized per-epoch. The async dataloader
  and staleness manager are also reset / validated at the end of each epoch.
"""

import asyncio
import collections
import os
import sys
from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX, LATEST_CHECKPOINT_FILE
from loguru import logger
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.progress import tqdm
from skyrl_train.utils import Timer, get_system_memory_metrics
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.utils.trainer_utils import ResumeMode, build_dataloader
from skyrl_train.utils.logging_utils import log_exception_as_text
from skyrl_train.trajectory_runners.trajectory_processing import (
    prepare_trajectory_request,
    concatenate_trajectory_batches,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from dataclasses import dataclass, field
from skyrl_train.utils.data_tracker import DataConsumptionTracker
from skyrl_train.callbacks.builtin import DataTrackingCallback, BufferCheckpointCallback
from torchdata.stateful_dataloader import StatefulDataLoader
from typing import List, Tuple, TypeVar
from enum import Enum, auto
from omegaconf import OmegaConf
from skyrl_train.callbacks import TrainerState
from skyrl_train.telemetry import (
    critical_phase,
    record_generated_work,
    record_policy_step,
    record_rollout_buffer,
    record_rollout_staleness,
)
from skyrl_train.timing_observability import publish_step_timings
from skyrl_train.async_rollout_state import GeneratedOutputGroup, GenerationBufferState
from skyrl_train.io import io
from skyrl_train.dynamic_sampling import (
    DynamicSamplingType,
    GroupSelectionPolicy,
    GroupSelectionResult,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.group_admission import AdmissionDecision, AdmissionRejection, GroupAdmissionPolicy
from skyrl_train.utils.algorithm_registry import policy_loss_requires_rollout_logprobs


_QueueItem = TypeVar("_QueueItem")


class GenerationStalledError(RuntimeError):
    """Raised when generation cannot make progress (no active producers, or dataset exhausted)."""


def _drain_queue(queue: asyncio.Queue[_QueueItem]) -> List[_QueueItem]:
    """Remove and return every item currently available without yielding."""
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return items


@dataclass
class _GenerationQueues:
    completed: asyncio.Queue[GeneratedOutputGroup]
    retries: asyncio.Queue[List[dict]]
    condition: asyncio.Condition
    active_producers: int = 0
    admitted_groups: List[GeneratedOutputGroup] = field(default_factory=list)
    admitted_groups_consumed: bool = False

    async def mark_producer_finished(self) -> None:
        """Wake admission when a generation worker permanently exits."""
        async with self.condition:
            if self.active_producers <= 0:
                raise RuntimeError("generation producer accounting underflow")
            self.active_producers -= 1
            self.condition.notify_all()

    def record_admitted(self, groups: List[GeneratedOutputGroup]) -> None:
        """Retain newly admitted groups until the step crosses its checkpoint boundary."""
        if self.admitted_groups_consumed:
            raise RuntimeError("cannot admit another group before clearing the consumed batch")
        self.admitted_groups.extend(groups)

    def mark_admitted_consumed(self) -> None:
        """Keep the trained batch available only to a final previous-checkpoint flush."""
        if not self.admitted_groups:
            raise RuntimeError("cannot consume an empty admitted batch")
        self.admitted_groups_consumed = True

    def clear_admitted(self) -> None:
        """Release the prior step's admitted groups before assembling the next step."""
        self.admitted_groups.clear()
        self.admitted_groups_consumed = False

    def snapshot(self) -> GenerationBufferState:
        """Copy queued and admitted work without yielding to another event-loop task."""
        admitted = [] if self.admitted_groups_consumed else list(self.admitted_groups)
        return self._snapshot(admitted)

    def shutdown_snapshot(self) -> GenerationBufferState:
        """Copy all work needed to recover from shutdown before the next checkpoint."""
        return self._snapshot(list(self.admitted_groups))

    def _snapshot(self, admitted_groups: List[GeneratedOutputGroup]) -> GenerationBufferState:
        completed = _drain_queue(self.completed)
        retries = _drain_queue(self.retries)
        for group in completed:
            self.completed.put_nowait(group)
        for prompts in retries:
            self.retries.put_nowait(prompts)
        return GenerationBufferState(
            completed_groups=completed,
            retry_prompts=retries,
            admitted_groups=admitted_groups,
        )


@dataclass
class _RolloutStat:
    """
    Global statistics of the trajectories used for staleness control in `_AsyncStalenessManager`.

    Note that these statistics are not per-epoch, but accumulates across all epochs.

    Attributes:
        submitted (int): The number of groups retained in submission-capacity accounting.
            Stale or cancelled attempts decrement it.
        accepted (int): The number of groups finished generation (can be either consumed by,
            or about to be consumed by the training worker). Discarded stale attempts decrement it.
        running (int): The number of groups currently being generated by the generation workers.

    For details, see https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html#async-staleness-manager
    """

    submitted: int = 0
    accepted: int = 0
    running: int = 0


class _GroupFreshness(Enum):
    FRESH = auto()
    STALE = auto()


@dataclass
class _AdmissionPartition:
    accepted_groups: List[GeneratedOutputGroup]
    rejected_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]]
    discarded_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]]


@dataclass
class _CandidateSelection:
    admitted_groups: List[GeneratedOutputGroup]
    surplus_groups: List[GeneratedOutputGroup]
    discarded_reasons: collections.Counter[str]
    candidate_count: int


class _AsyncStalenessManager:
    """
    A controller that manages the capacity of the generation workers based on staleness control.

    The goal is to never submit more trajectories to the generation workers than the training worker
    can consume, so that the trajectories are not too stale (relative to max_staleness_steps).
    This is enforced via a capacity rule, not a hard **per-group** staleness guarantee: we bound
    the **aggregate** number of groups that can be ahead of training so that, in **steady state**,
    staleness remains within the configured budget of `max_staleness_steps`.

    In pathological cases (e.g., very long-running trajectories), an individual group may take
    more than `max_staleness_steps` of training steps to finish. Such attempts are discarded and
    regenerated from the same source prompt before training proceeds.

    The key capacity formula is implemented in `_compute_capacity_unlocked`. For details and caveats,
    see https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html#async-staleness-manager.

    Reference:
    - Modeled after AReal's StalenessManager: https://github.com/inclusionAI/AReaL/blob/b755c4447c2fff97889d8828293ee85f17a806f9/areal/core/staleness_manager.py
    - The idea of this controller is from section 5.1 of AReal's paper: https://arxiv.org/pdf/2505.24298v3
    """

    def __init__(self, max_concurrent_generation_groups: int, mini_batch_size: int, max_staleness_steps: int):
        self.max_concurrent_generation_groups = max_concurrent_generation_groups
        self.mini_batch_size = mini_batch_size
        self.max_staleness_steps = max_staleness_steps

        # Control logics.
        self._stat = _RolloutStat()
        self._cond = asyncio.Condition()

        # The current version that is being worked on, i.e. FullyAsyncRayPPOTrainer.global_step.
        # `self._current_global_step - 1` is the number of steps the model has finished training.
        self._current_global_step = 1

    def load_state_from_checkpoint(self, global_step: int) -> None:
        """
        Load the state from a checkpoint.
        """
        self._current_global_step = global_step
        # trainer has already consumed (and hence submitted) this many trajectories.
        self._stat.accepted = (global_step - 1) * self.mini_batch_size
        self._stat.submitted = self._stat.accepted

    async def validate_state_at_epoch_end(self, global_step: int) -> None:
        """
        Check that the current version and accepted rollouts are consistent with the global step.

        Args:
            global_step: The global step we are about to train on (after incrementing).
        """
        async with self._cond:
            assert self._stat.running == 0, "We expect no rollouts are running at end of an epoch."
            assert self._stat.submitted == self._stat.accepted, (
                "We expect all submitted rollouts to be accepted at end of an epoch."
            )
            consumed = (global_step - 1) * self.mini_batch_size
            assert self._stat.accepted == consumed, (
                f"Unexpected number of accepted rollouts. Got {self._stat.accepted} != {consumed}."
            )
            assert self._current_global_step == global_step, (
                f"Unexpected current version. Got {self._current_global_step} != {global_step}."
            )
            assert self._stat.submitted == self._stat.accepted, (
                "We expect all submitted rollouts to be accepted at end of an epoch. "
                f"Got {self._stat.submitted} != {self._stat.accepted}."
            )

    def _compute_capacity_unlocked(self) -> int:
        # NOTE(Charlie): do not need a self._current_global_step + 1 here unlike AReal because our
        # `_current_global_step` is "the version being worked on", not already finished steps.
        consumer_capacity = (self.max_staleness_steps + self._current_global_step) * self.mini_batch_size
        producer_staleness_capacity = consumer_capacity - (self._stat.accepted + self._stat.running)
        producer_concurrency_capacity = self.max_concurrent_generation_groups - self._stat.running
        return min(producer_concurrency_capacity, producer_staleness_capacity)

    async def acquire_submission_slot(self) -> None:
        """Block until generation capacity is available, then reserve a slot.

        Individual long-running generations can still exceed the aggregate staleness
        budget; those attempts are regenerated before training.
        """
        async with self._cond:
            while self._compute_capacity_unlocked() <= 0:
                await self._cond.wait()
            self._stat.submitted += 1
            self._stat.running += 1

    async def on_rollout_accepted(self) -> None:
        async with self._cond:
            self._stat.accepted += 1
            self._stat.running -= 1
            self._cond.notify_all()

    async def on_rollouts_discarded(self, count: int) -> None:
        """Remove completed rejected attempts from capacity accounting."""
        async with self._cond:
            self._stat.accepted -= count
            self._stat.submitted -= count
            self._cond.notify_all()

    async def cancel_submission_slot(self) -> None:
        """Release a reserved slot without accepting a completed group."""
        async with self._cond:
            self._stat.submitted -= 1
            self._stat.running -= 1
            self._cond.notify_all()

    async def notify_capacity_change(self, new_global_step: int) -> None:
        # Called when current_global_step changes (e.g., after a training step)
        async with self._cond:
            self._current_global_step = int(new_global_step)
            self._cond.notify_all()


class _AsyncDataloader:
    """
    A train dataloader wrapper that accommodates the need of fully async training, including:
    - Thread-safe dataloader iteration with a lock, as there are multiple parallel generation workers polling data.
    - Skip-on-resume: uses DataConsumptionTracker's epoch-scoped UIDs to skip already-consumed data.
    - Finite iteration for ordinary async training and replacement passes for DAPO filter sampling.
    - For finite iteration, set the effective length to a multiple of the mini-batch size because the underlying
      fully async dataloader has batch size 1 and cannot use `drop_last` for the training mini-batch.

    Data consumption tracking (UID recording, epoch transitions, checkpoint persistence)
    is handled by DataConsumptionTracker + DataTrackingCallback, not by this class.
    """

    def __init__(
        self,
        train_dataloader: StatefulDataLoader,
        mini_batch_size: int,
        data_tracker: DataConsumptionTracker,
        dynamic_sampling_type: DynamicSamplingType | None = None,
    ):
        self._train_dataloader = train_dataloader
        self._train_dataloader_initial_state = train_dataloader.state_dict()
        self._sample_with_replacement = dynamic_sampling_type is DynamicSamplingType.FILTER
        self._effective_dataloader_length = (
            len(self._train_dataloader)
            if self._sample_with_replacement
            else len(self._train_dataloader) // mini_batch_size * mini_batch_size
        )
        self._iter = enumerate(self._train_dataloader)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._data_tracker = data_tracker
        self._pending_uids: set[str] = set()
        self._exhausted: bool = False
        self._eligible_rows_returned_in_pass = 0

    def reserve_pending_uids(self, uids: set[str]) -> None:
        """Exclude checkpointed pending work from restarted dataset iteration."""
        self._pending_uids.update(uids)

    def load_state_from_checkpoint(self) -> None:
        """
        Reset dataloader iteration state for resume.

        On resume, the DataConsumptionTracker already has the consumed UIDs loaded.
        We just need to reset the dataloader to the beginning so
        get_next_non_consumed_data() can skip already-consumed items.
        """
        # Reset in case the dataloader loaded the state from the checkpoint, which we do not want.
        self._train_dataloader.load_state_dict(self._train_dataloader_initial_state)
        # Re-create the iterator so get_next_non_consumed_data() starts from
        # the beginning of the dataset (not the exhausted iterator from __init__).
        self._iter = enumerate(self._train_dataloader)
        self._exhausted = False
        self._eligible_rows_returned_in_pass = 0

    async def reset_at_epoch_end(self) -> None:
        """Reset dataloader iterator for the next epoch.

        Note: epoch-scoped UID clearing is handled by DataTrackingCallback.on_epoch_end_async,
        which fires AFTER any checkpoint save — eliminating the race condition where UIDs
        were cleared before the checkpoint captured them.
        """
        async with self._lock:
            self._train_dataloader.load_state_dict(self._train_dataloader_initial_state)  # reset to initial state
            self._iter = enumerate(self._train_dataloader)
            self._pending_uids.clear()
            self._exhausted = False
            self._eligible_rows_returned_in_pass = 0

    async def get_next_non_consumed_data(self):
        """
        Return the next batch of training data.

        If we loaded from a checkpoint, it skips already-consumed data. DAPO filter sampling starts a new source
        pass after exhaustion so rejected prompts can be generated again. Returns None when finite sampling ends or
        when a complete filter pass contains no eligible UID.
        """
        assert self._iter is not None and self._lock is not None, "Dataloader not initialized; call reset() first"
        async with self._lock:
            # Read consumed UIDs after acquiring the iterator lock. Replacement passes can overlap a training step.
            skip_set = self._data_tracker.get_consumed_uids_in_epoch() | self._pending_uids
            while True:
                try:
                    # Keep polling until we get a non-consumed data or the dataloader is exhausted.
                    while True:
                        iter_idx, rand_prompts = next(self._iter)
                        if iter_idx >= self._effective_dataloader_length:
                            raise StopIteration
                        uid = rand_prompts[0]["uid"]
                        if uid not in skip_set:
                            self._eligible_rows_returned_in_pass += 1
                            return rand_prompts
                except StopIteration:
                    if self._sample_with_replacement and self._eligible_rows_returned_in_pass:
                        self._iter = enumerate(self._train_dataloader)
                        self._eligible_rows_returned_in_pass = 0
                        continue
                    self._exhausted = True
                    return None


class FullyAsyncRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        # Extract cfg before base init so we can initialize async-specific knobs used by our overrides.
        cfg = kwargs.get("cfg", args[0] if len(args) > 0 else None)
        assert cfg is not None, "cfg must be provided to FullyAsyncRayPPOTrainer"

        # Initialize async-specific knobs
        self.num_parallel_generation_workers = cfg.trainer.fully_async.num_parallel_generation_workers
        self.mini_batch_size = cfg.trainer.policy_mini_batch_size
        self.max_staleness_steps = cfg.trainer.fully_async.max_staleness_steps
        self.admission_stall_timeout = int(cfg.trainer.fully_async.admission_stall_timeout)
        if self.admission_stall_timeout <= 0:
            raise ValueError("trainer.fully_async.admission_stall_timeout must be positive")
        self._group_selection_policy = GroupSelectionPolicy.for_fully_async(
            cfg.trainer.algorithm.dynamic_sampling.type,
            criteria=resolve_dynamic_sampling_criteria(
                cfg.trainer.algorithm.dynamic_sampling.informative_on,
                float(cfg.trainer.algorithm.dynamic_sampling.min_reward_std),
            ),
        )
        self._dynamic_sampling_type = self._group_selection_policy.sampling_type
        max_sample_batches = int(cfg.trainer.algorithm.dynamic_sampling.max_sample_batches)
        self._dynamic_sampling_max_sample_batches = max_sample_batches
        self._dynamic_sampling_max_candidate_groups = (
            max_sample_batches * int(cfg.trainer.train_batch_size) if max_sample_batches > 0 else None
        )

        # Completed-but-unconsumed generation-buffer cap (head-node memory bound).
        #
        # WHY THIS KNOB (2026-07-10, 80B head-plasma/RAM overflow root-cause): the
        # per-epoch buffer below is `asyncio.Queue(maxsize=num_parallel_generation_workers)`.
        # Each buffered `GeneratedOutputGroup` holds a full `TrajectoryBatch` whose
        # `rollout_routed_experts` (R3) capture is O(response_len · num_moe_layers ·
        # top_k) per token — for Qwen3-Next-80B (L=48, K=10) that is ~15 MiB/sequence,
        # ~126 MiB per 8-sample group. With `num_parallel_generation_workers=900` the
        # buffer alone can pin ~113 GiB of head-node memory (the pre-existing occupancy
        # that starves the gs1 forward-chunk `ray.put`s — see
        # agent_logs/2026-07-09_80b_v5_98k_nccl_wedge_kill.md and 48593f42). The buffer
        # depth is NOT a throughput lever here: generation concurrency is capped by the
        # inference engines' working set (num_inference_engines · max_num_seqs /
        # n_samples_per_prompt), NOT by the worker count, so a deep buffer only lets a
        # rollout backlog accumulate. This knob bounds the completed-output backlog
        # without reducing the number of worker loops available for generation.
        #
        # Default None => maxsize == num_parallel_generation_workers, i.e. BYTE-IDENTICAL
        # to today's behavior (no config change => no behavior change). Set it to a small
        # multiple of the mini-batch (e.g. mini_batch_size · (max_staleness_steps + 1), or
        # a fixed 128) to cap the footprint to O(1) in async depth. NOTE: when this is set
        # below num_parallel_generation_workers, up to (num_parallel_generation_workers -
        # cap) workers may wait on the shared queue condition while each still holds ONE
        # completed group, so to fully bound the head-node footprint you should ALSO lower
        # num_parallel_generation_workers toward the engine working set.
        self.max_buffered_groups = (
            OmegaConf.select(cfg, "trainer.fully_async.max_buffered_groups", default=None)
            or self.num_parallel_generation_workers
        )

        assert (
            # otherwise wasted throughput
            self.mini_batch_size <= self.num_parallel_generation_workers
        ), (
            "Invalid num_parallel_generation_workers, must be >= mini_batch_size. Got: "
            f"{self.mini_batch_size=}, {self.num_parallel_generation_workers=}"
        )
        # Initialize base trainer
        super().__init__(*args, **kwargs)
        self._group_admission_policy = GroupAdmissionPolicy(
            self.group_advantage_invariant,
            max_staleness_steps=self.max_staleness_steps,
            rollout_logprobs_required=policy_loss_requires_rollout_logprobs(
                self.cfg.trainer.algorithm.policy_loss_type
            ),
        )
        # Some async-specific validations
        assert self.cfg.trainer.train_batch_size == self.cfg.trainer.policy_mini_batch_size, (
            "train_batch_size must equal policy_mini_batch_size for fully async training"
        )
        assert not self.cfg.generator.batched, "batched is not supported for fully async training."
        assert self.cfg.generator.async_engine, "async_engine must be True for fully async training."
        # TODO(Charlie): we can support it, just multi-turn partial rollout but synchronous.
        assert not self.colocate_all, "colocate_all is not supported for async training yet."

        # TODO(Charlie): need to assert we are doing TIS and returning logprobs

        # Async-specific states
        self.data_tracker = DataConsumptionTracker(
            mini_batch_size=self.mini_batch_size,
            num_steps_per_epoch=self.num_steps_per_epoch,
        )
        self.async_train_dataloader = _AsyncDataloader(
            self.train_dataloader,
            self.mini_batch_size,
            self.data_tracker,
            self._dynamic_sampling_type,
        )
        # Register the data tracking callback for checkpoint persistence and epoch transitions
        self.callback_handler.add_callback(DataTrackingCallback(self.data_tracker))
        # Register buffer checkpoint callback for saving/restoring generation buffer on resume
        self._buffer_checkpoint_callback = BufferCheckpointCallback()
        self.callback_handler.add_callback(self._buffer_checkpoint_callback)
        self._pending_buffer_restore_path = None
        self._staleness_manager = _AsyncStalenessManager(
            max_concurrent_generation_groups=self.num_parallel_generation_workers,
            mini_batch_size=self.mini_batch_size,
            max_staleness_steps=self.max_staleness_steps,
        )
        # Tracked at instance level so the finally block in train() can cancel
        # them even when an exception skips the per-epoch epilogue.
        self._active_trajectory_tasks: List[asyncio.Task] = []
        self._groups_rejected_since_step = 0
        self._rejection_reasons_since_step: collections.Counter[str] = collections.Counter()
        self._groups_inspected_since_step = 0
        self._step_time_history: collections.deque[float] = collections.deque(maxlen=5)

    def _configure_training_schedule(self):
        """
        Overrides to build dataloader for fully async training. See `_AsyncDataloader` for more details.
        """
        self.train_dataloader = build_dataloader(self.cfg, self.train_dataset, is_train=True, is_fully_async=True)
        self.num_steps_per_epoch = len(self.train_dataloader) // self.mini_batch_size
        self.total_training_steps = self.num_steps_per_epoch * self.cfg.trainer.epochs
        max_steps = getattr(self.cfg.trainer, "max_steps", None)
        if max_steps is not None and max_steps > 0:
            self.total_training_steps = min(self.total_training_steps, max_steps)
        logger.info(f"Length of train_dataloader: {len(self.train_dataloader)}")
        logger.info(f"Number of steps per epoch: {self.num_steps_per_epoch}")
        logger.info(f"Total training steps: {self.total_training_steps}")

    def _num_steps_per_epoch(self) -> int:
        """Account for fully async mini-batch accumulation."""
        return self.num_steps_per_epoch

    def _cancel_trajectory_tasks(self) -> None:
        """Cancel any active generation tasks left over from an abnormal exit.

        Normally the per-epoch epilogue cancels these, but if an exception
        breaks out of the inner training loop the epilogue is skipped.
        """
        tasks = self._active_trajectory_tasks
        if not tasks:
            return
        n_running = sum(1 for t in tasks if not t.done())
        if n_running:
            logger.warning(f"Cancelling {n_running} orphaned generation tasks from abnormal train loop exit")
            for t in tasks:
                t.cancel()
        self._active_trajectory_tasks = []

    def _restore_buffer_from_checkpoint(self, queues: _GenerationQueues, checkpoint_path: str) -> None:
        """Restore completed, admitted, and retryable rollout work from a checkpoint."""
        buffer_state = BufferCheckpointCallback.load_buffer_state(checkpoint_path)
        if len(buffer_state.completed_groups) > queues.completed.maxsize:
            raise ValueError(
                f"Checkpoint contains {len(buffer_state.completed_groups)} completed groups, exceeding buffer capacity "
                f"{queues.completed.maxsize}"
            )
        if len(buffer_state.admitted_groups) > self.mini_batch_size:
            raise ValueError(
                f"Checkpoint contains {len(buffer_state.admitted_groups)} admitted groups, exceeding mini-batch size "
                f"{self.mini_batch_size}"
            )
        self.async_train_dataloader.reserve_pending_uids(buffer_state.pending_uids())
        for item in buffer_state.completed_groups:
            queues.completed.put_nowait(item)
        for prompts in buffer_state.retry_prompts:
            queues.retries.put_nowait(prompts)
        queues.record_admitted(buffer_state.admitted_groups)
        restored_group_count = len(buffer_state.completed_groups) + len(buffer_state.admitted_groups)
        self._staleness_manager._stat.accepted += restored_group_count
        self._staleness_manager._stat.submitted += restored_group_count
        logger.info(
            f"Restored {len(buffer_state.completed_groups)} completed, "
            f"{len(buffer_state.admitted_groups)} admitted generation groups, and "
            f"{len(buffer_state.retry_prompts)} pending retries "
            "from checkpoint"
        )

    def _latest_checkpoint_step(self) -> int | None:
        marker = os.path.join(self.cfg.trainer.ckpt_path, LATEST_CHECKPOINT_FILE)
        if not io.exists(marker):
            return None
        with io.open_file(marker, "r") as f:
            return int(f.read().strip())

    async def _flush_generation_buffer_on_shutdown(self) -> None:
        """Attach volatile buffer state to the immediately preceding model checkpoint."""
        callback = getattr(self, "_buffer_checkpoint_callback", None)
        if callback is None or not callback.has_bound_queues():
            return
        target_step = self.global_step - 1
        latest_step = await asyncio.to_thread(self._latest_checkpoint_step)
        if latest_step != target_step:
            if callback.has_shutdown_state():
                logger.warning(
                    "Skipping final generation-buffer flush: latest checkpoint step {} is not the immediately "
                    "preceding step {}",
                    latest_step,
                    target_step,
                )
            return
        checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{target_step}")
        await callback.flush_to_checkpoint(checkpoint_path)

    async def shutdown(self) -> None:
        """Bank the compatible async buffer before releasing trainer resources."""
        if getattr(self, "_shutdown_complete", False):
            return
        try:
            await self._flush_generation_buffer_on_shutdown()
        finally:
            await super().shutdown()

    async def train(self):
        """
        Main fully async training loop for PPO
        """
        self.global_step = 0

        try:
            await self._startup_trajectory_runner()
            await self._train_loop()
        except Exception as e:
            log_exception_as_text(f"Train loop failed at global_step {self.global_step}", e)
            raise
        finally:
            # Cancel any orphaned generation tasks that survived an early exit
            # (the per-epoch epilogue only runs on normal loop completion).
            self._cancel_trajectory_tasks()

            await self.shutdown()

    async def _train_loop(self):
        """
        Internal training loop, separated for proper trajectory-runner lifecycle management.
        """
        # Load checkpoint state if resumption is enabled.
        # Data consumption state is loaded via DataTrackingCallback.load_from_checkpoint()
        # into self.data_tracker, which the async dataloader reads for skip-on-resume.
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints", self.all_startup_timings):
                self.global_step, checkpoint_path = self.load_checkpoints()
                logger.info(f"Resumed training from global_step {self.global_step}")

                if self.global_step > 0:
                    # Load data consumption state into the tracker
                    loaded = DataTrackingCallback.load_from_checkpoint(checkpoint_path, self.data_tracker)
                    if not loaded:
                        logger.warning(
                            "No data consumption state found in checkpoint — "
                            "resume may re-train on already-consumed data"
                        )

                    # Store checkpoint path for buffer restore after queue creation
                    self._pending_buffer_restore_path = checkpoint_path

                    # Reset dataloader iteration for skip-on-resume
                    self.async_train_dataloader.load_state_from_checkpoint()
                    self._staleness_manager.load_state_from_checkpoint(
                        self.global_step + 1
                    )  # +1 due to we haven't incremented yet

                    # Soft validation: log mismatch instead of crashing
                    steps_into_epoch = self.global_step % self.num_steps_per_epoch
                    if steps_into_epoch != 0:
                        expected_consumed_in_epoch = self.mini_batch_size * steps_into_epoch
                        actual_consumed_in_epoch = self.data_tracker.consumed_in_epoch_count
                        if actual_consumed_in_epoch != expected_consumed_in_epoch:
                            logger.warning(
                                f"Data consumption count mismatch on resume: "
                                f"expected {expected_consumed_in_epoch}, got {actual_consumed_in_epoch}. "
                                f"This can happen after epoch boundary transitions or error recovery."
                            )

        # Initialize weight sync state
        with Timer("init_weight_sync_state", self.all_startup_timings):
            self.init_weight_sync_state()

        # sync weights to inference engines
        with Timer("sync_weights_to_inference_engines", self.all_startup_timings) as weight_update_timer:
            await self.async_sync_policy_weights_to_inference_engines()
            # Drain the policy workers' event loops to a hard sync point so every FSDP
            # shard rank is free before the step-1 forward is dispatched (the MoE-RL
            # async-dispatch wedge fix). See _drain_policy_event_loops.
            await self._drain_policy_event_loops()
        self._log_weight_update_completed(
            reason="initial",
            duration_seconds=weight_update_timer.duration,
        )

        # Synchronize before checking completion so a requested final evaluation uses
        # the checkpoint weights. The loaded global_step is the completed step count;
        # >= treats a resume exactly at max_steps as complete without running gs N+1.
        if self.resume_mode != ResumeMode.NONE and self.global_step >= self.total_training_steps:
            await self._handle_resume_at_max_steps()
            return

        self._log_startup_timings()

        # Create initial trainer state for on_train_begin callback
        start_epoch = self.global_step // self.num_steps_per_epoch
        initial_state = self._create_trainer_state(epoch=start_epoch)

        # Call on_train_begin callbacks (handles eval_before_train via EvaluationCallback)
        self._control.reset()
        self._control = await self.callback_handler.call_event_async(
            "on_train_begin", initial_state, self._control, trainer=self
        )

        # Handle pre-training evaluation if requested by callbacks
        if self._control.should_evaluate and self.eval_dataset is not None:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self._log_metrics_stdout(eval_metrics, step=self.global_step, kind="eval")
                self.tracker.log(eval_metrics, step=self.global_step, commit=self.cfg.trainer.tracker_commit_each_step)
            self._control.should_evaluate = False

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Step Progress")
        start_epoch = self.global_step // self.num_steps_per_epoch
        last_completed_step = self.global_step
        self.global_step += 1  # start training at global_step 1
        for epoch in range(start_epoch, self.cfg.trainer.epochs):
            # 0. Per-epoch prologue. Note that we do not do any cross-epoch asynchrony here.

            # Buffer of completed generation. Cap defaults to num_parallel_generation_workers
            # (byte-identical to prior behavior) but can be bounded independently via
            # trainer.fully_async.max_buffered_groups to cap head-node memory — see
            # self.max_buffered_groups in __init__.
            generation_queues = _GenerationQueues(
                completed=asyncio.Queue(maxsize=self.max_buffered_groups),
                retries=asyncio.Queue(),
                condition=asyncio.Condition(),
                active_producers=self.num_parallel_generation_workers,
            )

            self._buffer_checkpoint_callback.bind_queues(generation_queues)

            # Restore buffer from checkpoint if resuming
            if self._pending_buffer_restore_path is not None:
                self._restore_buffer_from_checkpoint(
                    generation_queues,
                    self._pending_buffer_restore_path,
                )
                self._pending_buffer_restore_path = None

            # Provide the runner with a live reference to global_step so it can
            # capture the step at first vLLM inference (for accurate staleness tracking).
            self.trajectory_runner.global_step_fn = lambda: self.global_step

            # Maintain self.num_parallel_generation_workers concurrent group-generation workers.
            # Stored on self so the finally block in train() can cancel them on abnormal exit.
            self._active_trajectory_tasks = [
                asyncio.create_task(self._run_generate_for_a_group_loop(generation_queues))
                for _ in range(self.num_parallel_generation_workers)
            ]
            trajectory_tasks = self._active_trajectory_tasks

            for _ in range(self.global_step, (1 + epoch) * self.num_steps_per_epoch + 1):
                with Timer("step", self.all_timings) as step_timer:
                    # 1. Discard every completed stale attempt and wait for a full fresh batch.
                    logger.info(
                        "Rollout batch started: step={} mode=fully_async required_groups={}",
                        self.global_step,
                        self.mini_batch_size,
                    )
                    with (
                        Timer("wait_for_generation_buffer", self.all_timings) as rollout_wait_timer,
                        critical_phase("rollout_or_inference_wait", self.global_step),
                    ):
                        cur_generation_group_mini_batch = await self._get_admitted_generation_group_mini_batch(
                            generation_queues,
                        )

                    # 2. Post-process the complete generated mini-batch and convert it to training format.
                    with Timer("convert_to_training_input", self.all_timings):
                        training_input = await asyncio.to_thread(
                            self.convert_generation_group_mini_batch_to_training_input,
                            cur_generation_group_mini_batch,
                        )
                    response_ids = [
                        response_ids
                        for group in cur_generation_group_mini_batch
                        for response_ids in group.trajectory_batch["response_ids"]
                    ]
                    logger.info(
                        "Rollout batch completed: step={} mode=fully_async groups={} trajectories={} "
                        "response_tokens={} staleness_mean={:.3f} staleness_max={} duration_seconds={:.3f}",
                        self.global_step,
                        len(cur_generation_group_mini_batch),
                        len(response_ids),
                        sum(len(response) for response in response_ids),
                        self.all_metrics["async/staleness_mean"],
                        self.all_metrics["async/staleness_max"],
                        rollout_wait_timer.duration,
                    )

                    # TIS graceful-degrade observability (Fix A): record whether THIS
                    # training batch was missing all rollout logprobs (-> TIS skipped,
                    # standard policy loss used) and a running skipped-fraction. Driver-side
                    # metric only (NOT part of the worker per-key all_reduce(status)), so it
                    # cannot cause the keyset-mismatch NCCL deadlock. skipped_fraction ~1.0
                    # over the run = rollout-logprob capture systematically broken (R3<->logprob
                    # interaction); low = intermittent context-length errors.
                    if self.cfg.trainer.algorithm.use_tis:
                        batch_skipped = float(getattr(self, "_tis_batch_skipped_no_logprobs", 0.0))
                        self._tis_skipped_count = getattr(self, "_tis_skipped_count", 0.0) + batch_skipped
                        self._tis_total_count = getattr(self, "_tis_total_count", 0.0) + 1.0
                        self.all_metrics.update(
                            {
                                "tis/batch_skipped_no_logprobs": batch_skipped,
                                "tis/skipped_fraction": self._tis_skipped_count / self._tis_total_count,
                            }
                        )

                    # 3. Run training and record consumed UIDs in the tracker.
                    with Timer("run_training", self.all_timings):
                        status = await self._run_training(training_input)
                    train_duration = self.all_timings["train_critic_and_policy"]
                    self._log_optimizer_step_completed(
                        epoch=epoch,
                        training_input=training_input,
                        duration_seconds=train_duration,
                    )
                    await self.data_tracker.mark_consumed([g.uid for g in cur_generation_group_mini_batch])
                    generation_queues.mark_admitted_consumed()

                    # 4. After training: sync weights to the inference engines.
                    #    The inference engines are a SHARED HTTP backend that every
                    #    RolloutCoordinator calls, so the STOCK engine-level
                    #    pause/sync/resume below (fast NCCL broadcast with the
                    #    engines briefly quiesced) already propagates fresh weights
                    #    to every coordinator's subsequent requests. We deliberately
                    #    do NOT barrier-pause/drain the RolloutCoordinators at the
                    #    trial level: a coordinator-level drain is unnecessary for
                    #    correctness and defeats async overlap (the hard-drain stalled
                    #    the step boundary indefinitely when long-running trials never
                    #    drained). Rollouts in flight across the weight swap simply
                    #    return as STALE and are bounded by the dispatcher's existing
                    #    max_staleness_steps accounting — exactly like stock
                    #    fully_async, which never drains trial orchestration. This
                    #    block is now byte-identical for fan-out ON and OFF.
                    with Timer("sync_weights", self.all_timings) as weight_update_timer:
                        await self.inference_engine_client.pause_generation()
                        await self.async_sync_policy_weights_to_inference_engines()
                        # Drain the policy workers' event loops to a hard sync point so
                        # every FSDP shard rank is free before the NEXT step's forward is
                        # dispatched (the MoE-RL async-dispatch wedge fix). See
                        # _drain_policy_event_loops.
                        await self._drain_policy_event_loops()
                        await self.inference_engine_client.resume_generation()
                    self._log_weight_update_completed(
                        reason="training_step",
                        duration_seconds=weight_update_timer.duration,
                    )

                # 5. Log status and update metrics
                logger.info(status)
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})

                # 6. Create trainer state and call on_step_end callbacks
                is_epoch_end = _ == (1 + epoch) * self.num_steps_per_epoch
                is_last_step = self.global_step == self.total_training_steps
                step_state = TrainerState(
                    global_step=self.global_step,
                    epoch=epoch,
                    total_steps=self.total_training_steps,
                    num_steps_per_epoch=self.num_steps_per_epoch,
                    is_last_step=is_last_step,
                    is_epoch_end=is_epoch_end,
                    metrics=dict(self.all_metrics),
                    timings=dict(self.all_timings),
                )

                self._control.reset()
                self._control = await self.callback_handler.call_event_async(
                    "on_step_end", step_state, self._control, trainer=self
                )

                # 7. Handle callback control signals

                # Handle checkpoint saving
                if self._control.should_save:
                    await self._save_intermediate_checkpoint(step_state)
                    self._control.should_save = False

                # Handle HF model saving
                if self._control.should_save_hf_model:
                    await asyncio.to_thread(self.handle_hf_export)
                    self._control.should_save_hf_model = False

                # Handle evaluation
                if self._control.should_evaluate and self.eval_dataset is not None:
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)
                    await self.callback_handler.call_event_async(
                        "on_evaluate", step_state, self._control, metrics=eval_metrics, trainer=self
                    )
                    self._control.should_evaluate = False

                # 8. Log metrics
                if self._control.should_log:
                    log_payload = {
                        **self.all_metrics,
                        **{f"timing/{k}": v for k, v in self.all_timings.items()},
                        **get_system_memory_metrics(),
                    }
                    self._log_metrics_stdout(log_payload, step=self.global_step, kind="train")
                    self.tracker.log(
                        log_payload, step=self.global_step, commit=self.cfg.trainer.tracker_commit_each_step
                    )
                    await self.callback_handler.call_event_async(
                        "on_log", step_state, self._control, logs=log_payload, trainer=self
                    )

                self._log_training_step_completed(
                    epoch=epoch,
                    duration_seconds=step_timer.duration,
                )

                self.all_metrics = {}
                step_duration = self.all_timings.get("step")
                if step_duration is not None:
                    self._step_time_history.append(step_duration)
                publish_step_timings(self.all_timings, self.global_step)
                self.all_timings = {}
                pbar.update(1)

                last_completed_step = self.global_step
                record_policy_step(self.global_step)
                self.global_step += 1
                generation_queues.clear_admitted()

                # 9. Notify generation workers that the capacity has increased, unblocking them.
                await self._staleness_manager.notify_capacity_change(self.global_step)

                # 10. Check for max_steps
                if self.global_step > self.total_training_steps:
                    logger.info(f"Reached max training steps ({self.total_training_steps})")
                    break

                # 11. Check for early stopping
                if self._control.should_training_stop:
                    logger.info("Training stopped early by callback")
                    break

            # 12. Per-epoch epilogue.
            # Call on_epoch_end callbacks
            epoch_state = self._create_trainer_state(epoch=epoch)
            self._control.reset()
            self._control = await self.callback_handler.call_event_async(
                "on_epoch_end", epoch_state, self._control, trainer=self
            )

            # Handle ref model update at epoch end (via RefModelUpdateCallback or direct config)
            ref_callback = self._get_ref_update_callback()
            if self.ref_model is not None and ref_callback is not None and ref_callback.should_update_ref:
                with Timer("update_ref_with_policy", self.all_timings):
                    await asyncio.to_thread(self.update_ref_with_policy)

            # Cancel generation tasks for this epoch
            for t in trajectory_tasks:
                t.cancel()
            try:
                await asyncio.gather(*trajectory_tasks, return_exceptions=True)
            except Exception:
                pass
            self._active_trajectory_tasks = []

            # Per-epoch reset/validation for data loading and staleness management
            assert all(t.done() for t in trajectory_tasks), (
                "Trajectory runner tasks must be done before resetting the dataloader manager and validating the staleness manager."
            )
            # Drain any generation outputs that arrived after the training loop
            # stopped consuming (race between producer enqueue and consumer exit).
            n_drained = len(_drain_queue(generation_queues.completed))
            assert generation_queues.retries.empty(), (
                f"Epoch ended with {generation_queues.retries.qsize()} stale-group retries still pending"
            )
            if n_drained > 0:
                logger.warning(
                    f"Drained {n_drained} unconsumed generation output(s) at epoch boundary "
                    f"(global_step={self.global_step}). These were generated with stale weights "
                    f"and arrived after the training loop stopped consuming."
                )
                # Reconcile staleness counters for drained items.  Items in the
                # buffer may have been fully accepted (on_rollout_accepted ran)
                # or orphaned (put_nowait succeeded but on_rollout_accepted was
                # interrupted by CancelledError — those are already handled by
                # the CancelledError fix above).  Only undo accepted/submitted
                # for properly-accepted items that were never consumed.
                consumed = (self.global_step - 1) * self.mini_batch_size
                n_accepted_surplus = self._staleness_manager._stat.accepted - consumed
                if n_accepted_surplus > 0:
                    self._staleness_manager._stat.accepted -= n_accepted_surplus
                    self._staleness_manager._stat.submitted -= n_accepted_surplus
            await self.async_train_dataloader.reset_at_epoch_end()
            await self._staleness_manager.validate_state_at_epoch_end(self.global_step)

            if self.global_step > self.total_training_steps:
                break

            if self._control.should_training_stop:
                logger.info("Training stopped early by callback at epoch end")
                break

            # End of an epoch.

        # End of training
        pbar.close()

        await self._finalize_training(
            completed_step=last_completed_step,
            epoch=self.cfg.trainer.epochs - 1,
        )
        logger.info("Training done!")

    async def _run_training(self, training_input: TrainingInputBatch):
        # TODO(Charlie): share this code with the one-step-off async trainer.
        # Drain the policy workers' event loops to a hard sync point IMMEDIATELY
        # before dispatching this step's forward (the MoE-RL async-dispatch wedge
        # fix, step-1 completion 2026-06-29). The existing post-weight-sync drains
        # (_train_loop L603 after the INITIAL sync, L739 after each per-step sync)
        # leave a hazard at STEP 1 specifically: the L603 drain fires at job start,
        # then `wait_for_generation_buffer` blocks for HOURS (the fully-async rollout
        # fill — 8813s / 2.45h on rl-131k-30b-drainfix) before this first forward is
        # dispatched. By then the L603 barrier is stale: the policy async actors'
        # single event-loop thread has serviced other dispatched coroutines in the
        # interim, so when the SYNC `forward.remote()` finally arrives only rank 0's
        # loop is free and runs it (into a lonely mesh_fsdp param-unshard all-gather
        # — FR-proven: ONLY rank 0 logged WORKER_FORWARD_ENTER at step 1, ranks
        # 8/16/24 never scheduled their `forward` task → the embed-unshard
        # `_all_gather_base` on mesh_fsdp deadlocks, NCCL watchdog aborts at 1800s,
        # global_step never reaches 1). The L739 drain already makes this barrier
        # ADJACENT to the forward for steps 2+, which is why only step 1 wedged;
        # doing it here makes the drain adjacent for EVERY step (step 1 included)
        # regardless of how stale the preceding post-weight-sync drain is. Symmetric
        # on every rank, changes no tensor values (correctness-neutral), strict no-op
        # for single-rank / uninitialized runs. Idempotent with the L739 drain (a
        # second pass-through barrier on an already-free loop is a cheap no-op).
        await self._drain_policy_event_loops()
        # inference and calculate values, log probs, rewards, kl divergence
        with Timer("fwd_logprobs_values_reward", self.all_timings):
            training_input = await asyncio.to_thread(self.fwd_logprobs_values_reward, training_input)

        # calculate kl divergence and create experiences
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            with Timer("apply_reward_kl_penalty", self.all_timings):
                training_input = self.apply_reward_kl_penalty(training_input)

        # calculate advantages and returns / along with tensorboard logging
        with Timer("compute_advantages_and_returns", self.all_timings):
            training_input = self.compute_advantages_and_returns(training_input)
            training_input = self.finalize_advantages_for_training(training_input)

        if self.cfg.trainer.dump_data_batch:
            # dump data to file
            with Timer("dump_data_batch", self.all_timings):
                self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

        # train policy/critic model
        with Timer("train_critic_and_policy", self.all_timings), critical_phase("train_step", self.global_step):
            status = await asyncio.to_thread(self.train_critic_and_policy, training_input)

        return status

    async def _run_generate_for_a_group_loop(self, queues: _GenerationQueues):
        """Generate dataset rows or retries and route only fresh groups to the completed queue."""
        try:
            while True:
                slot_acquired = False
                rand_prompts = await self._next_generation_prompts(queues)
                await self._staleness_manager.acquire_submission_slot()
                slot_acquired = True
                assert len(rand_prompts) == 1
                trajectory_request, uids = prepare_trajectory_request(
                    rand_prompts,
                    self.cfg.generator.n_samples_per_prompt,
                    get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
                    self.cfg.environment.env_class,
                    "train",
                    self.global_step,
                )
                assert all(uid == uids[0] for uid in uids), "Expect all uids to be the same"

                # Capture a fallback global step before collection. Runners that
                # record sampled-token steps replace it with actual_global_step below.
                global_step_at_start = self.global_step

                # Disable each runner's progress bar so concurrent workers do not flood the console.
                cur_trajectory_batch: TrajectoryBatch = await self.trajectory_runner.run(
                    trajectory_request, disable_tqdm=True
                )
                actual_step = cur_trajectory_batch.get("actual_global_step")
                staleness_step = actual_step if actual_step is not None else global_step_at_start

                record_generated_work(
                    cur_trajectory_batch["response_ids"],
                    cur_trajectory_batch.get("is_last_step"),
                    staleness_step,
                )
                completed_group = GeneratedOutputGroup(
                    trajectory_batch=cur_trajectory_batch,
                    uid=uids[0],
                    earliest_model_step=staleness_step,
                    source_prompts=rand_prompts,
                )
                freshness = await self._enqueue_if_fresh(queues, completed_group)
                if freshness is _GroupFreshness.STALE:
                    await self._staleness_manager.cancel_submission_slot()
                    slot_acquired = False
                    self._record_admission_scan(
                        [(completed_group, AdmissionDecision((AdmissionRejection.STALE,)))],
                        inspected_count=1,
                    )
                    continue
                record_rollout_buffer(queues.completed.qsize(), queues.completed.maxsize)
                await self._staleness_manager.on_rollout_accepted()
                slot_acquired = False  # Slot properly released; safe for next iteration
        except asyncio.CancelledError:
            # If a slot was acquired but generation was cancelled before
            # on_rollout_accepted() ran, undo the slot acquisition so that
            # validate_state_at_epoch_end() sees running == 0 and
            # submitted == accepted.
            if slot_acquired:
                await self._staleness_manager.cancel_submission_slot()
            return
        except GenerationStalledError:
            # The dataset is exhausted and no retries are arriving — this
            # worker has no more work to do for the epoch. Exit gracefully.
            if slot_acquired:
                await self._staleness_manager.cancel_submission_slot()
            logger.info("Trajectory worker exiting: collection stalled (dataset exhausted, no retries)")
            return
        except Exception as e:
            log_exception_as_text("Trajectory worker failed", e)
            if slot_acquired:
                await self._staleness_manager.cancel_submission_slot()
            sys.exit(1)
        finally:
            await queues.mark_producer_finished()

    async def _next_generation_prompts(
        self,
        queues: _GenerationQueues,
    ) -> List[dict]:
        """Prefer retries and wait for one after the epoch's dataset rows are scheduled.

        Raises ``GenerationStalledError`` when the dataset is exhausted and no
        retries arrive within the stall deadline, so the caller can end the
        epoch instead of blocking forever.
        """
        try:
            return queues.retries.get_nowait()
        except asyncio.QueueEmpty:
            prompts = await self.async_train_dataloader.get_next_non_consumed_data()
            if prompts is not None:
                return prompts

        try:
            return await asyncio.wait_for(
                queues.retries.get(),
                timeout=self._generation_stall_timeout(),
            )
        except asyncio.TimeoutError:
            raise GenerationStalledError("Dataset exhausted and no retries arrived within the stall deadline")

    async def _enqueue_if_fresh(self, queues: _GenerationQueues, group: GeneratedOutputGroup) -> _GroupFreshness:
        """Enqueue a fresh group or route a stale group to retry."""
        async with queues.condition:
            while queues.completed.full():
                await queues.condition.wait()
            freshness = self._classify_and_route_group(queues, group)
            if freshness is _GroupFreshness.STALE:
                return freshness
            queues.completed.put_nowait(group)
            queues.condition.notify_all()
            return freshness

    async def async_sync_policy_weights_to_inference_engines(self):
        # Pre-broadcast drain: hard-sync every policy shard rank's event loop BEFORE the
        # weight-extract gather that broadcast_to_inference_engines runs. extract_weights
        # fires mesh_fsdp `_all_gather_base` collectives (fsdp_worker._gather_tensor) on a
        # submesh PG that inherits torch's default 600s timeout (WORLD PG is
        # SKYRL_WORKER_NCCL_TIMEOUT_IN_S); pre-gather per-rank skew (the streamed extract's
        # plan-build / a GDN backward slow-path) then makes a laggard miss the 600s window
        # -> waiter ranks SIGABRT (the r4h gs1 death, #6936 _all_gather_base at the
        # policy_train->sync_weights transition). Symmetric to the POST-broadcast drain at
        # the call sites + the ppo_train entry barrier (worker.py); reuses the proven
        # async-loop-safe barrier_all (WORLD PG >> the 600s submesh default).
        await self._drain_policy_event_loops()
        return await self.policy_model.async_run_method(
            "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
        )

    async def _drain_policy_event_loops(self):
        """Drain barrier before each forward (the MoE-RL async-dispatch wedge fix, 2026-06-29).

        FR-decode proved the gs-1 CoreWeave MoE wedge: the FSDP policy worker is a Ray
        ASYNC actor (single event-loop thread, because it has `async def` methods like
        `broadcast_to_inference_engines`). `worker.forward` is a plain SYNC method that
        runs to completion on the loop thread WITHOUT yielding. After the disaggregated
        per-step weight-sync, the peer ranks' loops were still occupied by the broadcast
        coroutine task -> only rank 0's loop was free and ran the forward (into a lonely
        mesh_fsdp unshard `_all_gather_base` -> 1800s NCCL watchdog); ranks 8/16/24's
        queued `forward` task was never scheduled.

        `barrier_all` is now an `async def` actor method (see worker.py). Dispatching it
        and awaiting its refs GUARANTEES each peer's loop drained to idle (Ray cannot
        resolve an async-method ObjectRef until the loop scheduled+ran the coroutine to
        completion, and the coroutine `await`s a loop turn before its collective). So by
        the time `await asyncio.gather(*refs)` returns, every policy shard rank's event
        loop is provably free, and the subsequent sync `forward.remote()` can be scheduled
        on every peer. Robust to BOTH M1 (loop busy with the sync forward HOL-block) and
        M2 (broadcast coroutine not yet unwound) — the prior SYNC `barrier_all` was not,
        because a sync drain method is HOL-blocked exactly like the sync forward it guards.

        We `await asyncio.gather(*refs)` rather than a blocking `ray.get` so we don't
        stall the async trainer's own event-loop thread; the driver coroutine still does
        not advance to the forward dispatch until every rank's drain has completed.
        """
        refs = self.policy_model.async_run_ray_method("pass_through", "barrier_all")
        await asyncio.gather(*refs)

    def _classify_and_route_group(self, queues: _GenerationQueues, group: GeneratedOutputGroup) -> _GroupFreshness:
        if self._group_admission_policy.is_stale(group, global_step=self.global_step):
            queues.retries.put_nowait(group.source_prompts)
            return _GroupFreshness.STALE
        return _GroupFreshness.FRESH

    def _record_admission_scan(
        self,
        rejected_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]],
        *,
        inspected_count: int,
    ) -> None:
        self._groups_rejected_since_step += len(rejected_groups)
        for _, decision in rejected_groups:
            assert decision.primary_rejection is not None
            self._rejection_reasons_since_step[decision.primary_rejection.value] += 1
        self._groups_inspected_since_step += inspected_count

    def _partition_completed_groups(
        self, completed_groups: List[GeneratedOutputGroup], occupied_uids: set[str]
    ) -> _AdmissionPartition:
        """Evaluate completed work and select at most one representative per UID."""
        decisions = [
            self._group_admission_policy.evaluate(group, global_step=self.global_step) for group in completed_groups
        ]
        selected_index_by_uid: dict[str, int] = {}
        for index, (group, decision) in enumerate(zip(completed_groups, decisions, strict=True)):
            if group.uid in occupied_uids:
                continue
            selected_index = selected_index_by_uid.get(group.uid)
            if selected_index is None or (decision.accepted and not decisions[selected_index].accepted):
                selected_index_by_uid[group.uid] = index

        duplicate_decision = AdmissionDecision((AdmissionRejection.DUPLICATE_UID,))
        accepted_groups = []
        rejected_groups = []
        discarded_groups = []
        for index, (group, decision) in enumerate(zip(completed_groups, decisions, strict=True)):
            if group.uid in occupied_uids or selected_index_by_uid[group.uid] != index:
                discarded_groups.append((group, duplicate_decision))
            elif decision.accepted:
                accepted_groups.append(group)
            else:
                rejected_groups.append((group, decision))
        return _AdmissionPartition(
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            discarded_groups=discarded_groups,
        )

    def _publish_admission_metrics(self, *, dynamic_candidate_count: int, dynamic_discarded_count: int) -> None:
        rejected = self._groups_rejected_since_step
        inspected = self._groups_inspected_since_step
        assert inspected > 0, "An admitted training batch requires at least one inspected completed group"
        reason_counts = self._rejection_reasons_since_step
        self._groups_rejected_since_step = 0
        self._rejection_reasons_since_step = collections.Counter()
        self._groups_inspected_since_step = 0
        metrics = {
            "async/rejected_count": rejected,
            "async/rejected_rate": rejected / inspected,
        }
        if self._dynamic_sampling_type is DynamicSamplingType.FILTER:
            metrics.update(
                {
                    "async/dynamic_sampling/candidate_count": dynamic_candidate_count,
                    "async/dynamic_sampling/discarded_count": dynamic_discarded_count,
                    "async/dynamic_sampling/discarded_rate": (
                        dynamic_discarded_count / dynamic_candidate_count if dynamic_candidate_count else 0.0
                    ),
                }
            )
        metrics.update(
            {f"async/rejected_count/{reason.value}": reason_counts[reason.value] for reason in AdmissionRejection}
        )
        self.all_metrics.update(metrics)
        if rejected:
            logger.warning(
                f"Rejected {rejected} completed groups before step {self.global_step}; "
                f"reasons={dict(reason_counts)}. Waiting produced a full "
                f"{self.mini_batch_size}-group replacement batch."
            )
        if dynamic_discarded_count:
            logger.info(
                f"Dynamic sampling discarded {dynamic_discarded_count} of {dynamic_candidate_count} "
                f"candidate groups before step {self.global_step}."
            )

    def _generation_stall_timeout(self) -> float:
        """Adaptive deadline for receiving new groups during a generation wait.

        Returns a multiple of the recent median step time (at least 10 minutes)
        so the stall fires long before a human would notice, but never during
        normal cadence.  When no step history exists (first step), defaults to
        30 minutes.
        """
        if not self._step_time_history:
            return 1800.0
        sorted_times = sorted(self._step_time_history)
        median = sorted_times[len(sorted_times) // 2]
        return max(median * 5.0, 600.0)

    def _raise_admission_stall(
        self,
        elapsed: float,
        rejection_counts: collections.Counter[str],
        *,
        active_producers: int,
    ) -> None:
        """Bound a step that has admitted no new groups, even if producer tasks remain alive."""
        raise GenerationStalledError(
            f"Generation stalled: no groups admitted for {elapsed:.0f}s; "
            f"active_producers={active_producers}, "
            f"rejected_completions={dict(rejection_counts)}"
        )

    def _select_dynamic_sampling_candidates(
        self,
        candidates: List[GeneratedOutputGroup],
        *,
        available_slots: int,
    ) -> _CandidateSelection:
        admitted_groups = []
        discarded_reasons: collections.Counter[str] = collections.Counter()
        candidate_count = 0

        for candidate_index, group in enumerate(candidates):
            if len(admitted_groups) >= available_slots:
                return _CandidateSelection(
                    admitted_groups=admitted_groups,
                    surplus_groups=candidates[candidate_index:],
                    discarded_reasons=discarded_reasons,
                    candidate_count=candidate_count,
                )

            selection_result = self._group_selection_policy.evaluate(group)
            candidate_count += int(self._dynamic_sampling_type is DynamicSamplingType.FILTER)
            if selection_result is GroupSelectionResult.KEEP:
                admitted_groups.append(group)
            else:
                discarded_reasons[selection_result.value] += 1

        return _CandidateSelection(
            admitted_groups=admitted_groups,
            surplus_groups=[],
            discarded_reasons=discarded_reasons,
            candidate_count=candidate_count,
        )

    async def _get_admitted_generation_group_mini_batch(self, queues: _GenerationQueues) -> List[GeneratedOutputGroup]:
        """Discard or retry rejected groups and wait for a full admitted mini-batch.

        Raises:
            GenerationStalledError: No producer can make admission progress.
            RuntimeError: Dynamic sampling exhausts its per-step candidate budget.
        """
        if queues.admitted_groups_consumed:
            raise RuntimeError("cannot assemble a new batch before clearing the previously consumed batch")
        accepted_groups = queues.admitted_groups
        loop = asyncio.get_event_loop()
        last_admitted_progress = loop.time()
        stall_timeout = float(self.admission_stall_timeout)
        rejection_counts_since_admission: collections.Counter[str] = collections.Counter()
        dynamic_candidate_count = 0
        dynamic_discarded_count = 0

        while True:
            async with queues.condition:
                while len(accepted_groups) < self.mini_batch_size and queues.completed.empty():
                    if queues.active_producers == 0:
                        raise GenerationStalledError(
                            "Generation exhausted its dataset before assembling a complete training batch: "
                            f"admitted={len(accepted_groups)}/{self.mini_batch_size}, "
                            f"dynamic_candidates={dynamic_candidate_count}, "
                            f"dynamic_discarded={dynamic_discarded_count}, "
                            f"rejections={dict(rejection_counts_since_admission)}"
                        )
                    elapsed = loop.time() - last_admitted_progress
                    remaining = stall_timeout - elapsed
                    if remaining <= 0:
                        self._raise_admission_stall(
                            elapsed,
                            rejection_counts_since_admission,
                            active_producers=queues.active_producers,
                        )
                    try:
                        await asyncio.wait_for(queues.condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        self._raise_admission_stall(
                            loop.time() - last_admitted_progress,
                            rejection_counts_since_admission,
                            active_producers=queues.active_producers,
                        )

                completed_groups = _drain_queue(queues.completed)
                partition = self._partition_completed_groups(
                    completed_groups,
                    occupied_uids={group.uid for group in accepted_groups}
                    | self.data_tracker.get_consumed_uids_in_epoch(),
                )
                for group, decision in partition.rejected_groups:
                    queues.retries.put_nowait(group.source_prompts)
                    assert decision.primary_rejection is not None
                    rejection_counts_since_admission[decision.primary_rejection.value] += 1

                selection = self._select_dynamic_sampling_candidates(
                    partition.accepted_groups,
                    available_slots=self.mini_batch_size - len(accepted_groups),
                )
                queues.record_admitted(selection.admitted_groups)
                dynamic_candidate_count += selection.candidate_count
                dynamic_discarded_this_scan = sum(selection.discarded_reasons.values())
                dynamic_discarded_count += dynamic_discarded_this_scan
                rejection_counts_since_admission.update(selection.discarded_reasons)

                for group in selection.surplus_groups:
                    queues.completed.put_nowait(group)

                if selection.admitted_groups:
                    last_admitted_progress = loop.time()
                    stall_timeout = float(self.admission_stall_timeout)
                    rejection_counts_since_admission.clear()

                if len(accepted_groups) >= self.mini_batch_size:
                    batch = accepted_groups[: self.mini_batch_size]
                else:
                    batch = None
                queues.condition.notify_all()

            self._record_admission_scan(
                partition.rejected_groups + partition.discarded_groups,
                inspected_count=len(completed_groups),
            )
            discarded_count = (
                len(partition.rejected_groups) + len(partition.discarded_groups) + dynamic_discarded_this_scan
            )
            if discarded_count:
                await self._staleness_manager.on_rollouts_discarded(discarded_count)

            if (
                batch is None
                and self._dynamic_sampling_max_candidate_groups is not None
                and dynamic_candidate_count >= self._dynamic_sampling_max_candidate_groups
            ):
                raise RuntimeError(
                    "Exiting training loop due to hitting dynamic sampling limit for filter strategy with "
                    f"{self._dynamic_sampling_max_sample_batches} max sample batches. "
                    f"Collected {len(accepted_groups)} of {self.mini_batch_size} required groups."
                )

            if batch is not None:
                break

        self._publish_admission_metrics(
            dynamic_candidate_count=dynamic_candidate_count,
            dynamic_discarded_count=dynamic_discarded_count,
        )
        return batch

    def convert_generation_group_mini_batch_to_training_input(
        self, cur_generation_group_mini_batch: List[GeneratedOutputGroup]
    ) -> TrainingInputBatch:
        """Convert one complete generated mini-batch to a training batch."""
        assert len(cur_generation_group_mini_batch) == self.mini_batch_size, (
            f"Expected {self.mini_batch_size} generated groups, got {len(cur_generation_group_mini_batch)}"
        )
        trajectory_batches = []
        uids = []
        stalenesses = []
        for cur_generated_output_group in cur_generation_group_mini_batch:
            cur_staleness = self.global_step - cur_generated_output_group.earliest_model_step
            stalenesses.append(cur_staleness)
            trajectory_batches.append(cur_generated_output_group.trajectory_batch)
            group_size = len(cur_generated_output_group.trajectory_batch["response_ids"])
            uids.extend([cur_generated_output_group.uid] * group_size)

        record_rollout_staleness(stalenesses, self.global_step)

        assert max(stalenesses) <= self.max_staleness_steps, (
            f"Fresh batch assembly returned staleness {max(stalenesses)} above max {self.max_staleness_steps}"
        )

        trajectory_batch = concatenate_trajectory_batches(
            trajectory_batches,
            require_rollout_logprobs=policy_loss_requires_rollout_logprobs(self.cfg.trainer.algorithm.policy_loss_type),
            tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold),
        )
        assert trajectory_batch["rollout_metrics"] is not None, "Rollout metrics should be non-null."
        self.all_metrics.update(trajectory_batch["rollout_metrics"])

        # Log staleness statistics for this step
        self.all_metrics.update(
            {
                "async/staleness_mean": sum(stalenesses) / len(stalenesses),
                "async/staleness_max": max(stalenesses),
                "async/staleness_min": min(stalenesses),
                "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
            }
        )

        # Convert rewards to per-token form and compute reward metrics before training conversion
        trajectory_batch = self.postprocess_trajectory_batch(trajectory_batch, uids)

        # print example just for debugging
        vis = self.tokenizer.decode(trajectory_batch["response_ids"][0])
        logger.debug(f"Example generated: {vis}")

        return self.convert_to_training_input(trajectory_batch, uids)

    def save_checkpoints(self):
        """
        Save checkpoints. Data consumption state is persisted by DataTrackingCallback.on_save,
        which fires after the base checkpoint save completes.
        """
        # The base method saves model, dataloader state, trainer_state, and latest_ckpt_global_step.txt.
        # DataTrackingCallback.on_save (registered in __init__) writes data_consumption_state.pt.
        super().save_checkpoints()

    def load_checkpoints(self) -> Tuple[int, str]:
        """
        Load the base checkpoint.

        Data consumption state loading is handled separately via
        DataTrackingCallback.load_from_checkpoint() in _train_loop().

        Returns the global step to resume from and the checkpoint path.
        """
        global_step, checkpoint_path = super().load_checkpoints()
        return global_step, checkpoint_path
