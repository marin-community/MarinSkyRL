"""Regression tests for hung-trial stall detection in fully async training.

Each test exercises the actual production code paths added to fix the three
defects in ``agent_logs/2026-08-13_escalation-one-hung-trial-permanently-stalls-an-async-epoch.md``.
"""

from __future__ import annotations

import asyncio
import collections

import pytest

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GenerationStalledError,
    _GenerationQueues,
)
from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.dynamic_sampling import GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant


def _make_queues(*, active_producers=0) -> _GenerationQueues:
    return _GenerationQueues(
        completed=asyncio.Queue(),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
        active_producers=active_producers,
    )


def _bare_trainer(
    mini_batch_size=2,
    step_times=None,
    tasks=None,
    admission_stall_timeout=21_600,
) -> FullyAsyncRayPPOTrainer:
    """Create a trainer shell with just enough state for stall-detection tests."""
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.mini_batch_size = mini_batch_size
    trainer._step_time_history = collections.deque(step_times or [], maxlen=5)
    trainer.admission_stall_timeout = admission_stall_timeout
    trainer._active_trajectory_tasks = tasks or []
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer._groups_rejected_since_step = 0
    trainer._rejection_reasons_since_step = collections.Counter()
    trainer._groups_inspected_since_step = 0
    trainer._dynamic_sampling_type = None
    trainer._dynamic_sampling_max_candidate_groups = None
    trainer._group_selection_policy = GroupSelectionPolicy.for_fully_async(None)
    trainer._group_admission_policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=1),
        max_staleness_steps=0,
        rollout_logprobs_required=False,
    )
    return trainer


# --------------------------------------------------------------------------- #
# Stall timeout computation                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], 1800.0),
        ([100.0, 200.0, 300.0], 1000.0),
        ([1.0, 2.0, 3.0], 600.0),
    ],
)
def test_generation_stall_timeout(history, expected):
    trainer = _bare_trainer(step_times=history)
    assert trainer._generation_stall_timeout() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [[], [100.0, 200.0, 300.0], [10_000.0, 20_000.0, 30_000.0]])
async def test_admission_stall_timeout_stops_live_but_unproductive_generators(history):
    alive_task = asyncio.create_task(asyncio.Event().wait())
    trainer = _bare_trainer(step_times=history, tasks=[alive_task], admission_stall_timeout=21_600)
    try:
        with pytest.raises(GenerationStalledError, match="active_producers=1"):
            trainer._raise_admission_stall(
                elapsed=21_600.0,
                rejection_counts=collections.Counter(),
                active_producers=1,
            )
    finally:
        alive_task.cancel()


# --------------------------------------------------------------------------- #
# _get_admitted_generation_group_mini_batch — end-to-end stall detection      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_admitted_batch_raises_when_generators_dead():
    """When all generators have exited and the buffer is short, raise immediately."""
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    queues = _make_queues()

    with pytest.raises(GenerationStalledError, match="admitted=0/2"):
        await trainer._get_admitted_generation_group_mini_batch(queues)


@pytest.mark.asyncio
async def test_get_admitted_batch_stops_when_last_producer_exhausts_dataset():
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    trainer.admission_stall_timeout = 21_600
    queues = _make_queues(active_producers=1)

    admission = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    await queues.mark_producer_finished()

    with pytest.raises(GenerationStalledError, match="admitted=0/2"):
        await asyncio.wait_for(admission, timeout=1)


@pytest.mark.asyncio
async def test_get_admitted_batch_returns_complete_group_set():
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    trainer.admission_stall_timeout = 10.0

    queues = _make_queues()
    for i in range(2):
        queues.completed.put_nowait(
            GeneratedOutputGroup(
                trajectory_batch={
                    "response_ids": [[1]],
                    "prompt_token_ids": [[1]],
                    "loss_masks": [[1]],
                    "rollout_logprobs": None,
                    "exclude_from_baseline": None,
                },
                uid=f"u{i}",
                earliest_model_step=0,
                source_prompts=[{}],
            )
        )

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)
    assert len(batch) == 2
