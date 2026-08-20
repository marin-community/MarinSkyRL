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


def _make_queues() -> _GenerationQueues:
    return _GenerationQueues(
        completed=asyncio.Queue(),
        retries=asyncio.Queue(),
        condition=asyncio.Condition(),
    )


def _bare_trainer(mini_batch_size=2, step_times=None, tasks=None) -> FullyAsyncRayPPOTrainer:
    """Create a trainer shell with just enough state for stall-detection tests."""
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.mini_batch_size = mini_batch_size
    trainer._step_time_history = collections.deque(step_times or [], maxlen=5)
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
# _generation_stall_timeout — adaptive computation                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], 1800.0),
        ([100.0, 200.0, 300.0], 1000.0),  # median 200 × 5
        ([1.0, 2.0, 3.0], 600.0),  # median 2 × 5 = 10, floored to 600
    ],
)
def test_generation_stall_timeout(history, expected):
    trainer = _bare_trainer(step_times=history)
    assert trainer._generation_stall_timeout() == expected


# --------------------------------------------------------------------------- #
# _any_generators_alive and _check_generation_stall                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_check_stall_raises_when_no_generators():
    trainer = _bare_trainer(tasks=[])
    with pytest.raises(GenerationStalledError, match="no active generators"):
        trainer._check_generation_stall(elapsed=600.0)


@pytest.mark.asyncio
async def test_check_stall_extends_when_generators_alive():
    alive_task = asyncio.create_task(asyncio.Event().wait())
    trainer = _bare_trainer(tasks=[alive_task])
    try:
        new_timeout = trainer._check_generation_stall(elapsed=600.0)
        assert new_timeout == trainer._generation_stall_timeout()
    finally:
        alive_task.cancel()


# --------------------------------------------------------------------------- #
# _get_admitted_generation_group_mini_batch — end-to-end stall detection      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_admitted_batch_raises_when_generators_dead(monkeypatch):
    """When all generators have exited and the buffer is short, raise immediately."""
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    # Patch the timeout to 0.05s so the test runs fast.
    monkeypatch.setattr(trainer, "_generation_stall_timeout", lambda: 0.05)

    queues = _make_queues()

    with pytest.raises(GenerationStalledError, match="no active generators"):
        await trainer._get_admitted_generation_group_mini_batch(queues)


@pytest.mark.asyncio
async def test_get_admitted_batch_returns_complete_group_set(monkeypatch):
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    monkeypatch.setattr(trainer, "_generation_stall_timeout", lambda: 10.0)

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
