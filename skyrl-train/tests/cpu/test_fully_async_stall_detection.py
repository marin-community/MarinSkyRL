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
    alive_task = asyncio.get_event_loop().create_task(asyncio.sleep(100))
    trainer = _bare_trainer(tasks=[alive_task])
    try:
        new_timeout = trainer._check_generation_stall(elapsed=600.0)
        assert new_timeout == trainer._generation_stall_timeout()
    finally:
        alive_task.cancel()


# --------------------------------------------------------------------------- #
# _get_fresh_generation_group_mini_batch — end-to-end stall detection         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_fresh_batch_raises_when_generators_dead(monkeypatch):
    """When all generators have exited and the buffer is short, raise immediately."""
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    # Patch the timeout to 0.05s so the test runs fast.
    monkeypatch.setattr(trainer, "_generation_stall_timeout", lambda: 0.05)

    queues = _make_queues()

    with pytest.raises(GenerationStalledError, match="no active generators"):
        await trainer._get_fresh_generation_group_mini_batch(queues)


@pytest.mark.asyncio
async def test_get_fresh_batch_returns_when_groups_arrive(monkeypatch):
    """Normal path: groups arrive and the batch completes before the stall deadline."""
    trainer = _bare_trainer(mini_batch_size=2, tasks=[])
    monkeypatch.setattr(trainer, "_generation_stall_timeout", lambda: 10.0)

    queues = _make_queues()

    # Stub partition: all groups are fresh, none stale.
    from skyrl_train.fully_async_trainer import _FreshnessPartition

    monkeypatch.setattr(
        trainer,
        "_partition_and_retry_stale_groups",
        lambda q, groups: _FreshnessPartition(fresh_groups=groups, stale_groups=[]),
    )
    # Stub metric helpers.
    monkeypatch.setattr(trainer, "_record_discard_scan", lambda *a, **kw: None)
    monkeypatch.setattr(trainer, "_publish_discard_metrics", lambda: None)

    # Feed two groups after a short delay.
    async def _producer():
        await asyncio.sleep(0.05)
        for i in range(2):
            await queues.completed.put(
                GeneratedOutputGroup(
                    trajectory_batch={"response_ids": [[1]], "prompt_token_ids": [[1]]},
                    uid=f"u{i}",
                    earliest_model_step=0,
                    source_prompts=[{}],
                )
            )
            async with queues.condition:
                queues.condition.notify_all()

    asyncio.get_event_loop().create_task(_producer())

    batch = await trainer._get_fresh_generation_group_mini_batch(queues)
    assert len(batch) == 2
