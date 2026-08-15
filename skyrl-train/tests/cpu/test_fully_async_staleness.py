import asyncio
import collections

import pytest

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncStalenessManager,
    _GenerationQueues,
)
from skyrl_train.trajectory_runners.base import TrajectoryID


def _generated_group(uid: str, earliest_model_step: int) -> GeneratedOutputGroup:
    trajectory_batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": [0.0, 1.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": [
            TrajectoryID(instance_id=uid, repetition_id=0),
            TrajectoryID(instance_id=uid, repetition_id=1),
        ],
        "is_last_step": [True, True],
        "exclude_from_baseline": [False, False],
    }
    return GeneratedOutputGroup(
        trajectory_batch=trajectory_batch,
        uid=uid,
        earliest_model_step=earliest_model_step,
        source_prompts=[{"uid": uid}],
    )


def _batch_assembly_state(mini_batch_size: int, accepted: int):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = mini_batch_size
    trainer.all_metrics = {}
    trainer._stale_groups_discarded_since_step = 0
    trainer._groups_inspected_since_step = 0
    trainer._step_time_history = collections.deque([1000.0], maxlen=5)
    trainer._active_generator_tasks = []
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=accepted,
        mini_batch_size=mini_batch_size,
        max_staleness_steps=2,
    )
    trainer._staleness_manager._stat.submitted = accepted
    trainer._staleness_manager._stat.accepted = accepted
    queues = _GenerationQueues(completed=asyncio.Queue(), retries=asyncio.Queue(), condition=asyncio.Condition())
    return trainer, queues


@pytest.mark.asyncio
async def test_staleness_manager_blocks_work_beyond_capacity_until_training_advances():
    manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=2,
        mini_batch_size=1,
        max_staleness_steps=0,
    )
    await manager.acquire_submission_slot()

    next_submission = asyncio.create_task(manager.acquire_submission_slot())
    done, _ = await asyncio.wait({next_submission}, timeout=0)
    assert next_submission not in done

    await manager.notify_capacity_change(new_global_step=2)
    await asyncio.wait_for(next_submission, timeout=1)

    await manager.on_rollout_accepted()
    await manager.on_rollout_accepted()


@pytest.mark.asyncio
async def test_batch_assembly_retries_stale_groups_from_entire_buffer():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=4)
    for group in [
        _generated_group("stale-in-batch", earliest_model_step=7),
        _generated_group("fresh-1", earliest_model_step=10),
        _generated_group("fresh-2", earliest_model_step=9),
        _generated_group("stale-beyond-batch", earliest_model_step=6),
    ]:
        queues.completed.put_nowait(group)

    batch = await trainer._get_fresh_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["fresh-1", "fresh-2"]
    assert queues.completed.empty()
    assert [queues.retries.get_nowait()[0]["uid"] for _ in range(2)] == [
        "stale-in-batch",
        "stale-beyond-batch",
    ]
    assert trainer.all_metrics["async/discarded_count"] == 2
    assert trainer.all_metrics["async/discard_rate"] == 0.5
    assert trainer._staleness_manager._stat.accepted == 2


@pytest.mark.asyncio
async def test_batch_assembly_waits_for_fresh_replacement():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=7))

    pending_batch = asyncio.create_task(trainer._get_fresh_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "retry-me"

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["retry-me"]
