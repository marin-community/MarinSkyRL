import asyncio

import pytest

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncStalenessManager,
)
from skyrl_train.generators.base import TrajectoryID


def _generated_group(uid: str, scheduled_step: int) -> GeneratedOutputGroup:
    generator_output = {
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
        generator_output=generator_output,
        uid=uid,
        global_step_when_scheduled=scheduled_step,
        source_prompts=[{"uid": uid}],
    )


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
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = 2
    trainer.all_metrics = {}
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=4,
        mini_batch_size=2,
        max_staleness_steps=2,
    )
    trainer._staleness_manager._stat.submitted = 4
    trainer._staleness_manager._stat.accepted = 4
    output_buffer = asyncio.Queue()
    retry_queue = asyncio.Queue()
    buffer_condition = asyncio.Condition()
    for group in [
        _generated_group("stale-in-batch", scheduled_step=7),
        _generated_group("fresh-1", scheduled_step=10),
        _generated_group("fresh-2", scheduled_step=9),
        _generated_group("stale-beyond-batch", scheduled_step=6),
    ]:
        output_buffer.put_nowait(group)

    batch = await trainer._get_fresh_generation_group_mini_batch(output_buffer, retry_queue, buffer_condition)

    assert [group.uid for group in batch] == ["fresh-1", "fresh-2"]
    assert output_buffer.empty()
    assert [retry_queue.get_nowait()[0]["uid"] for _ in range(2)] == ["stale-in-batch", "stale-beyond-batch"]
    assert trainer.all_metrics["async/discarded_count"] == 2
    assert trainer.all_metrics["async/discard_rate"] == 0.5
    assert trainer._staleness_manager._stat.accepted == 2


@pytest.mark.asyncio
async def test_batch_assembly_waits_for_fresh_replacement():
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = 1
    trainer.all_metrics = {}
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=1,
        mini_batch_size=1,
        max_staleness_steps=2,
    )
    trainer._staleness_manager._stat.submitted = 1
    trainer._staleness_manager._stat.accepted = 1
    output_buffer = asyncio.Queue()
    retry_queue = asyncio.Queue()
    buffer_condition = asyncio.Condition()
    output_buffer.put_nowait(_generated_group("retry-me", scheduled_step=7))

    pending_batch = asyncio.create_task(
        trainer._get_fresh_generation_group_mini_batch(output_buffer, retry_queue, buffer_condition)
    )
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert retry_queue.get_nowait()[0]["uid"] == "retry-me"

    async with buffer_condition:
        output_buffer.put_nowait(_generated_group("retry-me", scheduled_step=10))
        buffer_condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["retry-me"]
