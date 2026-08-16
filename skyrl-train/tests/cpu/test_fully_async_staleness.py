import asyncio
import collections

import pytest

from skyrl_train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GenerationStalledError,
    GeneratedOutputGroup,
    _AsyncStalenessManager,
    _GenerationQueues,
)
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.trajectory_runners.base import TrajectoryID


def _generated_group(uid: str, earliest_model_step: int, *, fully_masked: bool = False) -> GeneratedOutputGroup:
    trajectory_batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": [0.0, 1.0],
        "loss_masks": [[0], [0]] if fully_masked else [[1], [1]],
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
    trainer._groups_rejected_since_step = 0
    trainer._rejection_reasons_since_step = collections.Counter()
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
    trainer._group_admission_policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.exact_physical(physical_group_size=2),
        max_staleness_steps=2,
        rollout_logprobs_required=False,
    )
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

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["fresh-1", "fresh-2"]
    assert queues.completed.empty()
    assert [queues.retries.get_nowait()[0]["uid"] for _ in range(2)] == [
        "stale-in-batch",
        "stale-beyond-batch",
    ]
    assert trainer.all_metrics["async/rejected_count"] == 2
    assert trainer.all_metrics["async/rejected_rate"] == 0.5
    assert trainer.all_metrics["async/rejected_count/stale"] == 2


@pytest.mark.asyncio
async def test_batch_assembly_waits_for_fresh_replacement():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=7))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "retry-me"

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["retry-me"]


@pytest.mark.asyncio
async def test_batch_assembly_retries_fully_masked_group_and_waits_for_replacement():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    queues.completed.put_nowait(_generated_group("retry-me", earliest_model_step=10, fully_masked=True))

    pending_batch = asyncio.create_task(trainer._get_admitted_generation_group_mini_batch(queues))
    done, _ = await asyncio.wait({pending_batch}, timeout=0)
    assert pending_batch not in done
    assert queues.retries.get_nowait()[0]["uid"] == "retry-me"

    async with queues.condition:
        queues.completed.put_nowait(_generated_group("replacement", earliest_model_step=10))
        queues.condition.notify_all()
    batch = await asyncio.wait_for(pending_batch, timeout=1)

    assert [group.uid for group in batch] == ["replacement"]
    assert trainer.all_metrics["async/rejected_count/fully_masked"] == 1


@pytest.mark.asyncio
async def test_batch_assembly_scans_rejections_and_preserves_accepted_surplus():
    trainer, queues = _batch_assembly_state(mini_batch_size=2, accepted=4)
    for group in [
        _generated_group("accepted-1", earliest_model_step=10),
        _generated_group("accepted-2", earliest_model_step=10),
        _generated_group("masked-beyond-batch", earliest_model_step=10, fully_masked=True),
        _generated_group("accepted-surplus", earliest_model_step=10),
    ]:
        queues.completed.put_nowait(group)

    batch = await trainer._get_admitted_generation_group_mini_batch(queues)

    assert [group.uid for group in batch] == ["accepted-1", "accepted-2"]
    assert queues.retries.get_nowait()[0]["uid"] == "masked-beyond-batch"
    assert queues.completed.get_nowait().uid == "accepted-surplus"


@pytest.mark.asyncio
async def test_batch_assembly_rejected_only_progress_terminates_instead_of_livelocking():
    trainer, queues = _batch_assembly_state(mini_batch_size=1, accepted=1)
    trainer._generation_stall_timeout = lambda: 0.0
    queues.completed.put_nowait(_generated_group("always-masked", earliest_model_step=10, fully_masked=True))

    with pytest.raises(GenerationStalledError):
        await trainer._get_admitted_generation_group_mini_batch(queues)
