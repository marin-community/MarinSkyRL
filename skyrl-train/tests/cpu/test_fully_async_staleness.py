import asyncio
from types import MethodType

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


def test_stale_groups_do_not_reduce_training_batch_size():
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 10
    trainer.max_staleness_steps = 2
    trainer.mini_batch_size = 64
    trainer.all_metrics = {}
    trainer.tokenizer = type("Tokenizer", (), {"decode": lambda self, tokens: str(tokens)})()
    trainer.postprocess_generator_output = MethodType(lambda self, output, uids: output, trainer)
    trainer.convert_to_training_input = MethodType(lambda self, output, uids: output, trainer)

    stale_groups = [_generated_group(f"stale-{index}", scheduled_step=7) for index in range(61)]
    fresh_groups = [_generated_group(f"fresh-{index}", scheduled_step=10) for index in range(3)]
    training_input = trainer.convert_generation_group_mini_batch_to_training_input(stale_groups + fresh_groups)

    assert len(training_input["response_ids"]) == 64 * 2
    assert trainer.all_metrics["async/staleness_violation_count"] == 61
    assert trainer.all_metrics["async/staleness_violation_rate"] == 61 / 64
