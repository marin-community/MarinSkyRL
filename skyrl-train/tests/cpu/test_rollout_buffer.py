"""Persistent rollout handoff through FineStore."""

import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.fully_async_trainer import _AsyncStalenessManager, _GenerationQueues
from skyrl_train.rollout_buffer import FineStoreRolloutBuffer
from skyrl_train.trainer import RayPPOTrainer


def _group(uid: str) -> GeneratedOutputGroup:
    return GeneratedOutputGroup(
        uid=uid,
        earliest_model_step=3,
        source_prompts=[{"uid": uid, "prompt": "hello"}],
        trajectory_batch={
            "prompt_token_ids": [[1, 2]],
            "response_ids": [[3, 4]],
            "rewards": [1.0],
            "loss_masks": [[1, 1]],
            "rollout_logprobs": [[-0.1, -0.2]],
            "student_topk_indices": [[[3, 5], [4, 6]]],
            "behavior_topk_logprobs": [[[-0.1, -1.2], [-0.2, -1.4]]],
            "rollout_routed_experts": [[[[2], [3]]]],
            "rollout_metrics": {"reward_mean": 1.0},
        },
    )


def test_rollout_buffer_round_trips_complete_evidence_after_reopen(tmp_path):
    path = str(tmp_path / "rollouts")
    original = _group("example")
    buffer = FineStoreRolloutBuffer(path)
    rollout_id = buffer.writer().write_rollout(original)
    buffer.close()

    reopened = FineStoreRolloutBuffer(path)
    try:
        loaded = reopened.read_rollout(rollout_id)
    finally:
        reopened.close()

    assert loaded.uid == original.uid
    assert loaded.source_prompts == original.source_prompts
    assert loaded.earliest_model_step == original.earliest_model_step
    assert loaded.trajectory_batch == original.trajectory_batch
    assert loaded.rollout_id == rollout_id


def test_completed_queue_survives_snapshot_and_reads_in_bounded_order(tmp_path):
    async def exercise() -> tuple[list[str], list[str], list[str]]:
        buffer = FineStoreRolloutBuffer(str(tmp_path / "rollouts"))
        queues = _GenerationQueues(
            completed=asyncio.Queue(),
            retries=asyncio.Queue(),
            condition=asyncio.Condition(),
            rollout_buffer=buffer,
        )
        try:
            await queues.enqueue_completed(_group("fast"))
            await queues.enqueue_completed(_group("slow"))
            await queues.enqueue_completed(_group("later"))
            snapshot_uids = [item.uid for item in queues.snapshot().completed_rollouts]
            first_batch = [group.uid for group in await queues.drain_completed(max_items=2)]
            second_batch = [group.uid for group in await queues.drain_completed(max_items=2)]
            return snapshot_uids, first_batch, second_batch
        finally:
            buffer.close()

    assert asyncio.run(exercise()) == (["fast", "slow", "later"], ["fast", "slow"], ["later"])


def test_checkpoint_preserves_pending_references_without_copying_payloads(tmp_path):
    async def save() -> None:
        buffer = FineStoreRolloutBuffer(str(tmp_path / "rollouts"))
        queues = _GenerationQueues(
            completed=asyncio.Queue(),
            retries=asyncio.Queue(),
            condition=asyncio.Condition(),
            rollout_buffer=buffer,
        )
        callback = BufferCheckpointCallback()
        callback.bind_queues(queues)
        checkpoint = tmp_path / "global_step_1"
        checkpoint.mkdir()
        try:
            await queues.enqueue_completed(_group("pending"))
            trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path))))
            await callback.on_save_async(SimpleNamespace(global_step=1), SimpleNamespace(), trainer=trainer)
        finally:
            buffer.close()

    asyncio.run(save())
    state = BufferCheckpointCallback.load_buffer_state(str(tmp_path / "global_step_1"))
    assert state.completed_groups == []
    assert [reference.uid for reference in state.completed_rollouts] == ["pending"]
    reopened = FineStoreRolloutBuffer(str(tmp_path / "resumed_run_rollouts"))
    try:
        reference = state.completed_rollouts[0]
        assert (
            reopened.read_rollout(reference.rollout_id, store_path=reference.store_path).trajectory_batch
            == _group("pending").trajectory_batch
        )
    finally:
        reopened.close()


def test_off_policy_workers_can_keep_producing_while_trainer_step_waits():
    async def exercise() -> list[int]:
        slots = _AsyncStalenessManager(
            max_concurrent_generation_groups=2,
            mini_batch_size=1,
            max_staleness_steps=1,
            continuous_production=True,
        )
        completed = []
        for index in range(3):
            await asyncio.wait_for(slots.acquire_submission_slot(), timeout=1)
            await slots.on_rollout_accepted()
            completed.append(index)
        return completed

    assert asyncio.run(exercise()) == [0, 1, 2]


def test_on_policy_slot_waits_for_next_training_step():
    async def exercise() -> None:
        slots = _AsyncStalenessManager(
            max_concurrent_generation_groups=2,
            mini_batch_size=1,
            max_staleness_steps=0,
        )
        await slots.acquire_submission_slot()
        await slots.on_rollout_accepted()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slots.acquire_submission_slot(), timeout=0.01)
        await slots.notify_capacity_change(2)
        await asyncio.wait_for(slots.acquire_submission_slot(), timeout=1)

    asyncio.run(exercise())


def test_synchronous_trainer_consumes_committed_batch_with_complete_evidence(tmp_path):
    async def exercise():
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))
        trainer.global_step = 7
        trainer._sync_rollout_buffer = None
        original = _group("example").trajectory_batch
        try:
            batch, uids = await trainer._handoff_generated_batch(
                original, ["example"], [{"uid": "example", "prompt": "hello"}]
            )
            assert batch == original
            assert uids == ["example"]
            assert batch is not original
        finally:
            if trainer._sync_rollout_buffer is not None:
                trainer._sync_rollout_buffer.close()

    asyncio.run(exercise())

    reopened = FineStoreRolloutBuffer(str(tmp_path / "rollout_buffer"))
    try:
        rows = list(reopened.store.read_view().iter_rows("rollouts"))
        assert len(rows) == 1
        assert rows[0]["model_step"] == 7
        restored = reopened.read_rollout(rows[0]["rollout_id"])
        assert restored.source_prompts == [{"uid": "example", "prompt": "hello"}]
        assert restored.trajectory_batch == _group("example").trajectory_batch
    finally:
        reopened.close()
