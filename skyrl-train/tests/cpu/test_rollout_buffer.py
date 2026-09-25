"""Rollout buffer contract and trainer handoff tests."""

import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.fully_async_trainer import _AsyncStalenessManager, _GenerationQueues
from skyrl_train.rollout_buffer import FineStoreRolloutBuffer, MemoryRolloutBuffer
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


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_buffer_contract_preserves_order_and_evidence_across_snapshot(tmp_path, backend):
    def open_buffer(path):
        return MemoryRolloutBuffer() if backend == "memory" else FineStoreRolloutBuffer(str(path))

    async def exercise():
        buffer = open_buffer(tmp_path / "original")
        for uid in ("fast", "slow", "later"):
            await buffer.writer().write_rollout(_group(uid))
        snapshot = buffer.snapshot()
        assert snapshot.pending_uids == ("fast", "slow", "later")
        buffer.close()

        restored = open_buffer(tmp_path / "resumed")
        try:
            restored.restore(snapshot)
            first = await restored.next_batch(2)
            if backend == "memory":
                # Memory admission scans all ready groups to filter stale work.
                assert [group.uid for group in first] == ["fast", "slow", "later"]
            else:
                second = await restored.next_batch(2)
                assert [group.uid for group in first + second] == ["fast", "slow", "later"]
            assert first[0].trajectory_batch == _group("fast").trajectory_batch
            assert restored.empty()
        finally:
            restored.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_checkpoint_roundtrip_uses_opaque_buffer_snapshot(tmp_path, backend):
    async def exercise():
        buffer = MemoryRolloutBuffer() if backend == "memory" else FineStoreRolloutBuffer(str(tmp_path / "rollouts"))
        queues = _GenerationQueues(rollout_buffer=buffer, retries=asyncio.Queue(), condition=asyncio.Condition())
        callback = BufferCheckpointCallback()
        callback.bind_queues(queues)
        checkpoint = tmp_path / "global_step_1"
        checkpoint.mkdir()
        await queues.rollout_buffer.writer().write_rollout(_group("pending"))
        trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path))))
        await callback.on_save_async(SimpleNamespace(global_step=1), SimpleNamespace(), trainer=trainer)
        buffer.close()

        state = callback.load_buffer_state(str(checkpoint))
        assert state.buffer.backend == backend
        assert state.pending_uids() == {"pending"}
        resumed = MemoryRolloutBuffer() if backend == "memory" else FineStoreRolloutBuffer(str(tmp_path / "resumed"))
        try:
            resumed.restore(state.buffer)
            result = await resumed.next_batch(1)
            assert result[0].trajectory_batch == _group("pending").trajectory_batch
        finally:
            resumed.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_off_policy_workers_can_keep_producing_while_trainer_step_waits(tmp_path, backend):
    async def exercise() -> list[str]:
        slots = _AsyncStalenessManager(max_concurrent_generation_groups=2, mini_batch_size=1, max_staleness_steps=1)
        buffer = (
            MemoryRolloutBuffer(slot_policy=slots)
            if backend == "memory"
            else FineStoreRolloutBuffer(str(tmp_path / "rollouts"), slot_policy=slots)
        )
        for index in range(3):
            async with asyncio.timeout(1), buffer.request_slot() as slot:
                await slot.write_rollout(_group(str(index)))
        pending = list(buffer.snapshot().pending_uids)
        buffer.close()
        return pending

    assert asyncio.run(exercise()) == ["0", "1", "2"]


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_on_policy_slot_waits_for_next_training_step(tmp_path, backend):
    async def exercise() -> None:
        policy = _AsyncStalenessManager(max_concurrent_generation_groups=2, mini_batch_size=1, max_staleness_steps=0)
        buffer = (
            MemoryRolloutBuffer(capacity=1, slot_policy=policy)
            if backend == "memory"
            else FineStoreRolloutBuffer(str(tmp_path / "rollouts"), capacity=1, slot_policy=policy)
        )
        async with buffer.request_slot() as slot:
            await slot.write_rollout(_group("first"))
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.01), buffer.request_slot():
                pass
        assert [group.uid for group in await buffer.next_batch(1)] == ["first"]
        await policy.notify_capacity_change(2)
        async with asyncio.timeout(1), buffer.request_slot() as slot:
            await slot.write_rollout(_group("second"))
        buffer.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_synchronous_trainer_uses_buffer_contract(tmp_path, backend):
    async def exercise():
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = SimpleNamespace(
            trainer=SimpleNamespace(ckpt_path=str(tmp_path), rollout_buffer=SimpleNamespace(backend=backend))
        )
        trainer.global_step = 7
        trainer._rollout_buffer = None
        original = _group("example").trajectory_batch
        try:
            batch, uids = await trainer._handoff_generated_batch(original, ["example"], [{"uid": "example"}])
            assert batch == original
            assert uids == ["example"]
            if backend == "finestore":
                assert batch is not original
        finally:
            if trainer._rollout_buffer is not None:
                trainer._rollout_buffer.close()

    asyncio.run(exercise())
