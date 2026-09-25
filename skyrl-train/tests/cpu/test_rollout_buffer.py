"""Rollout buffer contract and trainer handoff tests."""

import asyncio
import pickle
from types import SimpleNamespace

import pytest

from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.fully_async_trainer import _AsyncStalenessManager, _GenerationQueues
from skyrl_train.rollout_buffer import FineStoreRolloutBuffer, MemoryRolloutBuffer, RolloutRequest
from skyrl_train.rollout_pipeline import SynchronousRolloutBuffer, SynchronousRolloutPipeline
from skyrl_train.rollout_worker import (
    AsyncRolloutWorkerPool,
    GenerationStalledError,
    LocalRolloutWorker,
    RolloutWorker,
    bind_rollout_worker,
)
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryRunner


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


def test_detached_writer_returns_receipt_without_trajectory_payload(tmp_path):
    async def exercise():
        buffer = FineStoreRolloutBuffer(str(tmp_path / "remote"))
        try:
            writer = pickle.loads(pickle.dumps(buffer.remote_writer()))
            receipt = await writer.write_rollout(_group("remote"))
            assert receipt.uids == ("remote",)
            assert buffer.empty()
            buffer.publish(receipt)
            assert (await buffer.next_batch(1))[0].uid == "remote"
        finally:
            buffer.close()

    asyncio.run(exercise())


def test_detached_writers_commit_concurrently(tmp_path):
    async def exercise():
        buffer = FineStoreRolloutBuffer(str(tmp_path / "concurrent"))
        try:
            writers = [pickle.loads(pickle.dumps(buffer.remote_writer())) for _ in range(8)]
            receipts = await asyncio.gather(
                *(writer.write_rollout(_group(str(index))) for index, writer in enumerate(writers))
            )
            for receipt in receipts:
                buffer.publish(receipt)
            assert [group.uid for group in await buffer.next_batch(8)] == [str(index) for index in range(8)]
        finally:
            buffer.close()

    asyncio.run(exercise())


def test_local_producer_writes_before_returning_receipt():
    class Producer(TrajectoryRunner):
        async def _run(self, _request, disable_tqdm=False):
            return _group("local").trajectory_batch

    async def exercise():
        buffer = MemoryRolloutBuffer()
        request = {"prompts": ["hello"], "env_extras": [{}]}
        receipts = await LocalRolloutWorker(Producer(), buffer.writer()).produce(
            RolloutRequest(request, [{"uid": "local"}], ["local"], 3, "group")
        )
        receipt = receipts[0]
        assert buffer.empty()
        buffer.publish(receipt)
        result = (await buffer.next_batch(1))[0]
        assert result.uid == "local"
        assert result.trajectory_batch["response_ids"] == [[3, 4]]
        assert result.trajectory_batch["rollout_metrics"]["reward_mean"] == 1.0

    asyncio.run(exercise())


def test_memory_backend_rejects_remote_writer():
    with pytest.raises(ValueError, match="remote producer"):
        MemoryRolloutBuffer().remote_writer()


def test_continuous_workers_publish_fast_group_before_straggler_finishes():
    async def exercise():
        published = asyncio.Event()
        release_slow = asyncio.Event()
        slow_started = asyncio.Event()

        class Buffer(MemoryRolloutBuffer):
            def publish(self, receipt):
                super().publish(receipt)
                if receipt.uids == ("fast",):
                    published.set()

        buffer = Buffer()
        prompts = iter(([{"uid": uid}] for uid in ("slow", "fast")))

        class Source:
            async def next_assignment(self):
                try:
                    group = next(prompts)
                except StopIteration as error:
                    raise GenerationStalledError("source exhausted") from error
                return RolloutRequest({}, group, [group[0]["uid"]], 1, "group")

        class Executor:
            async def produce(self, request, disable_tqdm=False):
                uid = request.uids[0]
                if uid == "slow":
                    slow_started.set()
                    await release_slow.wait()
                return [await buffer.writer().write_rollout(_group(uid))]

            async def retain(self, rollout):
                pass

        async def finished():
            pass

        pool = AsyncRolloutWorkerPool(RolloutWorker(Executor(), buffer), Source(), 2, finished)
        tasks = pool.start()
        async with asyncio.timeout(1):
            await published.wait()
        assert slow_started.is_set()
        assert [group.uid for group in await buffer.next_batch(1)] == ["fast"]
        release_slow.set()
        await asyncio.gather(*tasks)
        assert [group.uid for group in await buffer.next_batch(1)] == ["slow"]

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_buffer_contract_preserves_order_and_evidence_across_snapshot(tmp_path, backend):
    def open_buffer(path):
        return MemoryRolloutBuffer() if backend == "memory" else FineStoreRolloutBuffer(str(path))

    async def exercise():
        buffer = open_buffer(tmp_path / "original")
        for uid in ("fast", "slow", "later"):
            buffer.publish(await buffer.writer().write_rollout(_group(uid)))
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
        queues.rollout_buffer.publish(await queues.rollout_buffer.writer().write_rollout(_group("pending")))
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
                await slot.publish(await buffer.writer().write_rollout(_group(str(index))))
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
            await slot.publish(await buffer.writer().write_rollout(_group("first")))
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.01), buffer.request_slot():
                pass
        assert [group.uid for group in await buffer.next_batch(1)] == ["first"]
        await policy.notify_capacity_change(2)
        async with asyncio.timeout(1), buffer.request_slot() as slot:
            await slot.publish(await buffer.writer().write_rollout(_group("second")))
        buffer.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "finestore"])
def test_synchronous_trainer_uses_buffer_contract(tmp_path, backend):
    async def exercise():
        original = _group("example").trajectory_batch

        class Producer:
            async def run(self, request, disable_tqdm=False):
                return original

        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.cfg = SimpleNamespace(
            trainer=SimpleNamespace(
                ckpt_path=str(tmp_path),
                rollout_buffer=SimpleNamespace(backend=backend),
                step_wise_training=True,
                algorithm=SimpleNamespace(use_tis=False, policy_loss_type="grpo", tis_lcs_alert_threshold=0.5),
            )
        )
        trainer.global_step = 7
        trainer._rollout_buffer = None
        trainer.all_metrics = {}
        trainer.trajectory_runner = Producer()
        request = RolloutRequest(
            {"prompts": ["example"], "trajectory_ids": None}, [{"uid": "example"}], ["example"], 7, "batch"
        )

        class Curriculum:
            def __init__(self):
                self.sent = False

            async def next_assignment(self):
                if self.sent:
                    return None
                self.sent = True
                return request

        try:
            storage = await trainer._open_rollout_buffer()
            worker = bind_rollout_worker(trainer.trajectory_runner, storage)
            buffer = SynchronousRolloutBuffer(storage, worker.retain)
            pipeline = SynchronousRolloutPipeline(Curriculum(), buffer, worker)
            pipeline.start()
            buffer.open_slot()
            result = await trainer.read_synchronous_rollout_batch(buffer)
            assert result is not None
            batch, _ = result
            assert batch == original
            if backend == "finestore":
                assert batch is not original
        finally:
            await pipeline.stop()
            if trainer._rollout_buffer is not None:
                trainer._rollout_buffer.close()

    asyncio.run(exercise())
