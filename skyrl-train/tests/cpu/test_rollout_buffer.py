"""Synchronous-trainer rollout buffer contract and trainer handoff tests."""

import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.rollout_buffer import FineStoreRolloutBuffer, MemoryRolloutBuffer, RolloutRequest, SynchronousRollout
from skyrl_train.rollout_pipeline import SynchronousRolloutBuffer, SynchronousRolloutPipeline
from skyrl_train.rollout_worker import LocalRolloutWorker, bind_rollout_worker
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryRunner


def _group(uid: str) -> SynchronousRollout:
    return SynchronousRollout(
        uids=[uid],
        model_step=3,
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


def test_local_producer_writes_before_returning_receipt():
    class Producer(TrajectoryRunner):
        async def _run(self, _request, disable_tqdm=False):
            return _group("local").trajectory_batch

    async def exercise():
        buffer = MemoryRolloutBuffer()
        request = {"prompts": ["hello"], "env_extras": [{}]}
        receipts = await LocalRolloutWorker(Producer(), buffer.writer()).produce(
            RolloutRequest(request, [{"uid": "local"}], ["local"], 3)
        )
        receipt = receipts[0]
        assert buffer.empty()
        buffer.publish(receipt)
        result = (await buffer.next_batch(1))[0]
        assert result.uids == ["local"]
        assert result.trajectory_batch["response_ids"] == [[3, 4]]
        assert result.trajectory_batch["rollout_metrics"]["reward_mean"] == 1.0

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
                assert [group.uids[0] for group in first] == ["fast", "slow", "later"]
            else:
                second = await restored.next_batch(2)
                assert [group.uids[0] for group in first + second] == ["fast", "slow", "later"]
            assert first[0].trajectory_batch == _group("fast").trajectory_batch
            assert restored.empty()
        finally:
            restored.close()

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
        request = RolloutRequest({"prompts": ["example"], "trajectory_ids": None}, [{"uid": "example"}], ["example"], 7)

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
            buffer = SynchronousRolloutBuffer(storage)
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
