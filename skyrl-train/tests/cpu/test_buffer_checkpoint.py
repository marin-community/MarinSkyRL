"""Unit tests for BufferCheckpointCallback save/restore roundtrip.

Run with: uv run --isolated --group dev --extra cpu pytest tests/cpu/test_buffer_checkpoint.py
"""

import asyncio
import os
import tempfile
import pytest
import torch

from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.fully_async_trainer import _GenerationQueues
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.utils.io import io


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(uid: str, step: int) -> GeneratedOutputGroup:
    gen_out = {
        "prompt_token_ids": [[1, 2, 3]],
        "response_ids": [[4, 5, 6]],
        "rewards": [1.0],
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["eos"],
        "rollout_metrics": {"reward_mean": 1.0},
        "rollout_logprobs": [[-0.1, -0.2, -0.3]],
        "trajectory_ids": [TrajectoryID(instance_id=uid, repetition_id=0)],
        "is_last_step": [True],
        "exclude_from_baseline": [False],
        "actual_global_step": step,
    }
    return GeneratedOutputGroup(
        trajectory_batch=gen_out,
        uid=uid,
        earliest_model_step=step,
        source_prompts=[{"uid": uid}],
    )


class _FakeTrainer:
    def __init__(self, ckpt_path, buffer):
        class _Cfg:
            class trainer:
                ckpt_path = None

        self.cfg = _Cfg()
        self.cfg.trainer.ckpt_path = ckpt_path
        self._generation_queues = _GenerationQueues(
            completed=buffer,
            retries=asyncio.Queue(),
            condition=asyncio.Condition(),
        )


class _FakeState:
    def __init__(self, step):
        self.global_step = step


class _FakeControl:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_roundtrip_empty_buffer():
    """Empty buffer produces no artifact file."""
    buf = asyncio.Queue(maxsize=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        step_dir = os.path.join(tmpdir, "global_step_10")
        os.makedirs(step_dir)
        trainer = _FakeTrainer(tmpdir, buf)
        cb = BufferCheckpointCallback()
        cb.bind_queues(trainer._generation_queues)
        await cb.on_save_async(_FakeState(10), _FakeControl(), trainer=trainer)
        assert not os.path.exists(os.path.join(step_dir, cb.ARTIFACT_NAME))


@pytest.mark.asyncio
async def test_roundtrip_with_items():
    """Items survive save -> load roundtrip and queue is non-destructively snapshotted."""
    buf = asyncio.Queue(maxsize=8)
    items = [_make_item(f"uid_{i}", step=5) for i in range(3)]
    for item in items:
        buf.put_nowait(item)

    with tempfile.TemporaryDirectory() as tmpdir:
        step_dir = os.path.join(tmpdir, "global_step_5")
        os.makedirs(step_dir)
        trainer = _FakeTrainer(tmpdir, buf)
        cb = BufferCheckpointCallback()
        cb.bind_queues(trainer._generation_queues)
        await cb.on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)

        # Buffer should still have all 3 items (non-destructive)
        assert buf.qsize() == 3

        # Artifact should exist
        artifact_path = os.path.join(step_dir, cb.ARTIFACT_NAME)
        assert os.path.exists(artifact_path)

        # Load and verify
        buffer_state = BufferCheckpointCallback.load_buffer_state(step_dir)
        assert len(buffer_state.completed_groups) == 3
        assert buffer_state.retry_prompts == []
        for i, item in enumerate(buffer_state.completed_groups):
            assert item.uid == f"uid_{i}"
            assert item.earliest_model_step == 5
            assert item.source_prompts == [{"uid": f"uid_{i}"}]
            assert item.trajectory_batch["prompt_token_ids"] == [[1, 2, 3]]
            assert item.trajectory_batch["rewards"] == [1.0]
            tid = item.trajectory_batch["trajectory_ids"][0]
            assert tid.instance_id == f"uid_{i}"


@pytest.mark.asyncio
async def test_roundtrip_with_pending_retry():
    buf = asyncio.Queue(maxsize=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        step_dir = os.path.join(tmpdir, "global_step_5")
        os.makedirs(step_dir)
        trainer = _FakeTrainer(tmpdir, buf)
        trainer._generation_queues.retries.put_nowait([{"uid": "retry-me"}])
        cb = BufferCheckpointCallback()
        cb.bind_queues(trainer._generation_queues)

        await cb.on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)
        buffer_state = BufferCheckpointCallback.load_buffer_state(step_dir)

        assert buffer_state.completed_groups == []
        assert buffer_state.retry_prompts == [[{"uid": "retry-me"}]]
        assert trainer._generation_queues.retries.get_nowait() == [{"uid": "retry-me"}]


@pytest.mark.asyncio
async def test_save_failure_is_not_downgraded(monkeypatch, tmp_path):
    buffer = asyncio.Queue(maxsize=1)
    buffer.put_nowait(_make_item("uid", step=5))
    trainer = _FakeTrainer(str(tmp_path), buffer)
    callback = BufferCheckpointCallback()
    callback.bind_queues(trainer._generation_queues)

    def fail_open(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(io, "open_file", fail_open)

    with pytest.raises(OSError, match="storage unavailable"):
        await callback.on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)


def test_load_missing_file():
    """Missing artifacts return empty completed and retry collections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer_state = BufferCheckpointCallback.load_buffer_state(tmpdir)
        assert buffer_state.completed_groups == []
        assert buffer_state.retry_prompts == []


def test_load_malformed_state_fails_instead_of_dropping_retries():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, BufferCheckpointCallback.ARTIFACT_NAME)
        torch.save({"retry_prompts": [[{"uid": "retry-me"}]]}, artifact_path)

        with pytest.raises(KeyError, match="completed_groups"):
            BufferCheckpointCallback.load_buffer_state(tmpdir)


@pytest.mark.asyncio
async def test_no_trainer_in_kwargs():
    """on_save_async fails closed when trainer context is missing."""
    cb = BufferCheckpointCallback()
    with pytest.raises(RuntimeError, match="requires trainer context"):
        await cb.on_save_async(_FakeState(1), _FakeControl())
