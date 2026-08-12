"""Unit tests for BufferCheckpointCallback save/restore roundtrip.

Run with: uv run --isolated --group dev --extra cpu pytest tests/cpu/test_buffer_checkpoint.py
"""

import asyncio
import os
import tempfile
import pytest
import torch

from skyrl_train.callbacks.builtin import BufferCheckpointCallback
from skyrl_train.fully_async_trainer import GeneratedOutputGroup, _GenerationQueues
from skyrl_train.generators.base import TrajectoryID


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
        generator_output=gen_out,
        uid=uid,
        global_step_when_scheduled=step,
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
        await cb.on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)

        # Buffer should still have all 3 items (non-destructive)
        assert buf.qsize() == 3

        # Artifact should exist
        artifact_path = os.path.join(step_dir, cb.ARTIFACT_NAME)
        assert os.path.exists(artifact_path)

        # Load and verify
        loaded, retry_prompts = BufferCheckpointCallback.load_buffer_state(step_dir)
        assert len(loaded) == 3
        assert retry_prompts == []
        for i, item in enumerate(loaded):
            assert item.uid == f"uid_{i}"
            assert item.global_step_when_scheduled == 5
            assert item.source_prompts == [{"uid": f"uid_{i}"}]
            assert item.generator_output["prompt_token_ids"] == [[1, 2, 3]]
            assert item.generator_output["rewards"] == [1.0]
            tid = item.generator_output["trajectory_ids"][0]
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

        await cb.on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)
        completed, retry_prompts = BufferCheckpointCallback.load_buffer_state(step_dir)

        assert completed == []
        assert retry_prompts == [[{"uid": "retry-me"}]]
        assert trainer._generation_queues.retries.get_nowait() == [{"uid": "retry-me"}]


@pytest.mark.asyncio
async def test_save_failure_is_not_downgraded(monkeypatch, tmp_path):
    from skyrl_train.utils.io import io

    buffer = asyncio.Queue(maxsize=1)
    buffer.put_nowait(_make_item("uid", step=5))
    trainer = _FakeTrainer(str(tmp_path), buffer)

    def fail_open(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(io, "open_file", fail_open)

    with pytest.raises(OSError, match="storage unavailable"):
        await BufferCheckpointCallback().on_save_async(_FakeState(5), _FakeControl(), trainer=trainer)


def test_load_missing_file():
    """Missing artifacts return empty completed and retry collections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert BufferCheckpointCallback.load_buffer_state(tmpdir) == ([], [])


def test_load_malformed_state_fails_instead_of_dropping_retries():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, BufferCheckpointCallback.ARTIFACT_NAME)
        torch.save({"retry_prompts": [[{"uid": "retry-me"}]]}, artifact_path)

        with pytest.raises(KeyError, match="completed_groups"):
            BufferCheckpointCallback.load_buffer_state(tmpdir)


@pytest.mark.asyncio
async def test_restore_into_queue():
    """Loaded items can be put back into a fresh queue."""
    buf = asyncio.Queue(maxsize=8)
    items = [_make_item(f"uid_{i}", step=3) for i in range(4)]
    for item in items:
        buf.put_nowait(item)

    with tempfile.TemporaryDirectory() as tmpdir:
        step_dir = os.path.join(tmpdir, "global_step_3")
        os.makedirs(step_dir)
        trainer = _FakeTrainer(tmpdir, buf)
        cb = BufferCheckpointCallback()
        await cb.on_save_async(_FakeState(3), _FakeControl(), trainer=trainer)

        # Simulate resume: load into a fresh queue
        new_buf = asyncio.Queue(maxsize=8)
        loaded, retry_prompts = BufferCheckpointCallback.load_buffer_state(step_dir)
        for item in loaded:
            new_buf.put_nowait(item)
        assert new_buf.qsize() == 4
        assert retry_prompts == []


@pytest.mark.asyncio
async def test_no_trainer_in_kwargs():
    """on_save gracefully returns control when no trainer is provided."""
    cb = BufferCheckpointCallback()
    ctrl = _FakeControl()
    result = await cb.on_save_async(_FakeState(1), ctrl)
    assert result is ctrl
