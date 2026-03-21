"""Unit tests for BufferCheckpointCallback save/restore roundtrip.

Run with: uv run --isolated --extra dev pytest tests/cpu/test_buffer_checkpoint.py

These tests replicate the minimal dataclass/TypedDict shapes to avoid importing
the full skyrl_train package (which requires skyrl_gym and GPU deps).
"""

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Replicate minimal types to avoid heavy imports through generators/__init__.py
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryID:
    instance_id: str
    repetition_id: int

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class GeneratedOutputGroup:
    generator_output: dict  # GeneratorOutput is a TypedDict
    uid: str
    global_step_when_scheduled: int


# ---------------------------------------------------------------------------
# Monkey-patch so BufferCheckpointCallback.load_buffer_items can import these
# from "skyrl_train.fully_async_trainer" and "skyrl_train.generators.base"
# without pulling in the real module tree.
# ---------------------------------------------------------------------------

import types

_fake_base = types.ModuleType("skyrl_train.generators.base")
_fake_base.TrajectoryID = TrajectoryID
_fake_base.GeneratorOutput = dict
sys.modules.setdefault("skyrl_train.generators.base", _fake_base)

_fake_generators = types.ModuleType("skyrl_train.generators")
sys.modules.setdefault("skyrl_train.generators", _fake_generators)

_fake_fat = types.ModuleType("skyrl_train.fully_async_trainer")
_fake_fat.GeneratedOutputGroup = GeneratedOutputGroup
sys.modules.setdefault("skyrl_train.fully_async_trainer", _fake_fat)

# Now we can safely import the callback (its own imports are all lightweight)
from skyrl_train.callbacks.builtin import BufferCheckpointCallback


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
    )


class _FakeTrainer:
    def __init__(self, ckpt_path, buffer):
        class _Cfg:
            class trainer:
                ckpt_path = None
        self.cfg = _Cfg()
        self.cfg.trainer.ckpt_path = ckpt_path
        self._generation_output_group_buffer = buffer


class _FakeState:
    def __init__(self, step):
        self.global_step = step


class _FakeControl:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_roundtrip_empty_buffer():
    """Empty buffer produces no artifact file."""
    buf = asyncio.Queue(maxsize=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        step_dir = os.path.join(tmpdir, "global_step_10")
        os.makedirs(step_dir)
        trainer = _FakeTrainer(tmpdir, buf)
        cb = BufferCheckpointCallback()
        cb.on_save(_FakeState(10), _FakeControl(), trainer=trainer)
        assert not os.path.exists(os.path.join(step_dir, cb.ARTIFACT_NAME))


def test_roundtrip_with_items():
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
        cb.on_save(_FakeState(5), _FakeControl(), trainer=trainer)

        # Buffer should still have all 3 items (non-destructive)
        assert buf.qsize() == 3

        # Artifact should exist
        artifact_path = os.path.join(step_dir, cb.ARTIFACT_NAME)
        assert os.path.exists(artifact_path)

        # Load and verify
        loaded = BufferCheckpointCallback.load_buffer_items(step_dir)
        assert len(loaded) == 3
        for i, item in enumerate(loaded):
            assert item.uid == f"uid_{i}"
            assert item.global_step_when_scheduled == 5
            assert item.generator_output["prompt_token_ids"] == [[1, 2, 3]]
            assert item.generator_output["rewards"] == [1.0]
            tid = item.generator_output["trajectory_ids"][0]
            assert tid.instance_id == f"uid_{i}"


def test_load_missing_file():
    """Missing artifact returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = BufferCheckpointCallback.load_buffer_items(tmpdir)
        assert loaded == []


def test_restore_into_queue():
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
        cb.on_save(_FakeState(3), _FakeControl(), trainer=trainer)

        # Simulate resume: load into a fresh queue
        new_buf = asyncio.Queue(maxsize=8)
        loaded = BufferCheckpointCallback.load_buffer_items(step_dir)
        for item in loaded:
            new_buf.put_nowait(item)
        assert new_buf.qsize() == 4


def test_no_trainer_in_kwargs():
    """on_save gracefully returns control when no trainer is provided."""
    cb = BufferCheckpointCallback()
    ctrl = _FakeControl()
    result = cb.on_save(_FakeState(1), ctrl)
    assert result is ctrl
