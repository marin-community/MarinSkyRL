"""Unit tests for the resume-overshoot guard.

Regression tests for the long-standing "resume-overshoot trap": a run resumed at
(or past) ``max_steps`` used to execute one spurious extra training step (gs N+1)
before the post-increment ``max_steps`` check fired, wasting a node slot and
typically ending FAILED — which kept the ``afternotok`` restart chain alive and
spawned more overshoot links.

The fix adds a guard right after checkpoint load:

    if self.global_step >= self.total_training_steps:
        await self._handle_resume_at_max_steps()
        return

``_handle_resume_at_max_steps`` fires ``on_train_end`` callbacks (so a missing
final checkpoint / HF export still runs) and returns without training, so the
process exits 0 (clean COMPLETED).

These tests exercise the decision logic and the finalize handler directly,
without booting Ray / models, so they run on CPU.

    uv run --isolated --extra dev pytest tests/cpu/test_resume_overshoot.py
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.callbacks import TrainerControl, TrainerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resumed_at_max_is_complete(global_step: int, total_training_steps: int) -> bool:
    """Mirror of the guard predicate used in both trainers' _train_loop.

    A run is COMPLETE (must exit without another step) iff the checkpoint it
    resumed from was already at or past max_steps. The loaded global_step is the
    *completed* step count (save_checkpoints writes it after a step finishes), so
    the boundary is ``>=`` — "resumed exactly at max_steps" counts as done.
    """
    return global_step >= total_training_steps


def _make_bare_trainer(cls, global_step: int, total_training_steps: int, colocate_all: bool = False):
    """Construct a trainer instance without running the heavy __init__.

    We bypass __init__ (which builds dataloaders, Ray actor groups, etc.) and set
    only the attributes the resume guard / finalize handler touch.
    """
    trainer = cls.__new__(cls)
    trainer.global_step = global_step
    trainer.total_training_steps = total_training_steps
    trainer.colocate_all = colocate_all
    trainer.num_steps_per_epoch = max(total_training_steps, 1)
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._control = TrainerControl()

    # epochs is read from cfg in _handle_resume_at_max_steps
    cfg = MagicMock()
    cfg.trainer.epochs = 1
    trainer.cfg = cfg

    # _create_trainer_state for the base trainer reads len(self.train_dataloader);
    # the fully-async override reads self.num_steps_per_epoch instead.
    dl = MagicMock()
    dl.__len__ = lambda _self: max(total_training_steps, 1)
    trainer.train_dataloader = dl

    trainer.save_checkpoints = MagicMock(name="save_checkpoints")
    trainer.save_models = MagicMock(name="save_models")
    trainer.policy_model = MagicMock(name="policy_model")
    return trainer


class _RecordingCallbackHandler:
    """Minimal async callback handler that records events and applies a control delta."""

    def __init__(self, on_train_end_control: TrainerControl):
        self.events = []
        self._on_train_end_control = on_train_end_control

    async def call_event_async(self, event, state, control, **kwargs):
        self.events.append(event)
        if event == "on_train_end":
            return self._on_train_end_control
        return control


# ---------------------------------------------------------------------------
# Guard-predicate tests (the core termination/resume-at-max condition)
# ---------------------------------------------------------------------------


def test_guard_predicate_resumed_at_max_is_complete():
    # Resumed exactly AT max_steps -> done (no-op exit).
    assert _resumed_at_max_is_complete(global_step=80, total_training_steps=80) is True


def test_guard_predicate_resumed_past_max_is_complete():
    # Resumed PAST max_steps (e.g. max_steps lowered) -> done.
    assert _resumed_at_max_is_complete(global_step=81, total_training_steps=80) is True


def test_guard_predicate_resumed_below_max_continues():
    # Mid-training resume -> NOT complete, continue normally.
    assert _resumed_at_max_is_complete(global_step=79, total_training_steps=80) is False
    assert _resumed_at_max_is_complete(global_step=1, total_training_steps=80) is False


def test_guard_predicate_fresh_run_not_complete():
    # Fresh run (global_step == 0) is never treated as complete on load.
    assert _resumed_at_max_is_complete(global_step=0, total_training_steps=80) is False


def test_fresh_run_stops_at_exactly_max_steps():
    """Simulate the fresh-run loop arithmetic: it must stop at exactly max_steps.

    A fresh run starts global_step=0, increments to 1, trains steps 1..N, and after
    the post-step increment the check ``global_step > total_training_steps`` fires.
    The last step actually *trained* (and checkpointed) must be exactly N — no gsN+1.
    """
    total = 80
    global_step = 0
    global_step += 1  # start training at global_step 1
    trained_steps = []
    while True:
        trained_steps.append(global_step)  # this step is executed/checkpointed
        global_step += 1  # post-step increment
        if global_step > total:  # max_steps check (post-increment)
            break
    assert trained_steps[-1] == total, "fresh run must stop exactly at max_steps"
    assert max(trained_steps) == total, "fresh run must NOT train gsN+1 (no overshoot)"
    assert len(trained_steps) == total


# ---------------------------------------------------------------------------
# _handle_resume_at_max_steps finalize-handler tests (both trainers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [RayPPOTrainer, FullyAsyncRayPPOTrainer])
def test_handle_resume_at_max_steps_triggers_export_when_requested(cls):
    """on_train_end requests a save+HF export -> finalize handler performs both,
    and never runs a training step."""
    trainer = _make_bare_trainer(cls, global_step=80, total_training_steps=80)

    requested = TrainerControl()
    requested.should_save = True
    requested.should_save_hf_model = True
    trainer.callback_handler = _RecordingCallbackHandler(requested)

    asyncio.run(trainer._handle_resume_at_max_steps())

    assert "on_train_end" in trainer.callback_handler.events
    trainer.save_checkpoints.assert_called_once()
    trainer.save_models.assert_called_once()


@pytest.mark.parametrize("cls", [RayPPOTrainer, FullyAsyncRayPPOTrainer])
def test_handle_resume_at_max_steps_no_save_when_not_requested(cls):
    """If callbacks request no final save, the handler is still a clean no-op exit
    (it does not raise and does not invent a save)."""
    trainer = _make_bare_trainer(cls, global_step=80, total_training_steps=80)
    trainer.callback_handler = _RecordingCallbackHandler(TrainerControl())  # nothing requested

    asyncio.run(trainer._handle_resume_at_max_steps())

    trainer.save_checkpoints.assert_not_called()
    trainer.save_models.assert_not_called()


def test_handle_resume_at_max_steps_backloads_when_colocate(monkeypatch):
    """Base trainer with colocate_all=True backloads the policy model to GPU before
    finalize (mirrors the normal end-of-training path)."""
    trainer = _make_bare_trainer(RayPPOTrainer, global_step=80, total_training_steps=80, colocate_all=True)
    trainer.callback_handler = _RecordingCallbackHandler(TrainerControl())

    asyncio.run(trainer._handle_resume_at_max_steps())

    trainer.policy_model.backload_to_gpu.assert_called_once()


# ---------------------------------------------------------------------------
# Sanity: the fully-async trainer's _create_trainer_state used inside the handler
# does not require the heavy base attributes.
# ---------------------------------------------------------------------------


def test_fully_async_create_trainer_state_smoke():
    trainer = _make_bare_trainer(FullyAsyncRayPPOTrainer, global_step=80, total_training_steps=80)
    state = trainer._create_trainer_state(epoch=0)
    assert isinstance(state, TrainerState)
    assert state.global_step == 80
    assert state.total_steps == 80
    assert state.is_last_step is True
