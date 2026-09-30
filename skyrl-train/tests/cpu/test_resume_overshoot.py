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

These tests exercise the finalize handler directly, without booting Ray or models.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.callbacks.base import TrainerControl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bare_trainer(global_step: int, total_training_steps: int, colocate_all: bool = False):
    """Construct a trainer instance without running the heavy __init__.

    We bypass __init__ (which builds dataloaders, Ray actor groups, etc.) and set
    only the attributes the resume guard / finalize handler touch.
    """
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._last_saved_step = None
    trainer._pending_checkpoint_upload = None
    trainer.global_step = global_step
    trainer.total_training_steps = total_training_steps
    trainer.colocate_all = colocate_all
    trainer.num_steps_per_epoch = max(total_training_steps, 1)
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._control = TrainerControl()
    trainer.eval_dataset = object()

    # epochs is read from cfg in _handle_resume_at_max_steps
    cfg = MagicMock()
    cfg.trainer.epochs = 1
    trainer.cfg = cfg

    trainer.context = MagicMock(name="context")
    trainer.context.state_dict = AsyncMock(name="state_dict")

    trainer._snapshot_checkpoint = MagicMock(name="snapshot_checkpoint")
    trainer._finish_checkpoint_upload = AsyncMock(name="finish_checkpoint_upload", return_value=(0.0, 0.0))
    trainer.handle_hf_export = MagicMock(name="handle_hf_export")
    trainer.eval = AsyncMock(name="eval", return_value={"eval/accuracy": 0.75})
    trainer._log_metrics_stdout = MagicMock(name="_log_metrics_stdout")
    trainer.tracker = MagicMock(name="tracker")
    trainer.policy_model = MagicMock(name="policy_model")
    trainer.inference_engine_client = MagicMock(name="inference_engine_client")
    trainer.inference_engine_client.sleep = AsyncMock(name="sleep")
    return trainer


class _RecordingCallbackHandler:
    """Minimal async callback handler that records events and applies a control delta."""

    def __init__(self, on_train_end_control: TrainerControl):
        self.events = []
        self.states = []
        self._on_train_end_control = on_train_end_control

    async def call_event_async(self, event, state, control, **kwargs):
        self.events.append(event)
        self.states.append(state)
        if event == "on_train_end":
            return self._on_train_end_control
        return control


def test_train_end_saves_the_last_completed_step():
    """Final callbacks and artifacts must use completed steps, not the next step index."""
    trainer = _make_bare_trainer(global_step=17, total_training_steps=16)
    requested = TrainerControl()
    requested.should_save = True
    requested.should_save_hf_model = True
    trainer.callback_handler = _RecordingCallbackHandler(requested)
    saved_steps = []
    trainer._snapshot_checkpoint.side_effect = lambda _rollout_state: saved_steps.append(trainer.global_step)
    trainer.handle_hf_export.side_effect = lambda: saved_steps.append(trainer.global_step)

    asyncio.run(trainer._finalize_training(completed_step=16, epoch=0))

    assert trainer.global_step == 16
    assert trainer.callback_handler.states[0].global_step == 16
    assert saved_steps == [16, 16]
    assert trainer.callback_handler.events == ["on_train_end", "on_save"]


def test_train_end_runs_and_logs_the_requested_evaluation():
    trainer = _make_bare_trainer(global_step=17, total_training_steps=16)
    requested = TrainerControl()
    requested.should_evaluate = True
    trainer.callback_handler = _RecordingCallbackHandler(requested)

    asyncio.run(trainer._finalize_training(completed_step=16, epoch=0))

    trainer.eval.assert_awaited_once()
    trainer._log_metrics_stdout.assert_called_once_with({"eval/accuracy": 0.75}, step=16, kind="eval")
    trainer.tracker.log.assert_called_once_with({"eval/accuracy": 0.75}, step=16, commit=True)
    assert trainer.callback_handler.events == ["on_train_end", "on_evaluate"]


def test_train_end_still_saves_when_the_final_evaluation_fails():
    trainer = _make_bare_trainer(global_step=17, total_training_steps=16)
    requested = TrainerControl()
    requested.should_evaluate = True
    requested.should_save = True
    trainer.callback_handler = _RecordingCallbackHandler(requested)
    trainer.eval.side_effect = RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        asyncio.run(trainer._finalize_training(completed_step=16, epoch=0))

    trainer._snapshot_checkpoint.assert_called_once()
    assert trainer.callback_handler.events == ["on_train_end", "on_save"]


@pytest.mark.parametrize(
    ("should_save", "should_save_hf_model", "colocate_all", "expected_effects"),
    [
        (True, True, False, ["checkpoint@80", "hf_export@80"]),
        (False, False, False, []),
        (False, False, True, ["engines_asleep", "policy_backloaded"]),
    ],
)
def test_resume_at_max_steps_finalizes_without_training(
    should_save, should_save_hf_model, colocate_all, expected_effects
):
    """A run resumed at max_steps fires on_train_end and performs only the requested final artifacts."""
    trainer = _make_bare_trainer(global_step=80, total_training_steps=80, colocate_all=colocate_all)
    requested = TrainerControl()
    requested.should_save = should_save
    requested.should_save_hf_model = should_save_hf_model
    trainer.callback_handler = _RecordingCallbackHandler(requested)
    effects = []
    trainer._snapshot_checkpoint.side_effect = lambda _state: effects.append(f"checkpoint@{trainer.global_step}")
    trainer.handle_hf_export.side_effect = lambda: effects.append(f"hf_export@{trainer.global_step}")
    trainer.inference_engine_client.sleep.side_effect = lambda: effects.append("engines_asleep")
    trainer.policy_model.backload_to_gpu.side_effect = lambda: effects.append("policy_backloaded")

    asyncio.run(trainer._handle_resume_at_max_steps())

    assert trainer.callback_handler.events[0] == "on_train_end"
    assert trainer.callback_handler.states[0].global_step == 80
    assert effects == expected_effects
