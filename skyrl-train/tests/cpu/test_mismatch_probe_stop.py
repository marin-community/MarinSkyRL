from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl_train.callbacks.base import TrainerControl
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode


@pytest.mark.asyncio
async def test_zero_update_probe_finalizes_without_training():
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(algorithm=SimpleNamespace(use_kl_in_reward=False)))
    trainer.resume_mode = ResumeMode.NONE
    trainer.colocate_all = False
    trainer.global_step = 0
    trainer.total_training_steps = 1
    trainer.num_steps_per_epoch = 1
    trainer.context = SimpleNamespace(start=MagicMock(side_effect=AssertionError("training batch was fetched")))
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer._control = TrainerControl()
    trainer._init_weight_sync = AsyncMock()
    trainer._record_run_configuration = MagicMock()
    trainer._start_draft_trainer = AsyncMock()
    trainer._sync_policy_for_rollouts = AsyncMock()
    trainer._log_startup_timings = MagicMock()
    trainer._finalize_training = AsyncMock()
    trainer.generate = AsyncMock(side_effect=AssertionError("training batch was fetched"))

    class StopAtBegin:
        async def call_event_async(self, event, state, control, **kwargs):
            assert event == "on_train_begin"
            assert state.global_step == 0
            control.should_training_stop = True
            return control

    trainer.callback_handler = StopAtBegin()
    await trainer._train_loop()
    trainer._finalize_training.assert_awaited_once_with(completed_step=0, epoch=0)
    trainer.generate.assert_not_awaited()
