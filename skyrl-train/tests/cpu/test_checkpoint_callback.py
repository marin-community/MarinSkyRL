"""Behavioral tests for checkpoint callback scheduling."""

import pytest

from skyrl_train.callbacks import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import CheckpointCallback


@pytest.mark.parametrize(
    ("save_on_train_end", "save_steps", "expected"),
    [(True, 5, True), (False, 5, False), (True, -1, True)],
)
def test_checkpoint_callback_respects_final_checkpoint_configuration(save_on_train_end, save_steps, expected):
    callback = CheckpointCallback(save_steps=save_steps, save_on_train_end=save_on_train_end)
    state = TrainerState(global_step=7, epoch=0, total_steps=7, num_steps_per_epoch=7)
    control = callback.on_train_end(state, TrainerControl())

    assert control.should_save is expected
