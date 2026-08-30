"""Behavioral tests for evaluation callback scheduling."""

import pytest

from skyrl_train.callbacks import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import EvaluationCallback


@pytest.mark.parametrize(
    ("eval_on_train_end", "eval_steps", "expected"),
    [(True, 5, True), (False, 5, False), (True, 0, False)],
)
def test_evaluation_callback_respects_final_evaluation_configuration(eval_on_train_end, eval_steps, expected):
    callback = EvaluationCallback(eval_steps=eval_steps, eval_on_train_end=eval_on_train_end)
    state = TrainerState(global_step=7, epoch=0, total_steps=7, num_steps_per_epoch=7)
    control = callback.on_train_end(state, TrainerControl())

    assert control.should_evaluate is expected
