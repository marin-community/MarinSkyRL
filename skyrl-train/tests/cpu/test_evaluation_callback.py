"""Behavioral tests for evaluation callback scheduling."""

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import EvaluationCallback, create_default_callbacks


@pytest.mark.parametrize(
    ("eval_on_train_end", "eval_steps", "expected"),
    [(True, 5, True), (False, 5, False), (True, 0, False)],
)
def test_evaluation_callback_respects_final_evaluation_configuration(eval_on_train_end, eval_steps, expected):
    callback = EvaluationCallback(eval_steps=eval_steps, eval_on_train_end=eval_on_train_end)
    state = TrainerState(global_step=7, epoch=0, total_steps=7, num_steps_per_epoch=7)
    control = callback.on_train_end(state, TrainerControl())

    assert control.should_evaluate is expected


@pytest.mark.parametrize("explicit_callbacks", [False, True])
@pytest.mark.parametrize("steps", [[0, 5, 25], [5], []])
def test_exact_evaluation_schedule_through_configuration(explicit_callbacks, steps):
    trainer = {
        "eval_interval": 5,
        "eval_before_train": True,
        "eval_at_steps": steps,
        "ckpt_interval": -1,
        "enable_db_registration": False,
    }
    if explicit_callbacks:
        trainer["callbacks"] = [{"type": "evaluation", "eval_steps": 5, "eval_at_steps": steps}]
    callbacks = create_default_callbacks(OmegaConf.create({"trainer": trainer, "generator": {}}))
    observed = []
    for step in range(26):
        state = TrainerState(global_step=step, epoch=0, total_steps=25, num_steps_per_epoch=25)
        control = TrainerControl()
        for callback in callbacks:
            if step == 0:
                callback.on_train_begin(state, control)
            else:
                callback.on_step_end(state, control)
        if control.should_evaluate:
            observed.append(step)
    control = TrainerControl()
    for callback in callbacks:
        callback.on_train_end(state, control)
    assert observed == steps
    assert not control.should_evaluate  # No duplicate final evaluation or unrequested endpoint.


@pytest.mark.parametrize("resume_step", [5, 25])
def test_exact_evaluation_schedule_does_not_repeat_completed_update_on_resume(resume_step):
    callback = EvaluationCallback(eval_steps=-1, eval_at_steps=[0, 5, 25])
    state = TrainerState(global_step=resume_step, epoch=0, total_steps=25, num_steps_per_epoch=25)
    control = callback.on_train_begin(state, TrainerControl())
    assert not control.should_evaluate


@pytest.mark.parametrize("steps", [[-1], [True], [5, 5], [25, 5], [0.5]])
def test_exact_evaluation_schedule_rejects_ambiguous_updates(steps):
    with pytest.raises(ValueError, match="strictly increasing nonnegative integers"):
        EvaluationCallback(eval_at_steps=steps)
