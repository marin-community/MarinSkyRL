import pytest

from skyrl_train.callbacks.base import TrainerControl, TrainerState
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


@pytest.mark.asyncio
@pytest.mark.parametrize("requirement", [{"minimum": 0.65}, {"min_improvement": 0.4}])
async def test_evaluation_stops_at_the_first_qualifying_score(requirement):
    callback = EvaluationCallback(stop_when={"eval/score": requirement})
    state = TrainerState(global_step=0, epoch=0, total_steps=30, num_steps_per_epoch=30)
    for step, score, stopped in ((0, 0.25, False), (5, 0.64, False), (10, 0.65, True)):
        state.global_step = step
        result = await callback.on_evaluate_async(state, TrainerControl(), metrics={"eval/score": score}, trainer=None)
        assert result.should_training_stop is stopped


@pytest.mark.parametrize("total,step,expected", [(249, 20, False), (260, 20, True), (270, 10, False)])
def test_token_milestone_evaluates_and_saves_at_the_actual_consumed_count(total, step, expected):
    callback = EvaluationCallback(eval_steps=2, eval_loss_tokens=250)
    state = TrainerState(
        global_step=3,
        epoch=0,
        total_steps=100,
        num_steps_per_epoch=100,
        metrics={"consumed/loss_total": total, "consumed/loss_step": step},
    )
    control = callback.on_step_end(state, TrainerControl())
    assert control.should_evaluate is expected
    assert control.should_save is expected
