import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback, EvaluationCallback


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


@pytest.mark.asyncio
async def test_best_checkpoint_keeps_initial_ties_and_skips_invalid_coverage(tmp_path):
    state = TrainerState(global_step=0, epoch=0, total_steps=12, num_steps_per_epoch=12)

    class CheckpointWriter:
        cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))

        async def save_checkpoints(self):
            destination = tmp_path / f"global_step_{state.global_step}" / "policy"
            destination.mkdir(parents=True)
            (destination / "complete").write_text("saved")

    trainer = CheckpointWriter()
    callback = BestCheckpointCallback(initial_model_uri="initial-model", expected_tasks=123, minimum_scored_tasks=111)
    metrics = {"eval/all/num_attempted": 123, "eval/all/num_scored": 123, "eval/all/avg_verifier_score": 35 / 123}
    await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
    path = tmp_path / "best_checkpoint.json"
    assert json.loads(path.read_text())["model_uri"] == "initial-model"
    state.global_step = 4
    await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
    assert json.loads(path.read_text())["step"] == 0
    state.global_step = 8
    metrics.update({"eval/all/num_scored": 100, "eval/all/avg_verifier_score": 1.0})
    await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
    assert json.loads(path.read_text())["step"] == 0
    state.global_step = 12
    metrics.update({"eval/all/num_scored": 123, "eval/all/avg_verifier_score": 40 / 123})
    await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
    best = json.loads(path.read_text())
    assert best["step"] == 12 and best["model_format"] == "native"
    assert Path(best["model_uri"], "complete").read_text() == "saved"
