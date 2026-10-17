import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.checkpoint_generation import commit_attempt, new_attempt_path
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import BestCheckpointCallback, EvaluationCallback, HarborEvaluationCallback


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
@pytest.mark.parametrize("committed_attempt", [False, True])
async def test_best_checkpoint_keeps_initial_ties_and_skips_invalid_coverage(tmp_path, committed_attempt):
    state = TrainerState(global_step=0, epoch=0, total_steps=12, num_steps_per_epoch=12)

    class CheckpointWriter:
        cfg = SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path)))

        async def save_checkpoints(self):
            step = tmp_path / f"global_step_{state.global_step}"
            payload = Path(new_attempt_path(str(step))) if committed_attempt else step
            destination = payload / "policy"
            destination.mkdir(parents=True)
            (destination / "complete").write_text("saved")
            (payload / "trainer_state.pt").write_bytes(b"trainer state")
            if committed_attempt:
                commit_attempt(str(step), str(payload), required_files={"trainer_state.pt", "policy/complete"})

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
    # A resumed callback must not replace the retained winner with a worse checkpoint.
    resumed = BestCheckpointCallback(initial_model_uri="initial-model", expected_tasks=123, minimum_scored_tasks=111)
    state.global_step = 16
    metrics["eval/all/avg_verifier_score"] = 36 / 123
    await resumed.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
    assert json.loads(path.read_text()) == best
    assert not (tmp_path / "global_step_16").exists()


@pytest.mark.asyncio
async def test_harbor_scores_join_native_panel_and_incomplete_cohort_cannot_win(tmp_path, monkeypatch):
    state = TrainerState(global_step=0, epoch=0, total_steps=20, num_steps_per_epoch=20)
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(ckpt_path=str(tmp_path), export_path=str(tmp_path / "traces"))),
        inference_engine_client=SimpleNamespace(
            enable_http_endpoint=True,
            model_name="current-policy",
            http_endpoint_host="127.0.0.1",
            http_endpoint_port=8000,
        ),
    )
    monkeypatch.setenv("DAYTONA_RL_API_KEY", "test-rl-credential")
    monkeypatch.setenv("DAYTONA_API_KEY", "test-rl-credential")
    monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "test-eval-credential")
    scored = 30

    callback = HarborEvaluationCallback(
        cohorts=OmegaConf.create([{"name": "benchmark", "job": {"agents": [{"name": "terminus-2"}]}}]),
        timeout=10,
        minimum_coverage=0.9,
        daytona_api_key_env="DAYTONA_EVAL_API_KEY",
    )
    best = BestCheckpointCallback(initial_model_uri="initializer", expected_tasks=1000, minimum_scored_tasks=950)
    for scored, expected_eligible in [(30, 1.0), (20, 0.0)]:
        metrics = {"eval/all/num_attempted": 970, "eval/all/num_scored": 970, "eval/all/avg_verifier_score": 0.5}
        metrics.update(
            {
                "eval/harbor/benchmark/num_attempted": 30,
                "eval/harbor/benchmark/num_scored": scored,
                "eval/harbor/benchmark/reward_sum": 15,
                "eval/harbor/benchmark/avg_verifier_score": 15 / scored,
            }
        )
        await callback.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
        assert os.environ["DAYTONA_API_KEY"] == "test-rl-credential"
        assert os.environ["DAYTONA_RL_API_KEY"] == "test-rl-credential"
        assert metrics["eval/all/num_attempted"] == 1000
        assert metrics["eval/all/num_scored"] == 970 + scored
        assert metrics["eval/all/avg_verifier_score"] == pytest.approx(500 / (970 + scored))
        assert metrics["eval/selection_eligible"] == expected_eligible
        await best.on_evaluate_async(state, TrainerControl(), metrics=metrics, trainer=trainer)
        assert json.loads((tmp_path / "best_checkpoint.json").read_text())["accuracy"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("harbor_scored", [0, 2])
async def test_harbor_panel_preserves_missing_native_scores(harbor_scored):
    callback = HarborEvaluationCallback(
        cohorts=[{"name": "benchmark"}], timeout=10, minimum_coverage=0.9, daytona_api_key_env="unused"
    )
    metrics = {
        "eval/all/num_attempted": 4,
        "eval/all/num_scored": 0,
        "eval/harbor/benchmark/num_attempted": 2,
        "eval/harbor/benchmark/num_scored": harbor_scored,
        "eval/harbor/benchmark/reward_sum": harbor_scored / 2,
    }
    await callback.on_evaluate_async(None, TrainerControl(), metrics=metrics, trainer=None)
    assert metrics["eval/all/num_attempted"] == 6
    assert metrics["eval/all/num_scored"] == harbor_scored
    if harbor_scored:
        assert metrics["eval/all/avg_verifier_score"] == 0.5
    else:
        assert "eval/all/avg_verifier_score" not in metrics
        assert metrics["eval/selection_eligible"] == 0.0
