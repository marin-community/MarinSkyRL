"""Behavioral tests for evaluation callback scheduling."""

import pytest
import yaml
from pathlib import Path

from cloud.iris.rl_config_translation import parse_rl_config
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import EvaluationCallback, create_callback_from_config


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
async def test_iris_evaluation_callback_stops_when_group_reward_improves(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    raw = yaml.safe_load((repository / "cloud/iris/configs/qwen_megatron_smoke.yaml").read_text())
    raw["trainer"]["callbacks"] = [
        {
            "type": "evaluation",
            "metric_groups": {"eval/train/avg_score": ["eval/cat_count_n1/avg_score", "eval/cat_count_n2/avg_score"]},
            "stop_on_improvement": {"eval/train/avg_score": 0.4},
        }
    ]
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(raw))
    callback = create_callback_from_config(parse_rl_config(str(path)).trainer["callbacks"][0])
    control = TrainerControl()
    for step, scores, expected_reward, expected_stop in [(0, (0.1, 0.3), 0.2, False), (5, (0.6, 0.8), 0.7, True)]:
        metrics = {"eval/cat_count_n1/avg_score": scores[0], "eval/cat_count_n2/avg_score": scores[1]}
        control = await callback.on_evaluate_async(
            TrainerState(global_step=step, epoch=0, total_steps=10, num_steps_per_epoch=10),
            control,
            metrics=metrics,
            trainer=None,
        )
        assert metrics["eval/train/avg_score"] == pytest.approx(expected_reward)
        assert control.should_training_stop is expected_stop
    assert metrics["eval/train/avg_score_improvement"] == pytest.approx(0.5)
