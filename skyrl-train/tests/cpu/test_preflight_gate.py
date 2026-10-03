"""Pre-flight reward gate: the pure band check and the callback fed by a real trainer step."""

from pathlib import Path

import pytest
from loguru import logger as loguru_logger
from omegaconf import OmegaConf

from skyrl_train.callbacks import builtin as callbacks
from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import PreflightGateCallback, PreflightGateError
from skyrl_train.config.utils import get_default_config
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.preflight_gate import check_preflight_gate

STEP_ONE = TrainerState(global_step=1, epoch=0, total_steps=80, num_steps_per_epoch=80)


@pytest.mark.parametrize(
    ("rewards", "band", "passed", "reason"),
    [
        pytest.param([0.0] * 256, {}, False, "sparse", id="all-zero"),
        pytest.param([0.1] * 26 + [0.0] * 230, {}, False, "sparse", id="near-zero"),
        pytest.param([1.0] * 256, {}, False, "saturated", id="all-one"),
        pytest.param([1.0] * 220 + [0.0] * 36, {}, False, "saturated", id="mean-0.86"),
        pytest.param([1.0] * 128 + [0.0] * 128, {}, True, None, id="mean-0.5"),
        pytest.param([0.25] * 256, {}, True, None, id="min-bound-inclusive"),
        pytest.param([1.0] * 220 + [0.0] * 36, {"min_reward": 0.1, "max_reward": 0.9}, True, None, id="custom-band"),
    ],
)
def test_preflight_gate_band(rewards, band, passed, reason):
    result = check_preflight_gate(rewards, **band)

    assert result.passed is passed
    assert result.reason == reason
    assert result.mean_reward == pytest.approx(sum(rewards) / len(rewards))


def _trainer_after_first_step(rewards: list[float]) -> RayPPOTrainer:
    """Run the trainer's reward post-processing, which records what the gate reads.

    Regression: the gate once read an attribute no trainer set, so every enabled gate
    reported "no rewards" and passed. Feeding the callback a trainer that went through
    postprocess_trajectory_batch keeps the two sides paired.
    """
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.generator.n_samples_per_prompt = 2
    trainer.all_metrics = {}
    batch = {
        "prompt_token_ids": [[1]] * len(rewards),
        "response_ids": [[2, 3]] * len(rewards),
        "rewards": list(rewards),
        "loss_masks": [[1, 1]] * len(rewards),
        "stop_reasons": ["stop"] * len(rewards),
        "rollout_metrics": None,
    }
    trainer.postprocess_trajectory_batch(batch, [str(index // 2) for index in range(len(rewards))])
    return trainer


def test_preflight_callback_aborts_sparse_first_step_and_checks_only_once(generated_recipe_schema):
    root = Path(__file__).resolve().parents[3]
    assert Path(callbacks.__file__).resolve() == root / "skyrl-train/skyrl_train/callbacks/builtin.py"
    recipe_type, base = generated_recipe_schema
    recipe = recipe_type.from_document(
        {
            "trainer": {
                "enable_db_registration": False,
                "preflight_gate": {"enabled": True, "num_trials": 8, "on_failure": "warn"},
            },
            "generator": {"inference_stats_interval": 0},
        }
    ).with_settings(["trainer.preflight_gate.on_failure=abort"])
    with pytest.raises(ValueError):
        recipe.with_settings(["trainer.preflight_gate.on_failure=unknown-failure-policy"])
    callback = next(
        callback
        for callback in callbacks.create_default_callbacks(OmegaConf.merge(base, recipe.to_skyrl()))
        if isinstance(callback, PreflightGateCallback)
    )

    with pytest.raises(PreflightGateError, match="sparse"):
        callback.on_step_end(STEP_ONE, TrainerControl(), trainer=_trainer_after_first_step([0.0] * 8))

    # The gate is a one-shot pre-flight check; later sparse steps must not abort the run.
    callback.on_step_end(STEP_ONE, TrainerControl(), trainer=_trainer_after_first_step([0.0] * 8))


@pytest.mark.parametrize(
    ("rewards", "on_failure"),
    [
        pytest.param([1.0, 0.0] * 4, "abort", id="healthy"),
        pytest.param([0.0] * 8, "warn", id="sparse-warn-continues"),
    ],
)
def test_preflight_callback_lets_training_continue(rewards, on_failure, generated_recipe_schema):
    root = Path(__file__).resolve().parents[3]
    assert Path(callbacks.__file__).resolve() == root / "skyrl-train/skyrl_train/callbacks/builtin.py"
    recipe_type, base = generated_recipe_schema
    recipe = recipe_type.from_document(
        {
            "trainer": {
                "enable_db_registration": False,
                "preflight_gate": {"enabled": True, "num_trials": 8, "on_failure": "abort"},
            },
            "generator": {"inference_stats_interval": 0},
        }
    ).with_settings([f"trainer.preflight_gate.on_failure={on_failure}"])
    with pytest.raises(ValueError):
        recipe.with_settings(["trainer.preflight_gate.on_failure=unknown-failure-policy"])
    callback = next(
        callback
        for callback in callbacks.create_default_callbacks(OmegaConf.merge(base, recipe.to_skyrl()))
        if isinstance(callback, PreflightGateCallback)
    )
    control = TrainerControl()

    assert callback.on_step_end(STEP_ONE, control, trainer=_trainer_after_first_step(rewards)) is control


def test_preflight_callback_reports_loudly_when_trainer_exposes_no_rewards():
    """Silently skipping leaves the run ungated while looking gated, which is the failure this gate exists to prevent."""

    class TrainerWithoutRewards:
        """A plain object, not a MagicMock: auto-created attributes would hide the missing-attribute case."""

    callback = PreflightGateCallback(enabled=True, on_failure="abort")

    # The gate logs through loguru, which does not reach pytest's caplog.
    messages: list = []
    sink_id = loguru_logger.add(messages.append, level="ERROR")
    try:
        callback.on_step_end(STEP_ONE, TrainerControl(), trainer=TrainerWithoutRewards())
    finally:
        loguru_logger.remove(sink_id)

    assert any("UNGATED" in message for message in messages)
