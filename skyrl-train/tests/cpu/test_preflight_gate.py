import asyncio
from types import SimpleNamespace
import pytest
from loguru import logger as loguru_logger

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import PreflightGateCallback, PreflightGateError
from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.utils.preflight_gate import check_preflight_gate
from tests.rollout_fixtures import FixedPromptDataset, FixedRolloutRunner

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


def _trainer_after_first_step(rewards, training_trainer_factory):
    cfg = get_default_config()
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.train_batch_size = len(rewards) // 2
    cfg.trainer.max_steps = 1
    cfg.trainer.algorithm.advantage_estimator = "uniform"
    cfg.trainer.algorithm.off_policy_correction = "none"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.eval_before_train = False
    cfg.trainer.eval_interval = cfg.trainer.ckpt_interval = cfg.trainer.hf_save_interval = -1
    cfg.trainer.resume_mode = "none"
    groups = [
        RolloutGroup(
            {
                "prompt_token_ids": [[1], [1]],
                "response_ids": [[2, 3], [2, 3]],
                "rewards": rewards[index : index + 2],
                "loss_masks": [[1, 1], [1, 1]],
                "stop_reasons": ["stop", "stop"],
                "rollout_metrics": {},
            },
            uid=str(index // 2),
            policy_step=0,
            prompt={"uid": str(index // 2)},
        )
        for index in range(0, len(rewards), 2)
    ]
    trainer = training_trainer_factory(
        cfg,
        dataset=FixedPromptDataset([group.uid for group in groups]),
        runner=FixedRolloutRunner(groups),
        tokenizer=SimpleNamespace(decode=str, pad_token_id=0),
    )
    asyncio.run(trainer.train())
    return trainer


def test_preflight_callback_aborts_sparse_first_step_and_checks_only_once(training_trainer_factory):
    callback = PreflightGateCallback(enabled=True, num_trials=8, on_failure="abort")
    trainer = _trainer_after_first_step([0.0] * 8, training_trainer_factory)

    with pytest.raises(PreflightGateError, match="sparse"):
        callback.on_step_end(STEP_ONE, TrainerControl(), trainer=trainer)

    # The gate is a one-shot pre-flight check; later sparse steps must not abort the run.
    callback.on_step_end(STEP_ONE, TrainerControl(), trainer=trainer)


@pytest.mark.parametrize(
    ("rewards", "on_failure"),
    [
        pytest.param([1.0, 0.0] * 4, "abort", id="healthy"),
        pytest.param([0.0] * 8, "warn", id="sparse-warn-continues"),
    ],
)
def test_preflight_callback_lets_training_continue(rewards, on_failure, training_trainer_factory):
    callback = PreflightGateCallback(enabled=True, num_trials=8, on_failure=on_failure)
    control = TrainerControl()

    trainer = _trainer_after_first_step(rewards, training_trainer_factory)
    assert callback.on_step_end(STEP_ONE, control, trainer=trainer) is control


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
