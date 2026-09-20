from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import DistillationTokenBudgetCallback
from skyrl_train.callbacks.builtin import create_default_callbacks
from skyrl_train.config.utils import get_default_config


def _state(scored_tokens: float) -> TrainerState:
    return TrainerState(
        global_step=1,
        epoch=0,
        total_steps=10,
        num_steps_per_epoch=10,
        metrics={"distillation/scored_tokens": scored_tokens},
    )


def test_distillation_token_budget_accumulates_batches_and_stops_at_the_boundary():
    callback = DistillationTokenBudgetCallback(100)
    trainer = SimpleNamespace(distillation_scored_tokens_total=40, all_metrics={})

    control = callback.on_step_end(_state(32.0), TrainerControl(), trainer=trainer)
    assert control is not None
    assert not control.should_training_stop
    assert trainer.all_metrics["distillation/scored_tokens_total"] == 72.0

    control = callback.on_step_end(_state(32.0), TrainerControl(), trainer=trainer)
    assert control is not None
    assert control.should_training_stop
    assert trainer.distillation_scored_tokens_total == 104
    assert trainer.all_metrics["distillation/token_budget"] == 100.0


def test_distillation_token_budget_requires_opd_configuration():
    cfg = get_default_config()
    cfg.trainer.distillation_token_budget = 100

    with pytest.raises(ValueError, match="requires trainer.algorithm.distillation"):
        create_default_callbacks(cfg)


def test_distillation_token_budget_is_installed_for_explicit_callback_configs():
    cfg = get_default_config()
    cfg.trainer.distillation_token_budget = 100
    OmegaConf.update(
        cfg,
        "trainer.algorithm.distillation",
        {"objective": "sampled_reverse_kl"},
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "trainer.callbacks",
        [{"type": "checkpoint", "save_steps": 5}],
        force_add=True,
    )

    callbacks = create_default_callbacks(cfg)

    assert sum(isinstance(callback, DistillationTokenBudgetCallback) for callback in callbacks) == 1


def test_explicit_callbacks_reject_multiple_distillation_token_budgets():
    cfg = get_default_config()
    OmegaConf.update(
        cfg,
        "trainer.algorithm.distillation",
        {"objective": "sampled_reverse_kl"},
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "trainer.callbacks",
        [
            {"type": "distillation_token_budget", "token_budget": 100},
            {"type": "distillation_token_budget", "token_budget": 200},
        ],
        force_add=True,
    )

    with pytest.raises(ValueError, match="multiple distillation token budgets"):
        create_default_callbacks(cfg)
