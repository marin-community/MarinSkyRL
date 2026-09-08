"""Native actor fixture config with no reporting-service credentials."""

import hydra

from skyrl_train.entrypoints.main_base import config_dir


def correction_actor_config(model_path: str):
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config", overrides=["trainer.logger=console"])
    # Check the real default before applying fixture topology or objective settings.
    assert cfg.trainer.algorithm.loss_reduction == "token_mean"
    cfg.trainer.policy.model.path = model_path
    return cfg
