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


def register_correction_reference(mode: str, addfinalizer) -> str:
    """Refresh a cached actor after the parametrized fixture starts a new Ray cluster."""
    from functools import partial

    from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
    from tests.offpolicy_mask_reference import regular_correction_reference_policy_loss

    PolicyLossRegistry.sync_with_actor()
    name = "test_correction_" + mode
    PolicyLossRegistry.register(name, partial(regular_correction_reference_policy_loss, mode=mode))
    addfinalizer(lambda: PolicyLossRegistry.unregister(name))
    return name
