import pytest
from omegaconf import OmegaConf

import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def _config(**settings):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    for path, value in settings.items():
        OmegaConf.update(cfg, path, value)
    return cfg


def test_validation_rejects_stop_without_a_bit_identical_mode():
    cfg = _config(**{"trainer.numerics.check.on_failure": "stop"})

    with pytest.raises(ValueError, match="on_failure=stop"):
        validate_cfg(cfg)
