from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from skyrl_train.utils.utils import validate_cfg

config_dir = str((Path(__file__).resolve().parents[2] / "skyrl_train" / "config").resolve())


def _config(*overrides: str):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="ppo_base_config", overrides=list(overrides))


@pytest.mark.parametrize("mode", ["abort", "keep"])
def test_pause_modes_are_accepted(mode):
    validate_cfg(_config(f"trainer.fully_async.pause_mode={mode}", "trainer.logger=console"))


def test_unknown_pause_mode_is_rejected():
    with pytest.raises(ValueError, match="trainer.fully_async.pause_mode must be one of"):
        validate_cfg(_config("trainer.fully_async.pause_mode=wait", "trainer.logger=console"))
