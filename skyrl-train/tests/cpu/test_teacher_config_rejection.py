from hydra import compose, initialize_config_dir
import pytest

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg


def test_packaged_entrypoints_reject_ad_hoc_teacher_configuration():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+teacher.model_path=Qwen/Qwen3-4B"])

    with pytest.raises(ValueError, match="teacher configuration is unsupported"):
        validate_cfg(cfg)
