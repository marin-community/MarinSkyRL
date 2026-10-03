from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def test_tis_enables_behavior_logprobs():
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.generator.sampling_params.logprobs = None
    validate_cfg(cfg)
    assert cfg.generator.sampling_params.logprobs == 0
