from hydra import compose, initialize_config_dir
import pytest
from omegaconf import OmegaConf

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils import validate_cfg


def test_packaged_entrypoints_reject_ad_hoc_teacher_configuration():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+teacher.model_path=Qwen/Qwen3-4B"])

    with pytest.raises(ValueError, match="legacy teacher configuration is not supported"):
        validate_cfg(cfg)


def test_packaged_entrypoints_accept_distillation_only_replace_mode():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    }
                }
            },
            "teachers": {
                "primary": {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                    "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
                    "tokenizer_fingerprint": f"sha256:{'a' * 64}",
                    "max_sequence_length": 32768,
                    "request_timeout_seconds": 120,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "opd": {
                    "revision": "route-revision",
                    "routes": {"default": {"teacher": "primary", "weight": 1.0}},
                }
            },
        },
    )
    cfg.trainer.logger = "console"

    validate_cfg(cfg)
