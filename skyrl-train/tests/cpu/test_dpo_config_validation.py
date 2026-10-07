import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.objective.losses import PolicyLossInputs  # noqa: F401 — populate the loss registry
from skyrl_train.utils import validate_cfg


def dpo_config(**overrides):
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "environment": {"env_class": "preference_pair"},
            "trainer": {
                "algorithm": {
                    "policy_loss_type": "dpo",
                    "advantage_estimator": "uniform",
                    "loss_reduction": "pair_mean",
                    "use_kl_loss": False,
                    "use_kl_in_reward": False,
                    "use_entropy_loss": False,
                    "off_policy_correction": "none",
                    "dynamic_sampling": {"type": None},
                },
                "placement": {"colocate_all": False},
                "use_sample_packing": False,
                "micro_train_batch_size_per_gpu": 2,
            },
            "generator": {"n_samples_per_prompt": 2},
        },
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    cfg.trainer.logger = "console"
    return cfg


def test_dpo_config_validates():
    validate_cfg(dpo_config())


def test_dpo_recipe_matches_valid_config():
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config", overrides=["+algorithm_recipe=dpo"])
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(
        cfg,
        {
            "environment": {"env_class": "preference_pair"},
            "trainer": {
                "placement": {"colocate_all": False},
                "micro_train_batch_size_per_gpu": 2,
                "use_sample_packing": False,
            },
            "generator": {"n_samples_per_prompt": 2},
        },
    )
    cfg.trainer.logger = "console"
    validate_cfg(cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trainer": {"algorithm": {"loss_reduction": "token_mean"}}}, "pair_mean"),
        ({"trainer": {"algorithm": {"dpo": {"beta": 0.0}}}}, "beta"),
        ({"trainer": {"algorithm": {"dpo": {"label_smoothing": 0.5}}}}, "label_smoothing"),
        ({"environment": {"env_class": "gsm8k"}}, "preference_pair"),
        ({"generator": {"n_samples_per_prompt": 4}}, "n_samples_per_prompt=2"),
        ({"trainer": {"algorithm": {"use_kl_loss": True}}}, "use_kl_loss"),
        ({"trainer": {"algorithm": {"advantage_estimator": "grpo"}}}, "uniform"),
        ({"trainer": {"algorithm": {"dynamic_sampling": {"type": "filter"}}}}, "dynamic sampling"),
        ({"trainer": {"use_sample_packing": True}}, "packing"),
        ({"trainer": {"placement": {"colocate_all": True}}}, "colocate_all"),
        ({"trainer": {"micro_train_batch_size_per_gpu": 3}}, "even"),
        ({"trainer": {"trajectory_selector": {"type": "best_of_n"}}}, "selector"),
        ({"trainer": {"step_wise_training": True}}, "step-wise"),
        ({"trainer": {"update_ref_every_epoch": True}}, "frozen reference"),
        ({"trainer": {"policy": {"sequence_parallel_size": 2}}}, "sequence_parallel_size=1"),
        ({"trainer": {"ref": {"sequence_parallel_size": 2}}}, "sequence_parallel_size=1"),
        ({"trainer": {"policy": {"megatron_config": {"context_parallel_size": 2}}}}, "context_parallel_size=1"),
        # 4 policy DP ranks cannot shard 2 pairs of rows into whole pairs.
        ({"trainer": {"train_batch_size": 1, "policy_mini_batch_size": 1}}, "pair-aligned shards"),
    ],
)
def test_dpo_config_rejections(overrides, message):
    with pytest.raises((ValueError, AssertionError), match=message):
        validate_cfg(dpo_config(**overrides))
