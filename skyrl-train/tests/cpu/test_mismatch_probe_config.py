import pytest
from omegaconf import OmegaConf

from skyrl_train.config.mismatch_probe import validate_mismatch_probe_config
from skyrl_train.config.utils import get_default_config


def _probe_cfg():
    cfg = get_default_config()
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.mismatch_probe.enabled = True
    cfg.trainer.mismatch_probe.seed = 17
    cfg.trainer.max_steps = 2
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.engine_init_kwargs.logprobs_mode = "processed_logprobs"
    cfg.generator.engine_init_kwargs.generation_config = "vllm"
    return cfg


def test_valid_probe_recipe_and_disabled_recipe():
    cfg = _probe_cfg()
    validate_mismatch_probe_config(cfg)
    cfg.trainer.mismatch_probe.enabled = False
    cfg.generator.sampling_params.temperature = 0.0
    validate_mismatch_probe_config(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("trainer.max_steps", 3, "max_steps"),
        ("trainer.mismatch_probe.prompts.count", 0, "prompts.count"),
        ("trainer.mismatch_probe.score_after_updates", [1, 2], "start at 0"),
        ("generator.sampling_params.temperature", 0.7, "temperature=1"),
        ("generator.sampling_params.top_p", 0.9, "top_p"),
        ("generator.sampling_params.logprobs", None, "generation logprobs"),
        ("generator.engine_init_kwargs.logprobs_mode", "raw_logprobs", "logprobs_mode"),
        ("trainer.mismatch_probe.extra_trainer_modes", ["router_replay"], "route capture"),
        ("trainer.mismatch_probe.layer_tokens", 1, "eager vLLM"),
    ],
)
def test_invalid_probe_launch_rejected(path, value, message):
    cfg = _probe_cfg()
    OmegaConf.update(cfg, path, value, force_add=True)
    with pytest.raises(ValueError, match=message):
        validate_mismatch_probe_config(cfg)


def test_filtered_replay_requires_explicit_fraction_and_moe_route_flags():
    cfg = _probe_cfg()
    cfg.trainer.mismatch_probe.extra_trainer_modes = ["router_replay_filtered"]
    cfg.generator.enable_return_routed_experts = True
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    with pytest.raises(ValueError, match="keep_fraction"):
        validate_mismatch_probe_config(cfg)
    cfg.trainer.mismatch_probe.filtered_replay.keep_fraction = 0.5
    validate_mismatch_probe_config(cfg)


def test_probe_requires_synchronous_trainer_and_adapter_for_layers():
    cfg = _probe_cfg()
    with pytest.raises(ValueError, match="synchronous"):
        validate_mismatch_probe_config(cfg, synchronous=False)
    cfg.generator.enforce_eager = True
    cfg.trainer.mismatch_probe.layer_tokens = 4
    with pytest.raises(ValueError, match="no registered layer adapter"):
        validate_mismatch_probe_config(cfg)
