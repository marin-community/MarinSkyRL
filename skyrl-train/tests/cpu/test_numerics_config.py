from unittest import mock

import pytest
from omegaconf import OmegaConf

import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.utils.utils import validate_cfg


def _exact_cfg(*, clear_cache: bool):
    cfg = get_default_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.algorithm.numerics = "exact"
    cfg.generator.vllm_attention_backend = "FLASH_ATTN"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.num_inference_engines = 4
    OmegaConf.update(cfg, "generator.engine_init_kwargs.moe_backend", "triton", force_add=True)
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.weight_sync_pause.mode = "keep"
    cfg.generator.weight_sync_pause.clear_cache = clear_cache
    return cfg


@pytest.mark.parametrize(
    ("clear_cache", "numerics", "fallback_from", "logprobs_mode"),
    [(True, "exact", None, "processed_logprobs"), (False, "native", "exact", None)],
)
def test_exact_numerics_run_only_when_weight_syncs_clear_the_engines_prefix_caches(
    clear_cache, numerics, fallback_from, logprobs_mode
):
    cfg = _exact_cfg(clear_cache=clear_cache)

    with mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
        validate_cfg(cfg)

    assert cfg.trainer.algorithm.numerics == numerics
    assert cfg.trainer.algorithm.numerics_fallback_from == fallback_from
    # Exact numerics sample with the program whose log-probabilities the trainer computes.
    assert cfg.generator.engine_init_kwargs.get("logprobs_mode") == logprobs_mode
    if numerics == "exact":
        assert get_vllm_sampling_params(cfg.generator.sampling_params)["min_tokens"] == 0
