import pytest
from transformers import PretrainedConfig

from skyrl_train.inference_engines.ray_wrapped_inference_engine import validate_grug_vllm_support
from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE


def test_grug_vllm_support_guard_is_conditional() -> None:
    grug_config = PretrainedConfig()
    grug_config.model_type = "grug_moe"

    with pytest.raises(RuntimeError, match="GrugMoeForCausalLM"):
        validate_grug_vllm_support(grug_config, set())

    validate_grug_vllm_support(grug_config, {GRUG_MOE_ARCHITECTURE})
    validate_grug_vllm_support(PretrainedConfig(), set())
