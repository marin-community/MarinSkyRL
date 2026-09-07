"""Reject TIS recipes whose serving probabilities differ from the trainer."""

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.tis import configure_tis_logprobs, validate_tis_sampling
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.2])
def test_tis_config_reaches_engine_and_sampling_options(temperature):
    generator = OmegaConf.create(
        {
            "sampling_params": {
                "temperature": temperature,
                "max_generate_length": 8,
                "logprobs": 0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
            },
            "engine_init_kwargs": {},
        }
    )
    configure_tis_logprobs(generator)
    sampling = get_vllm_sampling_params(generator.sampling_params)
    options = OmegaConf.to_container(generator.engine_init_kwargs)
    serving = pop_openai_kwargs(options)

    assert options == {"logprobs_mode": "processed_logprobs", "generation_config": "vllm"}
    assert serving == {"enforce_tis_sampling": True}
    assert sampling["temperature"] == temperature
    assert sampling["min_tokens"] == 0
    assert sampling["top_p"] == 1.0 and sampling["top_k"] == -1
    assert sampling["repetition_penalty"] == 1.0


@pytest.mark.parametrize(
    "settings",
    [
        {"temperature": 0.0},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_p": 0.95},
        {"top_k": 20},
        {"min_p": 0.1},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"min_tokens": 1},
        {"allowed_token_ids": [1]},
        {"logit_bias": {"1": 1.0}},
        {"response_format": {"type": "json_object"}},
        {"tool_choice": "required"},
    ],
)
def test_tis_rejects_distributions_the_trainer_does_not_reproduce(settings):
    with pytest.raises(ValueError, match="TIS"):
        validate_tis_sampling(settings)


@pytest.mark.parametrize(
    "options",
    [
        {"logprobs_mode": "raw_logprobs"},
        {"generation_config": "auto"},
        {"override_generation_config": {"top_k": 20}},
        {"logits_processors": ["custom.Processor"]},
    ],
)
def test_tis_rejects_conflicting_engine_options(options):
    generator = OmegaConf.create({"sampling_params": {"temperature": 1.2}, "engine_init_kwargs": options})
    with pytest.raises(ValueError, match="TIS"):
        configure_tis_logprobs(generator)
