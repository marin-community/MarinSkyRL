"""Warn about TIS probability mismatches while preserving configured sampling."""

import warnings

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.tis import configure_tis_sampling, warn_if_tis_sampling_mismatch
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling, pop_vllm_wrapper_kwargs


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
    configure_tis_sampling(generator)
    sampling = get_vllm_sampling_params(generator.sampling_params)
    options = OmegaConf.to_container(generator.engine_init_kwargs)
    serving = pop_vllm_wrapper_kwargs(options)

    assert options == {"logprobs_mode": "processed_logprobs", "generation_config": "vllm"}
    assert serving == {"warn_on_tis_sampling": True}
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
def test_tis_warns_about_unvalidated_sampling(settings):
    with pytest.warns(UserWarning, match="TIS"):
        warn_if_tis_sampling_mismatch(settings)


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
    with pytest.raises(ValueError, match="TIS requires processed rollout logprobs"):
        configure_tis_sampling(generator)


@pytest.mark.parametrize("logprobs", [0, 1, True])
def test_tis_openai_logprob_requests_warn_and_preserve_penalties(logprobs):
    body = {"logprobs": logprobs, "presence_penalty": 0.5}
    with pytest.warns(UserWarning, match="TIS"):
        apply_openai_sampling(body, {}, warn_on_tis_sampling=True)
    assert body["presence_penalty"] == 0.5


@pytest.mark.parametrize("logprobs", [None, False])
def test_tis_openai_evaluation_preserves_penalties(logprobs):
    body = {"logprobs": logprobs, "presence_penalty": 0.5}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_openai_sampling(body, {}, warn_on_tis_sampling=True)
    assert body["presence_penalty"] == 0.5


def test_tis_openai_checks_after_generator_overrides():
    body = {"logprobs": 0, "temperature": 0.0, "top_p": 0.5, "top_k": 10, "min_p": 0.1}
    apply_openai_sampling(body, {"temperature": 0.7}, warn_on_tis_sampling=True)
    assert body == {"logprobs": 0, "temperature": 0.7, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def test_non_tis_openai_allows_penalties_with_logprobs():
    body = {"logprobs": 0, "presence_penalty": 0.5}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_openai_sampling(body, {}, warn_on_tis_sampling=False)


def test_tis_config_warns_and_preserves_explicit_truncation_and_penalties():
    sampling = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.1, "min_tokens": 1}
    generator = OmegaConf.create(
        {"sampling_params": sampling | {"max_generate_length": 8, "logprobs": 0}, "engine_init_kwargs": {}}
    )
    with pytest.warns(UserWarning, match="TIS"):
        configure_tis_sampling(generator)
    resolved = get_vllm_sampling_params(generator.sampling_params)
    for key, value in sampling.items():
        assert resolved[key] == value
    assert generator.engine_init_kwargs.logprobs_mode == "processed_logprobs"
