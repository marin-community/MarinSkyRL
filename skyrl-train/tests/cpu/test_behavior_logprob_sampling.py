"""Validate the probability convention for behavior-logprob objectives."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from skyrl_train.config.behavior_logprobs import (
    configure_behavior_logprob_sampling,
    validate_behavior_logprob_sampling,
)
from skyrl_train.inference_engines.utils import get_vllm_sampling_params
from skyrl_train.inference_engines.vllm.utils import apply_openai_sampling, pop_vllm_wrapper_kwargs
from skyrl_train.utils.algorithm_registry import rollout_logprobs_enabled


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.2])
def test_behavior_logprob_config_reaches_engine_and_sampling_options(temperature):
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
    configure_behavior_logprob_sampling(generator)
    sampling = get_vllm_sampling_params(generator.sampling_params)
    options = OmegaConf.to_container(generator.engine_init_kwargs)
    serving = pop_vllm_wrapper_kwargs(options)

    assert options == {"logprobs_mode": "processed_logprobs", "generation_config": "vllm"}
    assert serving == {"validate_rollout_logprob_sampling": True}
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
def test_behavior_logprobs_reject_unmatched_sampling(settings):
    with pytest.raises(ValueError, match="temperature-only sampling"):
        validate_behavior_logprob_sampling(settings)


@pytest.mark.parametrize(
    "options",
    [
        {"logprobs_mode": "raw_logprobs"},
        {"generation_config": "auto"},
        {"override_generation_config": {"top_k": 20}},
        {"logits_processors": ["custom.Processor"]},
    ],
)
def test_behavior_logprobs_reject_conflicting_engine_options(options):
    generator = OmegaConf.create({"sampling_params": {"temperature": 1.2}, "engine_init_kwargs": options})
    with pytest.raises(ValueError, match="processed rollout logprobs"):
        configure_behavior_logprob_sampling(generator)


@pytest.mark.parametrize("logprobs", [0, 1, True])
def test_behavior_logprob_openai_requests_reject_penalties(logprobs):
    body = {"logprobs": logprobs, "presence_penalty": 0.5}
    with pytest.raises(ValueError, match="temperature-only sampling"):
        apply_openai_sampling(body, {}, validate_rollout_logprob_sampling=True)


@pytest.mark.parametrize("logprobs", [None, False])
def test_openai_request_without_rollout_logprobs_preserves_penalties(logprobs):
    body = {"logprobs": logprobs, "presence_penalty": 0.5}
    apply_openai_sampling(body, {}, validate_rollout_logprob_sampling=True)
    assert body["presence_penalty"] == 0.5


def test_openai_request_checks_after_generator_overrides():
    body = {"logprobs": 0, "temperature": 0.0, "top_p": 0.5, "top_k": 10, "min_p": 0.1}
    apply_openai_sampling(body, {"temperature": 0.7}, validate_rollout_logprob_sampling=True)
    assert body == {"logprobs": 0, "temperature": 0.7, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def test_disabled_rollout_logprob_validation_allows_penalties():
    body = {"logprobs": 0, "presence_penalty": 0.5}
    apply_openai_sampling(body, {}, validate_rollout_logprob_sampling=False)
    assert body["presence_penalty"] == 0.5


def test_behavior_logprob_config_rejects_explicit_truncation_and_penalties():
    sampling = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.1, "min_tokens": 1}
    generator = OmegaConf.create(
        {"sampling_params": sampling | {"max_generate_length": 8, "logprobs": 0}, "engine_init_kwargs": {}}
    )
    with pytest.raises(ValueError, match="temperature-only sampling"):
        configure_behavior_logprob_sampling(generator)


def test_checked_in_behavior_logprob_configs_use_validated_sampling():
    checked = []
    for path in sorted((REPO_ROOT / "cloud" / "iris" / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text()) or {}
        algorithm = (config.get("trainer") or {}).get("algorithm") or {}
        algorithm = OmegaConf.create(
            {"use_tis": algorithm.get("use_tis", False), "policy_loss_type": algorithm.get("policy_loss_type")}
        )
        if not rollout_logprobs_enabled(algorithm):
            continue
        validate_behavior_logprob_sampling((config.get("generator") or {}).get("sampling_params") or {})
        checked.append(path.name)

    assert checked
