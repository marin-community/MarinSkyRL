"""Probability convention supported by temperature-scaled TIS."""

import math
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

TIS_ENFORCEMENT_KEY = "enforce_tis_sampling"
TIS_ENGINE_OPTIONS = {"logprobs_mode": "processed_logprobs", "generation_config": "vllm"}

# Processed logprobs match the trainer only when temperature is the sole processor.
TIS_NEUTRAL_SAMPLING = {
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "min_tokens": 0,
}
TIS_UNSUPPORTED_PROCESSORS = (
    "logits_processors",
    "logit_bias",
    "allowed_token_ids",
    "bad_words",
    "structured_outputs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
)


def validate_tis_sampling(params: Mapping[str, Any]) -> None:
    """Reject sampling distributions the trainer cannot reproduce."""
    temperature = float(params.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature < 1e-5:
        raise ValueError("TIS requires a finite temperature >= 1e-5; greedy sampling is unsupported")
    for key, neutral in TIS_NEUTRAL_SAMPLING.items():
        value = params.get(key)
        if value is not None and value != neutral:
            raise ValueError(f"TIS requires {key}={neutral}; processed logprobs otherwise differ from the trainer")
    for key in TIS_UNSUPPORTED_PROCESSORS:
        if params.get(key):
            raise ValueError(f"TIS does not support the {key} sampling processor")
    if (params.get("response_format") or {}).get("type", "text") != "text":
        raise ValueError("TIS does not support constrained response_format")
    if params.get("tool_choice") not in (None, "none", "auto"):
        raise ValueError("TIS does not support constrained tool_choice")


def configure_tis_sampling(generator: DictConfig) -> None:
    """Select processed logprobs and a temperature-only sampling recipe for TIS."""
    validate_tis_sampling(generator.sampling_params)
    options = generator.engine_init_kwargs
    for key, required in TIS_ENGINE_OPTIONS.items():
        if key in options and options[key] != required:
            raise ValueError(f"TIS requires generator.engine_init_kwargs.{key}={required}")
    if options.get("override_generation_config") or options.get("logits_processors"):
        raise ValueError("TIS does not support engine-level generation overrides or logits processors")
    for key, value in TIS_NEUTRAL_SAMPLING.items():
        OmegaConf.update(generator.sampling_params, key, value, force_add=True)
    # Do not inherit checkpoint penalties or filters that the trainer does not apply.
    for key, required in TIS_ENGINE_OPTIONS.items():
        OmegaConf.update(options, key, required, force_add=True)
    OmegaConf.update(options, TIS_ENFORCEMENT_KEY, True, force_add=True)
