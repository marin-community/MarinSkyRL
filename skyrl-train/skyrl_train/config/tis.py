"""Probability convention supported by temperature-scaled TIS."""

import math
import warnings
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

MIN_NON_GREEDY_TEMPERATURE = 1e-5

TIS_WARNING_KEY = "warn_on_tis_sampling"
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


def warn_if_tis_sampling_mismatch(params: Mapping[str, Any]) -> None:
    """Warn about sampling outside the validated temperature-only TIS recipe."""
    mismatches = []
    temperature = float(params.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature < MIN_NON_GREEDY_TEMPERATURE:
        mismatches.append(f"temperature must be finite and >= {MIN_NON_GREEDY_TEMPERATURE:g} for non-greedy sampling")
    for key, neutral in TIS_NEUTRAL_SAMPLING.items():
        value = params.get(key)
        if value is not None and value != neutral:
            mismatches.append(f"{key}={value!r} (validated value: {neutral!r})")
    for key in TIS_UNSUPPORTED_PROCESSORS:
        if params.get(key):
            mismatches.append(f"{key} modifies the sampling distribution")
    if (params.get("response_format") or {}).get("type", "text") != "text":
        mismatches.append("constrained response_format")
    if params.get("tool_choice") not in (None, "none", "auto"):
        mismatches.append("constrained tool_choice")
    if mismatches:
        warnings.warn(
            "TIS probability matching is only validated for full-distribution, temperature-only sampling: "
            + "; ".join(mismatches)
            + ". Continuing with the configured settings; importance weighting may not correct the sampling bias.",
            UserWarning,
            stacklevel=2,
        )


def configure_tis_sampling(generator: DictConfig) -> None:
    """Default to processed logprobs and warn about incompatible explicit settings."""
    warn_if_tis_sampling_mismatch(generator.sampling_params)
    options = generator.engine_init_kwargs
    mismatches = []
    for key, recommended in TIS_ENGINE_OPTIONS.items():
        if key in options and options[key] != recommended:
            mismatches.append(f"{key}={options[key]!r} (validated value: {recommended!r})")
    if options.get("override_generation_config") or options.get("logits_processors"):
        mismatches.append("engine-level generation overrides or logits processors")
    if mismatches:
        warnings.warn(
            "TIS probability matching is not validated with "
            + "; ".join(mismatches)
            + ". Continuing with the configured engine options.",
            UserWarning,
            stacklevel=2,
        )
    for key, value in TIS_NEUTRAL_SAMPLING.items():
        if generator.sampling_params.get(key) is None:
            OmegaConf.update(generator.sampling_params, key, value, force_add=True)
    # Avoid inheriting checkpoint filters by default, while preserving explicit user options.
    for key, recommended in TIS_ENGINE_OPTIONS.items():
        if key not in options:
            OmegaConf.update(options, key, recommended, force_add=True)
    OmegaConf.update(options, TIS_WARNING_KEY, True, force_add=True)
