# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration semantics for Grug router query-bias updates."""

import math
from enum import StrEnum

from omegaconf import DictConfig


class GrugQueryBiasUpdateMode(StrEnum):
    """Policy for updating Grug router query-bias buffers during RL."""

    FROZEN = "frozen"
    INTERPOLATE = "interpolate"
    REPLACE = "replace"


def resolve_grug_query_bias_update_mode(policy_config: DictConfig) -> GrugQueryBiasUpdateMode:
    """Resolve the required query-bias update mode."""

    config_key = "grug_query_bias_update_mode"
    if config_key not in policy_config:
        raise ValueError(f"missing required policy configuration: {config_key}")
    raw_mode = policy_config[config_key]
    try:
        return GrugQueryBiasUpdateMode(raw_mode)
    except ValueError as error:
        valid_modes = tuple(mode.value for mode in GrugQueryBiasUpdateMode)
        raise ValueError(f"invalid {config_key}: {raw_mode}. Must be one of {valid_modes}") from error


def resolve_grug_query_bias_target_weight(
    policy_config: DictConfig,
    mode: GrugQueryBiasUpdateMode,
) -> float:
    """Return the fraction of the quantile target applied per optimizer step."""

    config_key = "grug_query_bias_interpolation_weight"
    raw_weight = policy_config.get(config_key)
    if mode is GrugQueryBiasUpdateMode.FROZEN:
        if raw_weight is not None:
            raise ValueError(f"{config_key} is only valid when grug_query_bias_update_mode=interpolate")
        return 0.0
    if mode is GrugQueryBiasUpdateMode.REPLACE:
        if raw_weight is not None:
            raise ValueError(f"{config_key} is only valid when grug_query_bias_update_mode=interpolate")
        return 1.0
    if raw_weight is None:
        raise ValueError(f"{config_key} is required when grug_query_bias_update_mode=interpolate")
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{config_key} must be a number; got {raw_weight}") from error
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError(f"{config_key} must be finite and between 0 and 1 exclusively; got {raw_weight}")
    return weight
