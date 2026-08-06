# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration semantics for Grug router query-bias updates."""

from enum import StrEnum

from omegaconf import DictConfig


class GrugQueryBiasUpdateMode(StrEnum):
    """Policy for updating Grug router query-bias buffers during RL."""

    FROZEN = "frozen"
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
