# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Configuration semantics for Grug router query-bias updates."""

from collections.abc import Mapping
from enum import StrEnum


class GrugQueryBiasUpdateMode(StrEnum):
    """Policy for updating Grug router query-bias buffers during RL."""

    FROZEN = "frozen"
    REPLACE = "replace"


def resolve_grug_query_bias_update_mode(policy_config: Mapping[str, object]) -> GrugQueryBiasUpdateMode:
    """Resolve the policy mode; a missing field selects frozen behavior."""

    config_key = "grug_query_bias_update_mode"
    raw_mode = policy_config.get(config_key, GrugQueryBiasUpdateMode.FROZEN)
    try:
        return GrugQueryBiasUpdateMode(raw_mode)
    except ValueError as error:
        valid_modes = tuple(mode.value for mode in GrugQueryBiasUpdateMode)
        raise ValueError(f"invalid {config_key}: {raw_mode}. Must be one of {valid_modes}") from error
