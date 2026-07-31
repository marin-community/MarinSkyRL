from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from harbor_config.errors import ErrorCategory, error_category
from loguru import logger


class ErrorTreatment(StrEnum):
    """How a persisted Harbor trial failure contributes to RL training."""

    MASK = "mask"
    ZERO = "zero"
    PASSTHROUGH = "passthrough"


DEFAULT_ERROR_TREATMENT = ErrorTreatment.ZERO


@dataclass(frozen=True)
class ErrorHandlingConfig:
    """Typed training treatment for Harbor trial failures."""

    enable_error_classification: bool = False
    passthrough_exceptions: frozenset[str] = field(default_factory=frozenset)
    mask_exceptions: frozenset[str] = field(default_factory=frozenset)
    zero_exceptions: frozenset[str] = field(default_factory=frozenset)
    default_error_treatment: ErrorTreatment = DEFAULT_ERROR_TREATMENT
    preserve_logprobs_on_timeout: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ErrorHandlingConfig":
        """Validate the schema-derived mapping at the terminal-bench boundary."""
        return cls(
            enable_error_classification=bool(config.get("enable_error_classification", False)),
            passthrough_exceptions=frozenset(config.get("passthrough_exceptions", ())),
            mask_exceptions=frozenset(config.get("mask_exceptions", ())),
            zero_exceptions=frozenset(config.get("zero_exceptions", ())),
            default_error_treatment=ErrorTreatment(config.get("default_error_treatment", DEFAULT_ERROR_TREATMENT)),
            preserve_logprobs_on_timeout=bool(config.get("preserve_logprobs_on_timeout", True)),
        )


_CATEGORY_TREATMENTS = {
    ErrorCategory.INFRASTRUCTURE: ErrorTreatment.MASK,
    ErrorCategory.AGENT: ErrorTreatment.ZERO,
    ErrorCategory.PASSTHROUGH: ErrorTreatment.PASSTHROUGH,
}


def classify_exception_type(exception_type: str, config: ErrorHandlingConfig) -> ErrorTreatment:
    """Classify a persisted Harbor exception name, honoring campaign overrides."""
    override_fields = (
        (config.passthrough_exceptions, ErrorTreatment.PASSTHROUGH),
        (config.mask_exceptions, ErrorTreatment.MASK),
        (config.zero_exceptions, ErrorTreatment.ZERO),
    )
    for exception_types, treatment in override_fields:
        if exception_type in exception_types:
            return treatment

    category = error_category(exception_type)
    if category is not ErrorCategory.UNKNOWN:
        return _CATEGORY_TREATMENTS[category]

    logger.error(
        "Unknown Harbor exception type {}; applying explicit default_error_treatment={}",
        exception_type,
        config.default_error_treatment.value,
    )
    return config.default_error_treatment
