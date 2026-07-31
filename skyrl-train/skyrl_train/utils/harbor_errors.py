from enum import StrEnum
from typing import Any, Mapping

from harbor_config.errors import ErrorCategory, error_category
from loguru import logger


class ErrorTreatment(StrEnum):
    """How a persisted Harbor trial failure contributes to RL training."""

    MASK = "mask"
    ZERO = "zero"
    PASSTHROUGH = "passthrough"


_CATEGORY_TREATMENTS = {
    ErrorCategory.INFRASTRUCTURE: ErrorTreatment.MASK,
    ErrorCategory.AGENT: ErrorTreatment.ZERO,
    ErrorCategory.PASSTHROUGH: ErrorTreatment.PASSTHROUGH,
}


def classify_exception_type(exception_type: str, config: Mapping[str, Any]) -> ErrorTreatment:
    """Classify a persisted Harbor exception name, honoring campaign overrides."""
    override_fields = (
        ("passthrough_exceptions", ErrorTreatment.PASSTHROUGH),
        ("mask_exceptions", ErrorTreatment.MASK),
        ("zero_exceptions", ErrorTreatment.ZERO),
    )
    for field, treatment in override_fields:
        if exception_type in config.get(field, ()):
            return treatment

    category = error_category(exception_type)
    if category is not ErrorCategory.UNKNOWN:
        return _CATEGORY_TREATMENTS[category]

    fallback = ErrorTreatment(config.get("default_error_treatment", ErrorTreatment.ZERO))
    logger.error(
        "Unknown Harbor exception type {}; applying explicit default_error_treatment={}",
        exception_type,
        fallback.value,
    )
    return fallback
