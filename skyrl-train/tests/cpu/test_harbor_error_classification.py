import json

from loguru import logger

from skyrl_train.utils.harbor_errors import ErrorTreatment, classify_exception_type


def _serialized_exception_type(exception_type: str) -> str:
    exception_info = json.loads(
        json.dumps(
            {
                "exception_type": exception_type,
                "exception_message": "persisted Harbor trial failure",
                "exception_traceback": "",
                "occurred_at": "2026-07-31T18:00:00Z",
            }
        )
    )
    return exception_info["exception_type"]


def test_context_management_failure_is_masked_from_serialized_exception_info():
    treatment = classify_exception_type(
        _serialized_exception_type("ContextManagementInfrastructureError"),
        {"default_error_treatment": "zero"},
    )

    assert treatment is ErrorTreatment.MASK


def test_shared_taxonomy_maps_agent_and_passthrough_failures():
    assert classify_exception_type("AgentTimeoutError", {}) is ErrorTreatment.ZERO
    assert classify_exception_type("OutputLengthExceededError", {}) is ErrorTreatment.PASSTHROUGH


def test_campaign_override_takes_precedence_over_shared_taxonomy():
    treatment = classify_exception_type(
        "ContextManagementInfrastructureError",
        {"zero_exceptions": {"ContextManagementInfrastructureError"}},
    )

    assert treatment is ErrorTreatment.ZERO


def test_unknown_error_is_loud_before_explicit_fallback():
    messages = []
    sink = logger.add(messages.append, level="ERROR", format="{message}")
    try:
        treatment = classify_exception_type("FutureHarborError", {"default_error_treatment": "mask"})
    finally:
        logger.remove(sink)

    assert treatment is ErrorTreatment.MASK
    assert messages == [
        "Unknown Harbor exception type FutureHarborError; applying explicit default_error_treatment=mask\n"
    ]
