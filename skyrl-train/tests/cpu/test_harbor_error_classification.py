from loguru import logger

from skyrl_train.utils.harbor_errors import ErrorHandlingConfig, ErrorTreatment, classify_exception_type


def test_error_handling_mapping_is_validated_at_config_boundary():
    config = ErrorHandlingConfig.from_mapping(
        {
            "enable_error_classification": True,
            "mask_exceptions": ["CampaignInfrastructureError"],
            "default_error_treatment": "passthrough",
        }
    )

    assert config.enable_error_classification is True
    assert config.mask_exceptions == frozenset({"CampaignInfrastructureError"})
    assert config.default_error_treatment is ErrorTreatment.PASSTHROUGH


def test_context_management_failure_is_masked_from_persisted_exception_type():
    treatment = classify_exception_type(
        "ContextManagementInfrastructureError",
        ErrorHandlingConfig(default_error_treatment=ErrorTreatment.ZERO),
    )

    assert treatment is ErrorTreatment.MASK


def test_shared_taxonomy_maps_agent_and_passthrough_failures():
    config = ErrorHandlingConfig()
    assert classify_exception_type("AgentTimeoutError", config) is ErrorTreatment.ZERO
    assert classify_exception_type("OutputLengthExceededError", config) is ErrorTreatment.PASSTHROUGH


def test_campaign_override_takes_precedence_over_shared_taxonomy():
    treatment = classify_exception_type(
        "ContextManagementInfrastructureError",
        ErrorHandlingConfig(zero_exceptions=frozenset({"ContextManagementInfrastructureError"})),
    )

    assert treatment is ErrorTreatment.ZERO


def test_unknown_error_is_loud_before_explicit_fallback():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        treatment = classify_exception_type(
            "FutureHarborError",
            ErrorHandlingConfig(default_error_treatment=ErrorTreatment.MASK),
        )
    finally:
        logger.remove(sink)

    assert treatment is ErrorTreatment.MASK
    assert len(records) == 1
    assert records[0]["level"].name == "ERROR"
    assert "FutureHarborError" in records[0]["message"]
