"""Shared assertions for composed trainer configuration tests."""

TRAINER_MODEL_ROLES = ("policy", "ref", "critic")


def assert_role_fsdp_defaults(cfg, expected_defaults):
    """Assert that every trainer model role exposes the expected FSDP defaults."""
    for role in TRAINER_MODEL_ROLES:
        fsdp = cfg.trainer[role].fsdp_config
        for field, expected in expected_defaults.items():
            assert field in fsdp, f"trainer.{role}.fsdp_config missing {field}"
            assert fsdp[field] == expected, f"trainer.{role}.fsdp_config.{field}={fsdp[field]!r}, expected {expected!r}"
