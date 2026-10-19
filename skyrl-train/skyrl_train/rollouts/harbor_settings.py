"""Harbor task runtime, verifier, and retry settings."""

from enum import StrEnum
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

HARBOR_EXCEPTION = "harbor_exception_type"

HARBOR_ENVIRONMENT_KWARGS = {
    "env_network_policy": "network_policy",
    "ttl_minutes": "ttl_minutes",
    "daytona_api_key_env": "api_key_env",
    "eval_daytona_api_key_env": "eval_api_key_env",
}


class EnvironmentType(StrEnum):
    DOCKER = "docker"
    DAYTONA = "daytona"


class EnvironmentConfig(BaseModel):
    type: EnvironmentType = EnvironmentType.DOCKER
    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_harbor_config(cls, config: Mapping[str, Any]) -> "EnvironmentConfig":
        """Resolve the shared launcher/worker environment without importing training code."""
        fields = {
            name: config[name]
            for name in cls.model_fields
            if name.startswith("override_") and config.get(name) is not None
        }
        kwargs = {
            target: config[name] for name, target in HARBOR_ENVIRONMENT_KWARGS.items() if config.get(name) is not None
        }
        return cls(type=config.get("environment_type", EnvironmentType.DAYTONA), kwargs=kwargs, **fields)


class VerifierConfig(BaseModel):
    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    disable: bool = False


DEFAULT_RETRY_EXCLUSIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)


class RetryConfig(BaseModel):
    max_retries: int = Field(default=0, ge=0)
    include_exceptions: set[str] | None = None
    exclude_exceptions: set[str] | None = Field(default_factory=lambda: set(DEFAULT_RETRY_EXCLUSIONS))
    wait_multiplier: float = 1.0
    min_wait_sec: float = 1.0
    max_wait_sec: float = 60.0
