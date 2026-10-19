"""Source-owned Harbor agent defaults and integrity-checked prompt resources."""

import hashlib
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class SourceAgentConfig(BaseModel):
    """Named agent defaults without importing the optional Harbor runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentExecutionProfile(BaseModel):
    """An agent recipe whose prompt file travels with the source task package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    agent: SourceAgentConfig
    config_file: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def task_agent_config(directory: Path, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve source defaults, then explicit run overrides, before agent construction."""
    metadata = tomllib.loads((directory / "task.toml").read_text()).get("metadata", {})
    if "agent_profile" not in metadata:
        return dict(overrides)
    profile = AgentExecutionProfile.model_validate(metadata["agent_profile"])
    path = (directory / profile.config_file).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Source agent configuration must be inside its task package")
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile.config_sha256:
        raise ValueError(f"Source agent prompt checksum mismatch: {profile.config_file}")

    explicit = dict(overrides)
    if (explicit.get("name") is not None and explicit["name"] != profile.agent.name) or explicit.get(
        "import_path"
    ) is not None:
        logger.warning("Run agent overrides source profile {} for {}", profile.agent.name, directory)
        return dict(overrides)
    kwargs = {**profile.agent.kwargs, "config_file": str(path), **explicit.pop("kwargs", {})}
    resolved = {**profile.agent.model_dump(), **explicit, "kwargs": kwargs}
    logger.info(
        "Source agent profile: agent={} prompt={} sha256={} run_overrides={}",
        resolved["name"],
        profile.config_file,
        profile.config_sha256,
        sorted(overrides),
    )
    return resolved
