"""Source-owned Harbor agent defaults and integrity-checked prompt resources."""

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from harbor_config.models.agent.name import AgentName
from harbor_config.models.trial.config import AgentConfig
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class AgentExecutionProfile(BaseModel):
    """An agent recipe whose prompt file travels with the source task package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    agent: AgentConfig
    config_file: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def task_agent_config(directory: Path, overrides: AgentConfig) -> AgentConfig:
    """Resolve source defaults, then explicit run overrides, before agent construction."""
    metadata = tomllib.loads((directory / "task.toml").read_text()).get("metadata", {})
    if "agent_profile" not in metadata:
        return overrides
    profile = AgentExecutionProfile.model_validate(metadata["agent_profile"])
    if profile.agent.name is None or profile.agent.import_path is not None:
        raise ValueError("Source agent profiles must select a registered agent, not a host import path")
    AgentName(profile.agent.name)
    path = (directory / profile.config_file).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Source agent configuration must be inside its task package")
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile.config_sha256:
        raise ValueError(f"Source agent prompt checksum mismatch: {profile.config_file}")

    explicit = overrides.model_dump(exclude_unset=True)
    if (explicit.get("name") is not None and explicit["name"] != profile.agent.name) or explicit.get(
        "import_path"
    ) is not None:
        logger.warning("Run agent overrides source profile {} for {}", profile.agent.name, directory)
        return overrides
    kwargs = {**profile.agent.kwargs, "config_file": str(path), **explicit.pop("kwargs", {})}
    resolved = AgentConfig.model_validate({**profile.agent.model_dump(), **explicit, "kwargs": kwargs})
    logger.info(
        "Source agent profile: agent={} prompt={} sha256={} run_overrides={}",
        resolved.name,
        profile.config_file,
        profile.config_sha256,
        sorted(overrides.model_fields_set),
    )
    return resolved
