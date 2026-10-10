# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert single-stage Harbor packages with prebuilt images to TaskSpec."""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    MissingArtifactPolicy,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource


class TaskOS(StrEnum):
    LINUX = "linux"
    WINDOWS = "windows"


class VerifierEnvironmentMode(StrEnum):
    SHARED = "shared"
    SEPARATE = "separate"


class EnvironmentConfig(BaseModel):
    """Harbor task fields used to create SkyRL's task and machine specs."""

    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    os: TaskOS = TaskOS.LINUX
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    gpu_types: list[str] | None = None
    tpu: dict[str, Any] | None = None
    allow_internet: bool = True
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None
    healthcheck: dict[str, Any] | None = None
    workdir: str | None = None


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None
    environment_mode: VerifierEnvironmentMode | None = None
    environment: EnvironmentConfig | None = None
    collect: list[dict[str, Any]] = Field(default_factory=list)


class AgentConfig(BaseModel):
    timeout_sec: float | None = None
    user: str | int | None = None


class ArtifactConfig(BaseModel):
    source: str
    destination: str | None = None
    exclude: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """Parse the subset of task.toml that SkyRL executes."""

    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    multi_step_reward_strategy: str | None = None
    steps: list[dict[str, Any]] | None = None
    artifacts: list[str | ArtifactConfig] = Field(default_factory=list)

    @classmethod
    def model_validate_toml(cls, contents: str) -> "TaskConfig":
        return cls.model_validate(tomllib.loads(contents))


ARTIFACTS_PATH = "/logs/artifacts"
REWARD_PATH = "/logs/verifier"
AGENT_LOG_PATH = "/logs/agent"
GRADER_DIRECTORY = "/tests"
GRADER_PATH = f"{GRADER_DIRECTORY}/test.sh"


def _directory_resources(directory: Path, prefix: str = "") -> tuple[TaskResource, ...]:
    resources = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Task files cannot contain symlinks: {path}")
        if path.is_file():
            resources.append(
                inline_resource(f"{prefix}{path.relative_to(directory).as_posix()}", path.read_bytes()).model_copy(
                    update={"mode": f"{path.stat().st_mode & 0o777:03o}"}
                )
            )
    return tuple(resources)


def _environment(config: EnvironmentConfig, *, capabilities: tuple[str, ...] = ()) -> EnvironmentRequirements:
    unsupported = {
        key: value
        for key, value in config.model_dump().items()
        if key in {"gpu_types", "tpu", "mcp_servers", "skills_dir", "healthcheck"} and value
    }
    if config.os != TaskOS.LINUX or unsupported:
        raise NotImplementedError(f"Unsupported Harbor environment: os={config.os}, fields={sorted(unsupported)}")
    if not config.docker_image:
        raise NotImplementedError(
            "Harbor tasks require a prebuilt, digest-pinned image; Dockerfile tasks are unsupported"
        )
    return EnvironmentRequirements(
        capabilities=capabilities,
        docker_image=config.docker_image,
        working_directory=config.workdir,
        environment_variables=config.env,
    )


def harbor_task(directory: Path, *, source: Source) -> TaskSpec:
    """Preserve task semantics and grade with the package's tests in a separate machine.

    Shared verifier mode, multi-stage packages, image builds, and healthchecks are unsupported.
    Machine limits, users, and deadlines belong to runtime lowering.
    """
    config = TaskConfig.model_validate_toml((directory / "task.toml").read_text())
    if config.steps or config.multi_step_reward_strategy:
        raise NotImplementedError("Multi-stage Harbor tasks are unsupported")
    if config.verifier.environment_mode != VerifierEnvironmentMode.SEPARATE:
        raise NotImplementedError("Harbor shell grading requires a separate verifier environment")
    requirements = _environment(config.environment, capabilities=("shell", "filesystem"))
    requirements = requirements.model_copy(
        update={
            "setup_commands": (
                f"mkdir -p {ARTIFACTS_PATH} {REWARD_PATH} {AGENT_LOG_PATH}",
                f"chmod 777 {ARTIFACTS_PATH} {REWARD_PATH} {AGENT_LOG_PATH}",
            )
        }
    )
    worker = _directory_resources(directory / "setup_files", "setup_files/")
    private = _directory_resources(directory / "tests")
    verifier_requirements = _environment(config.verifier.environment or config.environment).model_copy(
        update={
            "environment_variables": {
                **(config.verifier.environment or config.environment).env,
                **config.verifier.env,
            },
        }
    )
    if config.verifier.collect:
        raise NotImplementedError("Harbor collect hooks with per-hook users and deadlines are unsupported")
    artifacts = [ArtifactConfig(source=item) if isinstance(item, str) else item for item in config.artifacts]
    if not any(artifact.source.rstrip("/") == ARTIFACTS_PATH for artifact in artifacts):
        artifacts.append(ArtifactConfig(source=ARTIFACTS_PATH))
    grader = ScriptGrader(
        argv=("bash", GRADER_PATH),
        # Harbor runs its tests in the image's working directory; an unset one becomes the root.
        cwd=verifier_requirements.working_directory or "/",
        environment=verifier_requirements,
        answer_path=None,
        artifacts=tuple(
            VerifierArtifact(
                source=artifact.source,
                target=artifact.source,
                kind=ArtifactKind.AUTO,
                exclude=tuple(artifact.exclude),
                missing=MissingArtifactPolicy.SKIP,
            )
            for artifact in artifacts
        ),
        reward=FileReward(
            pass_above=0,
            files=(
                RewardFile(path=f"{REWARD_PATH}/reward.json", format=RewardFileFormat.JSON),
                RewardFile(path=f"{REWARD_PATH}/reward.txt", format=RewardFileFormat.NUMBER),
            ),
        ),
    )
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=ConversationInput(
            events=(TextMessage(role="user", content=(directory / "instruction.md").read_text()),)
        ),
        environment_requirements=requirements,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        resources=ResourceGroups(worker=worker, verifier=private),
        source=source,
        tags=("harbor",),
    )
