"""Add run-specific machines and session settings to prepared TaskSpecs."""

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    ScriptGrader,
    SessionGrader,
    TaskSpec,
    VerifyitGrader,
)
from taskcompendium.runtime.shell import resolve_shell_tool


class TaskExecutionSettings(BaseModel):
    """Machine and session settings shared by one source in a training run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: TaskRuntimeSpec
    session: TaskSessionSpec


def source_execution_settings(
    config: Mapping[str, Any],
) -> tuple[TaskExecutionSettings, dict[str, TaskExecutionSettings]]:
    """Return shared and source-specific settings from the run config."""
    unknown = set(config) - {"runtime", "session", "sources", "max_verifier_workers"}
    if unknown:
        raise ValueError(f"Unknown task execution settings: {sorted(unknown)}")
    execution = {"runtime": config["runtime"], "session": config["session"]}
    sources = {
        name: TaskExecutionSettings.model_validate(
            OmegaConf.to_container(OmegaConf.merge(execution, overrides), resolve=True)
        )
        for name, overrides in config.get("sources", {}).items()
    }
    return TaskExecutionSettings.model_validate(execution), sources


def lowered_task(task: TaskSpec, settings: TaskExecutionSettings) -> LoweredTaskSpec:
    """Select machines for the task requirements without changing the task definition."""
    if isinstance(task.grader, SessionGrader) and settings.session.task_session == SHELLBOX_SESSION:
        raise ValueError(f"Task {task.id!r} requires a registered task session instead of the Shellbox session")
    session = (
        settings.session
        if isinstance(task.grader, SessionGrader)
        else settings.session.model_copy(update={"task_session": SHELLBOX_SESSION})
    )
    task_machine = (
        settings.runtime.task_machine
        if task.environment_requirements != EnvironmentRequirements()
        or task.resources.all
        or task.resources.worker
        or task.answer_type in {AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE}
        or (session.task_session == SHELLBOX_SESSION and resolve_shell_tool(task, session.shell_tool) is not None)
        else None
    )
    verifier_machine = (
        settings.runtime.verifier_machine
        if isinstance(task.grader, ScriptGrader | VerifyitGrader) and task.grader.environment is not None
        else None
    )
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(task_machine=task_machine, verifier_machine=verifier_machine),
        session=session,
    )
