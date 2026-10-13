"""Explicit runtime settings for task execution tests."""

from collections.abc import Mapping

from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import NetworkPolicy
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata
from skyrl_train.dataset.prepared_tasks import TaskSpecDataset
from skyrl_train.rollouts.task_lowering import TaskExecutionSettings

SHELL_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.SHELL_SIMULATOR, capabilities=("shell", "filesystem")
)
NATIVE_ENVIRONMENT = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS,
    docker_image="fixture@sha256:" + "0" * 64,
)


def text_task(task_id: str = "task", *, prompt="Question", source="fixture") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Projection fixture"),
        source=Source(dataset=source, revision="1", row="0", importer_revision="1"),
    )


def task_input(
    lowered: LoweredTaskSpec,
    *,
    metric_class=None,
    teacher_route=None,
    data_source=None,
    group_grader=None,
    metric_tags=(),
) -> TaskInput:
    return TaskInput(
        lowered=lowered,
        metadata=TrajectoryMetadata(
            uid=lowered.task.id,
            metric_class=metric_class
            or ("taskcompendium" if lowered.session.task_session == "shellbox" else lowered.session.task_session),
            data_source=lowered.task.source.dataset if data_source is None else data_source,
            teacher_route=teacher_route,
            metric_tags=metric_tags,
        ),
        group_grader=group_grader,
    )


def machine_runtime(backend: str) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend=backend,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=5,
        cleanup_timeout=5,
    )


def session_spec(
    session_name: str = "shellbox",
    *,
    max_turns: int = 2,
    **limits,
) -> TaskSessionSpec:
    settings = dict(
        task_session=session_name,
        max_turns=max_turns,
        model_turn_timeout=5,
        command_timeout=1,
        tool_turn_timeout=5,
        total_turn_timeout=None,
        attempt_timeout=None,
        verifier_timeout=5,
        cleanup_timeout=5,
    )
    settings.update(limits)
    return TaskSessionSpec(**settings)


def lowered_task(
    task: TaskSpec,
    session_name: str = "shellbox",
    *,
    max_turns: int = 2,
    backend: str | None = None,
    verifier_backend: str | None = None,
    **limits,
) -> LoweredTaskSpec:
    return LoweredTaskSpec(
        task=task,
        runtime=TaskRuntimeSpec(
            task_machine=None if backend is None else machine_runtime(backend),
            verifier_machine=None if verifier_backend is None else machine_runtime(verifier_backend),
        ),
        session=session_spec(session_name, max_turns=max_turns, **limits),
    )


def task_dataset(
    paths: list[str],
    tokenizer,
    max_prompt_length: int = 100,
    *,
    session: TaskSessionSpec | None = None,
    source_executions: Mapping[str, TaskExecutionSettings] | None = None,
    num_workers: int = 1,
) -> TaskSpecDataset:
    return TaskSpecDataset(
        paths,
        tokenizer,
        max_prompt_length,
        execution=TaskExecutionSettings(
            runtime=TaskRuntimeSpec(
                task_machine=machine_runtime("docker"),
                verifier_machine=machine_runtime("docker"),
            ),
            session=session_spec() if session is None else session,
        ),
        source_executions={} if source_executions is None else source_executions,
        num_workers=num_workers,
    )
