"""Explicit runtime settings for task execution tests."""

from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import NetworkPolicy
from taskcompendium.models import TaskSpec


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
