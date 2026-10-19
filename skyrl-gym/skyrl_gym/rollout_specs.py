"""Build published Marin rollout specs from SkyRL's task configuration."""

import json
from collections.abc import Mapping
from typing import Any

from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.shell import ShellToolConfig

SHELLBOX_HARNESS = "shellbox"
_SHELLBOX_PREFIX = SHELLBOX_HARNESS + ":"


def rollout_spec(
    task: TaskSpec,
    session: Mapping[str, Any],
    *,
    task_machine: MachineRuntimeSpec | None = None,
    verifier_machine: MachineRuntimeSpec | None = None,
) -> TaskRolloutSpec:
    """Separate harness selection from rollout limits in the published task spec."""
    settings = dict(session)
    harness = settings.pop("task_session")
    shell_tool = settings.pop("shell_tool", {})
    command_timeout = settings.pop("command_timeout", None)
    trace = TraceMode(settings.pop("trace", TraceMode.EXACT_TOKENS))
    if harness == SHELLBOX_HARNESS:
        shell_settings = {"command_timeout": command_timeout, "shell_tool": ShellToolConfig.model_validate(shell_tool).model_dump()}
        harness += ":" + json.dumps(shell_settings, sort_keys=True, separators=(",", ":"))
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=RolloutLimits.model_validate(settings),
        trace=trace,
    )


def shell_harness(name: str) -> ShellHarness:
    """Restore the shell configuration encoded in a task's harness name."""
    if not name.startswith(_SHELLBOX_PREFIX):
        raise ValueError(f"Not a Shellbox harness: {name!r}")
    settings = json.loads(name[len(_SHELLBOX_PREFIX) :])
    return ShellHarness(
        command_timeout=settings["command_timeout"],
        shell_tool=ShellToolConfig.model_validate(settings["shell_tool"]),
    )


def is_shell_harness(name: str) -> bool:
    return name.startswith(_SHELLBOX_PREFIX)
