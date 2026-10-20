"""A Shellbox harness for workspace tasks that tells the model where its files are and keeps it on the tool loop.

``environment.task_sessions.guided_shell`` enables it in place of RolloutEngine's ``ShellHarness`` for every
``shellbox:`` task. It keeps the shell tool, the tool loop and the grading, and changes four things; the settings
``instructions``, ``workspace_alias`` (both default true) and ``max_parse_retries`` (0 disables) turn off the first
three, so ``{instructions: false, workspace_alias: false, max_parse_retries: 0}`` is RolloutEngine's harness with
the metrics:

- Prompt. A task with a shell tool no longer ends with a separate ``Give your answer as plain text.`` user turn.
  One instruction is folded into the task message instead: the task's files are in the working directory and must
  be read with the shell tool, and the final answer is plain text; a task graded on written files is told to write
  its deliverables at the stated paths. Text-only tasks keep RolloutEngine's prompt.
- ``workspace/`` paths. Prompts name files as ``workspace/X`` while the machine starts inside ``/workspace``; a
  ``workspace -> .`` link in the working directory makes those paths resolve.
- Unparsed tool calls. A reply whose ``<tool_call>`` the server's parser could not read is answered with a tool
  response naming the error, up to ``max_parse_retries`` times, instead of ending the attempt with the raw call as
  the answer.
- Metrics. The final step records ``tool_calls``, ``parse_retry``, ``opened_input_file`` (tasks whose prompt names
  a file of the task) and ``wrote_output_file`` (tasks graded on written files).
"""

import json
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from loguru import logger
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.shell import ShellHarness, _call_tools, _ShellToolSession, task_prompt
from rolloutengine.models import LENGTH_STOP_REASON, ModelTurn, Prompt
from shellbox.machine import Command, Machine
from taskcompendium.chat import assistant_message
from taskcompendium.models import AssistantToolCalls, PlainText, ScriptGrader, TaskSpec
from taskcompendium.runtime.shell import BoundShellTool, resolve_shell_tool
from taskcompendium.submission import conversation_messages, submission_instruction

GUIDED_SHELL_SETTINGS = frozenset({"instructions", "workspace_alias", "max_parse_retries"})
DEFAULT_MAX_PARSE_RETRIES = 3
WORKSPACE_ALIAS = "workspace"
SETUP_TIMEOUT = 60.0
TOOL_CALL_MARKER = "<tool_call>"
TOOL_CALL_BODY = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
HARNESS_METRIC_KEYS = ("tool_calls", "parse_retry", "opened_input_file", "wrote_output_file")

FILES_INSTRUCTION = (
    "The files of this task are in your current working directory ({workdir}). Read them with the `{tool}` tool; "
    "do not assume their contents."
)
TOOL_INSTRUCTION = "You can run commands with the `{tool}` tool."
ANSWER_INSTRUCTION = "When you have finished, give your final answer as plain text."
DELIVERABLE_INSTRUCTION = (
    "Write each deliverable file at the path the task states (relative paths are relative to the working "
    "directory) with the `{tool}` tool. When the files are written, reply with a short summary."
)
PARSE_ERROR = (
    "Error: your tool call could not be parsed ({error}). Write exactly one JSON object "
    '{{"name": "{tool}", "arguments": {{"{parameter}": "..."}}}} inside <tool_call></tool_call>, with every '
    "string JSON-escaped (no \\' escapes), and try again."
)


@dataclass(frozen=True)
class TaskFiles:
    """What the harness knows about a task's files: the inputs its prompt names and the outputs it is graded on."""

    workdir: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    names_workspace: bool


def task_files(task: TaskSpec) -> TaskFiles:
    prompt = "\n".join(str(message.get("content") or "") for message in conversation_messages(task.context.events))
    workdir = task.environment_requirements.working_directory or "/workspace"
    resources = (*task.resources.all, *task.resources.worker)
    inputs = tuple(sorted({resource.path for resource in resources if PurePosixPath(resource.path).name in prompt}))
    graded_on_files = isinstance(task.grader, ScriptGrader) and task.grader.answer_path is None
    return TaskFiles(
        workdir=workdir,
        inputs=inputs,
        outputs=tuple(task.output_paths) if graded_on_files else (),
        names_workspace=f"{WORKSPACE_ALIAS}/" in prompt,
    )


def guided_prompt(task: TaskSpec, shell_tool: BoundShellTool | None, files: TaskFiles) -> Prompt:
    """RolloutEngine's prompt, with the trailing answer instruction folded into the task message for tool tasks."""
    prompt = task_prompt(task, shell_tool, ())
    if shell_tool is None:
        return prompt
    messages = conversation_messages(task.context.events)
    tool = shell_tool.definition.name
    sentences = [FILES_INSTRUCTION if files.inputs else TOOL_INSTRUCTION]
    if files.outputs:
        sentences.append(DELIVERABLE_INSTRUCTION)
    elif isinstance(task.answer_format, PlainText):
        sentences.append(ANSWER_INSTRUCTION)
    else:
        sentences.append(submission_instruction(task.answer_format))
    guidance = " ".join(sentence.format(workdir=files.workdir, tool=tool) for sentence in sentences if sentence)
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
        messages[-1] = {**messages[-1], "content": f"{messages[-1]['content'].rstrip()}\n\n{guidance}"}
    else:
        messages.append({"role": "user", "content": guidance})
    return Prompt(tuple(messages), prompt.options)


def unparsed_tool_call(message: Mapping[str, Any]) -> str | None:
    """The parse error of a reply that wrote a ``<tool_call>`` the server's parser did not return, else None."""
    content = message.get("content")
    if message.get("tool_calls") or not isinstance(content, str) or TOOL_CALL_MARKER not in content:
        return None
    match = TOOL_CALL_BODY.search(content)
    body = match.group(1).strip() if match else ""
    try:
        json.loads(body)
    except json.JSONDecodeError as error:
        return f"invalid JSON: {error.msg} at character {error.pos}"
    return "the call was not recognized as a tool call"


@dataclass
class _Tracker:
    files: TaskFiles
    shell_tool: BoundShellTool | None
    tool_calls: int = 0
    parse_retry: int = 0
    opened_input_file: bool = False
    commands: list[str] = field(default_factory=list)

    def observe(self, turn: ModelTurn) -> None:
        try:
            message = assistant_message(turn.message)
        except (TypeError, ValueError):
            return
        if not isinstance(message, AssistantToolCalls):
            return
        self.tool_calls += len(message.calls)
        if self.shell_tool is None:
            return
        for call in message.calls:
            command = (
                call.arguments.get(self.shell_tool.command_parameter) if isinstance(call.arguments, dict) else None
            )
            if isinstance(command, str) and any(PurePosixPath(path).name in command for path in self.files.inputs):
                self.opened_input_file = True

    def metrics(self, wrote_output_file: bool | None) -> dict[str, float]:
        metrics = {"tool_calls": float(self.tool_calls), "parse_retry": float(self.parse_retry)}
        if self.files.inputs:
            metrics["opened_input_file"] = float(self.opened_input_file)
        if wrote_output_file is not None:
            metrics["wrote_output_file"] = float(wrote_output_file)
        return metrics


async def _outputs_written(machine: Machine | None, files: TaskFiles) -> bool | None:
    if not files.outputs:
        return None
    if machine is None:
        return False
    test = " && ".join(f"test -e {shlex.quote(path)}" for path in files.outputs)
    result = await machine.run(Command(("sh", "-c", test), timeout=SETUP_TIMEOUT, user="0"))
    return result.exit_code == 0


async def _link_workspace(machine: Machine, files: TaskFiles) -> None:
    """Make ``workspace/X`` name ``X`` when the machine starts inside a directory called ``workspace``."""
    if not files.names_workspace or PurePosixPath(files.workdir).name != WORKSPACE_ALIAS:
        return
    command = f"[ -e {WORKSPACE_ALIAS} ] || ln -s . {WORKSPACE_ALIAS}"
    result = await machine.run(Command(("sh", "-c", command), timeout=SETUP_TIMEOUT))
    if result.exit_code != 0:
        raise RuntimeError(f"Linking {WORKSPACE_ALIAS} -> . failed: {result.stderr.decode(errors='replace')[-500:]}")


@dataclass(frozen=True, kw_only=True)
class GuidedShellHarness(ShellHarness):
    """``ShellHarness`` with folded workspace instructions, a ``workspace`` alias, parse retries and tool metrics."""

    instructions: bool = True
    """Fold the workspace instruction into the task message; False keeps RolloutEngine's prompt."""
    workspace_alias: bool = True
    max_parse_retries: int = DEFAULT_MAX_PARSE_RETRIES

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        task, machine = context.task, context.machine
        if task.environment_requirements.tool_providers:
            await super().run(context, session)
            return
        shell_tool = resolve_shell_tool(task, self.shell_tool)
        if shell_tool is not None and machine is None:
            raise ValueError("Shell tools require a task machine")
        tools = {}
        if machine is not None and shell_tool is not None:
            tools[shell_tool.definition.name] = _ShellToolSession(machine, shell_tool, self.command_timeout)
        files = task_files(task)
        if machine is not None and shell_tool is not None:
            if self.workspace_alias:
                await _link_workspace(machine, files)
        tracker = _Tracker(files, shell_tool)
        prompt = guided_prompt(task, shell_tool, files) if self.instructions else task_prompt(task, shell_tool, ())
        turn = await session.start(prompt)
        try:
            while True:
                tracker.observe(turn)
                error = None if shell_tool is None else unparsed_tool_call(turn.message)
                if (
                    error is not None
                    and turn.stop_reason != LENGTH_STOP_REASON
                    and tracker.parse_retry < self.max_parse_retries
                ):
                    tracker.parse_retry += 1
                    feedback = PARSE_ERROR.format(
                        error=error, tool=shell_tool.definition.name, parameter=shell_tool.command_parameter
                    )
                    turn = await session.reply(({"role": "user", "content": _tool_response(feedback)},))
                    continue
                transition = await _call_tools(task, machine, tools, context.events, turn)
                if transition.done:
                    metrics = tracker.metrics(await _outputs_written(machine, files))
                    session.record(observations=transition.observations, metrics={**transition.metrics, **metrics})
                    return
                turn = await session.reply(transition.observations)
        except Exception:
            # A limit or a failed model request ends the attempt; its tool metrics still belong to the last step.
            try:
                session.record(metrics=tracker.metrics(await _outputs_written(machine, files)))
            except Exception:
                logger.opt(exception=True).warning("Recording guided-shell metrics after a failed attempt failed")
            raise


def _tool_response(content: str) -> str:
    """A user turn the Qwen chat templates render exactly like a tool response."""
    return f"<tool_response>\n{content}\n</tool_response>"


def guided_shell_harness(base: ShellHarness, settings: Mapping[str, Any]) -> GuidedShellHarness:
    unknown = set(settings) - GUIDED_SHELL_SETTINGS
    if unknown:
        raise ValueError(f"Unknown environment.task_sessions.guided_shell settings: {sorted(unknown)}")
    return GuidedShellHarness(
        command_timeout=base.command_timeout,
        shell_tool=base.shell_tool,
        tools=base.tools,
        instructions=bool(settings.get("instructions", True)),
        workspace_alias=bool(settings.get("workspace_alias", True)),
        max_parse_retries=int(settings.get("max_parse_retries", DEFAULT_MAX_PARSE_RETRIES)),
    )


def harness_metrics(step_metrics: Mapping[str, Any]) -> dict[str, float]:
    """The guided-shell metrics among a final step's metrics."""
    return {key: float(step_metrics[key]) for key in HARNESS_METRIC_KEYS if key in step_metrics}
