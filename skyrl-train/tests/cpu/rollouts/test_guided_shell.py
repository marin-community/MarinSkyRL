"""The guided shell harness: folded workspace instructions, the workspace alias, parse retries and tool metrics."""

import json
from dataclasses import replace

import pytest
from rolloutengine.harness.shell import task_prompt
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, PackageSpec
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ResourceGroups
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellToolConfig, resolve_shell_tool

from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.metric_names import TASK_ROLLOUT_METRIC_PREFIX
from skyrl_train.rollouts.guided_shell import (
    ANSWER_INSTRUCTION,
    DELIVERABLE_INSTRUCTION,
    _link_workspace,
    _outputs_written,
    guided_prompt,
    task_files,
    unparsed_tool_call,
)
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from tests.cpu.rollouts.engine_fakes import ConversationClient, Tokenizer
from tests.cpu.task_specs import lowered_task, task_input

PROMPT = "Read workspace/data.txt and report six plus the number in it."


@pytest.fixture
def numeric_task(task_inputs):
    """The conftest arithmetic task's grader on a ShellSim workspace task that names its input file."""
    _, request = task_inputs
    base = request.rows[0].spec.task
    task = base.model_copy(
        update={
            "context": base.context.model_copy(
                update={"events": (base.context.events[0].model_copy(update={"content": PROMPT}),)}
            ),
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.SHELL_SIMULATOR,
                capabilities=("shell", "filesystem"),
                working_directory="/workspace",
            ),
            "resources": ResourceGroups(worker=(inline_resource("workspace/data.txt", b"6"),)),
        }
    )
    return task


def _shell(task):
    return resolve_shell_tool(task, ShellToolConfig())


def test_tool_task_folds_one_instruction_into_the_task_message(numeric_task):
    prompt = guided_prompt(numeric_task, _shell(numeric_task), task_files(numeric_task))

    (message,) = prompt.messages
    assert message["role"] == "user"
    assert message["content"].startswith(PROMPT)
    assert "current working directory (/workspace)" in message["content"]
    assert message["content"].endswith(ANSWER_INSTRUCTION)
    assert "Give your answer as plain text." not in message["content"]
    assert prompt.options == task_prompt(numeric_task, _shell(numeric_task), ()).options


def test_file_graded_task_is_told_to_write_its_deliverables(numeric_task):
    files = replace(task_files(numeric_task), outputs=("/workspace/report.md",))

    (message,) = guided_prompt(numeric_task, _shell(numeric_task), files).messages

    assert message["content"].endswith(DELIVERABLE_INSTRUCTION.format(tool="shell"))
    assert ANSWER_INSTRUCTION not in message["content"]


def test_text_only_task_keeps_the_rolloutengine_prompt(task_inputs):
    _, request = task_inputs
    task = request.rows[0].spec.task

    assert guided_prompt(task, None, task_files(task)) == task_prompt(task, None, ())


@pytest.mark.parametrize(
    ("message", "unparsed"),
    [
        ({"content": '<tool_call>\n{"name": "shell", "arguments": {\'command\': \'ls\'}}\n</tool_call>'}, True),
        ({"content": "The answer is 12."}, False),
        ({"content": "", "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}]}, False),
    ],
)
def test_unparsed_tool_calls_are_recognized(message, unparsed):
    assert (unparsed_tool_call(message) is not None) is unparsed


@pytest.mark.asyncio
async def test_guided_shell_resolves_workspace_paths_retries_parse_errors_and_records_metrics(
    task_inputs, numeric_task
):
    config, request = task_inputs
    lowered = lowered_task(numeric_task, backend="shellsim", max_turns=4, command_timeout=10, tool_turn_timeout=20)
    request = replace(request, rows=(task_input(lowered),))
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": json.dumps({"command": "cat workspace/data.txt"})},
                }
            ],
        },
        {"role": "assistant", "content": '<tool_call>\n{"name": "shell", \'arguments\': {}}\n</tool_call>'},
        {"role": "assistant", "content": "12"},
    ]
    model = ConversationClient(["", "", "12"], messages=messages)
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, Tokenizer()),
        model,
        {"shellsim": ShellSimMachineFactory()},
        shutdown_timeout=30,
        guided_shell={},
    )
    try:
        (rollout,) = await worker.generate(request)
        batch = await worker.training_batch(request, [rollout])
    finally:
        await worker.shutdown()

    first_prompt = model.requests[0]["prompts"][0]
    assert [message["role"] for message in first_prompt] == ["user"]
    observation = json.loads(model.requests[1]["prompts"][0][-1]["content"])
    assert (observation["exit_code"], observation["stdout"]) == (0, "6")
    retry = model.requests[2]["prompts"][0][-1]
    assert retry["role"] == "user" and retry["content"].startswith("<tool_response>\nError: your tool call")
    assert rollout.grade.reward == 1.0
    expected = {"tool_calls": 1.0, "parse_retry": 1.0, "opened_input_file": 1.0}
    assert batch["env_metrics"] == [expected]
    metrics = batch["rollout_metrics"]
    assert metrics[f"{TASK_ROLLOUT_METRIC_PREFIX}opened_input_file"] == 1.0
    assert metrics[f"{TASK_ROLLOUT_METRIC_PREFIX}opened_input_file_tasks"] == 1.0
    assert metrics["environment/tool_calls"] == 1.0
    eval_metrics = _calculate_eval_metrics(batch, [rollout.task_id], ["fixture"], 1)
    assert eval_metrics["eval/all/environment/taskcompendium/opened_input_file"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("written", [True, False])
async def test_written_deliverables_are_detected_on_the_task_machine(numeric_task, written):
    machine = await ShellSimMachineFactory().create(MachineSpec(source=PackageSpec(), workdir="/workspace"))
    files = replace(task_files(numeric_task), outputs=("/workspace/report/summary.md",))
    try:
        await _link_workspace(machine, files)
        if written:
            await machine.run(
                Command(("sh", "-c", "mkdir -p workspace/report && echo ok > workspace/report/summary.md"))
            )
        assert await _outputs_written(machine, files) is written
    finally:
        await machine.close()
