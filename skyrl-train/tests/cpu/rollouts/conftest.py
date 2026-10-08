import json

import pytest
from omegaconf import OmegaConf
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, Source, EnvironmentRequirements, TaskSpec, TextMessage
from skyrl_gym.source_task import source_task
from tests.cpu.task_specs import lowered_task
from skyrl_train.trajectory_runners.types import TrajectoryID


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer("12", tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 2,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = {
        "prompts": [[{"role": "user", "content": "What is six plus six?"}]],
        "env_classes": ["taskcompendium"],
        "env_extras": [
            {
                "lowered_task_spec": lowered_task(task, "shellbox").model_dump_json(),
                "teacher_route": "arithmetic",
                "data_source": "arithmetic",
            }
        ],
        "trajectory_ids": [TrajectoryID("arithmetic", 0)],
        "sampling_params": None,
        "batch_metadata": None,
    }
    return config, request


@pytest.fixture
def python_tool_task(task_inputs):
    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "ns_tools_simple_agent",
                    "record_json": json.dumps({"question": "What is 2 + 2?", "expected_answer": "4"}),
                    "request_json": "{}",
                }
            }
        },
        config={},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "python-1",
                "type": "function",
                "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"2 + 2"}'},
            }
        ],
    }
    task = lowered_task(task, "nemotron_ultra", backend="shellsim")
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    return task, message
