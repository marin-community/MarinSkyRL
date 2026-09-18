import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.fixture
def searchcode_env():
    env = skyrl_gym.make(
        "searchcode",
        env_config=DictConfig({"env_class": "searchcode"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": "random"}, "max_turns": 2},
    )
    env.init([])
    return env


@pytest.mark.parametrize(
    "action, expected_tool_name, expected_tool_input",
    [
        (
            "<tool><search>how to reverse a string in Python</search></tool>",
            "search",
            ["how to reverse a string in Python"],
        ),
        ("<tool><python>print('hello')</python></tool>", "python", ["print('hello')"]),
        ("<tool><search>binary search in Java</search></tool>", "search", ["binary search in Java"]),
    ],
)
def test_tool_parsing(searchcode_env, action, expected_tool_name, expected_tool_input):
    # Test the parsing logic and reward logic with dummy input
    _, tool_name, tool_input = searchcode_env._parse_action(action)

    assert tool_name == expected_tool_name
    assert tool_input == expected_tool_input


@pytest.mark.parametrize(
    "action, expected_output",
    [
        (
            "<tool><python>print('hello world')</python></tool>",
            "hello world",
        ),
        (
            "<tool><python>print(1 + 1)</python></tool>",
            "2",
        ),
    ],
)
def test_python_code_execution(searchcode_env, action, expected_output):
    output = searchcode_env.step(action)
    observation_content = output["observations"][0]["content"]

    assert expected_output == observation_content


def test_python_code_execution_surfaces_exception_type_and_message(searchcode_env):
    output = searchcode_env.step("<tool><python>raise ValueError('fail')</python></tool>")
    observation_content = output["observations"][0]["content"]

    # The tool returns the interpreter's stderr verbatim, whose traceback rendering
    # differs across Python versions (3.13 echoes the failing source line).
    assert observation_content.startswith("Error executing Python code: Traceback")
    assert "ValueError: fail" in observation_content
