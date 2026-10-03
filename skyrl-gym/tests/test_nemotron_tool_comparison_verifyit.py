"""Source parity for the verifyit backed Nemotron tool comparison route."""

import json

import pytest

from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.envs.nemotron_ultra.tool_comparison_verifyit import grade_expected_action_verifyit


@pytest.mark.parametrize(
    ("expected_arguments", "actual_arguments"),
    [
        ({"name": "Alice", "items": [1, {"weight": 0.5}]}, {"items": [1, {"weight": 0.5000005}], "name": "Alice"}),
        ({"weight": 0.5}, {"weight": 0.500001}),
        ({"items": ["first", "second"]}, {"items": ["second", "first"]}),
        ({"items": [1]}, {"items": [1, 2]}),
        ({"items": [1]}, {"other": [1]}),
        ({"value": 1}, {"value": 1.0}),
        ({"value": None}, {"value": False}),
    ],
)
def test_tool_argument_comparison_matches_pinned_source(expected_arguments, actual_arguments):
    expected = {"type": "function_call", "name": "transfer", "arguments": json.dumps(expected_arguments)}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": json.dumps(actual_arguments)}}]}
    assert grade_expected_action_verifyit(expected, assistant) == grade_expected_action(expected, assistant)


@pytest.mark.parametrize(
    "assistant",
    [
        {"content": "done", "tool_calls": []},
        {"tool_calls": []},
        {"tool_calls": [{"function": {"name": "wrong", "arguments": "{}"}}]},
        {"tool_calls": [{"function": {"name": "transfer", "arguments": "{"}}]},
        {"tool_calls": [{"function": {"name": "transfer", "arguments": "{}"}}] * 2},
    ],
)
def test_tool_action_failures_match_pinned_source(assistant):
    expected = {"type": "function_call", "name": "transfer", "arguments": "{}"}
    assert grade_expected_action_verifyit(expected, assistant) == grade_expected_action(expected, assistant)


def test_tool_message_action_matches_pinned_source():
    expected = {"type": "message", "content": "done"}
    for assistant in ({"content": "done", "tool_calls": []}, {"content": "done", "tool_calls": [{}]}):
        assert grade_expected_action_verifyit(expected, assistant) == grade_expected_action(expected, assistant)


def test_invalid_reference_cannot_be_scored_as_candidate_failure():
    with pytest.raises(ValueError, match="expected tool arguments"):
        grade_expected_action_verifyit(
            {"type": "function_call", "name": "transfer", "arguments": '{"amount": Infinity}'},
            {"tool_calls": [{"function": {"name": "transfer", "arguments": "{}"}}]},
        )


def test_boolean_does_not_match_integer_argument():
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 1}'}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": true}'}}]}
    assert grade_expected_action(expected, assistant)[0] == 1.0  # Source Python isinstance/equality bug.
    assert grade_expected_action_verifyit(expected, assistant)[0] == 0.0


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_candidate_cannot_receive_credit(value):
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 1.0}'}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": ' + value + "}"}}]}
    assert grade_expected_action_verifyit(expected, assistant)[0] == 0.0
