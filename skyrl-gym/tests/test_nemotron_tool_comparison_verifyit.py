"""Source parity for the verifyit backed Nemotron tool comparison route."""

import json
import math

import pytest
import skyrl_gym
from skyrl_gym.verification import RolloutEvidence
from verifyit.grade import InvalidTask, Status
from verifyit.preparation.errors import InvalidPreparation

from skyrl_gym.envs.nemotron_ultra.tool_call import grade_expected_action
from skyrl_gym.envs.nemotron_ultra.tool_comparison_verifyit import (
    grade_expected_action_verifyit,
    grade_prepared_tool_action,
    prepare_tool_action,
    structure_tool_action,
)


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
    with pytest.raises(InvalidTask, match="numeric expected"):
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


@pytest.mark.parametrize(
    "value,score", [(math.nextafter(1e-6, 0.0), 1.0), (1e-6, 0.0), (math.nextafter(1e-6, math.inf), 0.0)]
)
def test_numeric_strict_threshold_is_owned_by_core(value, score):
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 0.0}'}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": json.dumps({"value": value})}}]}
    assert grade_expected_action_verifyit(expected, assistant)[0] == score


@pytest.mark.parametrize("actual,score", [("null", 1.0), ("false", 0.0), ("[]", 0.0)])
def test_nested_null_contract(actual, score):
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": [null]}'}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": [' + actual + "]}"}}]}
    assert grade_expected_action_verifyit(expected, assistant)[0] == score


def test_duplicate_candidate_keys_fail_closed():
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 1}'}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": 0, "value": 1}'}}]}
    assert grade_expected_action_verifyit(expected, assistant)[0] == 0.0


def test_missing_call_receives_core_structural_failure():
    expected = {"type": "function_call", "name": "transfer", "arguments": "{}"}
    assert grade_expected_action_verifyit(expected, {"tool_calls": []})[0] == 0.0


@pytest.mark.parametrize("calls,score", [(None, 1.0), ([], 1.0), (False, 0.0), (0, 0.0), ("", 0.0), ({}, 0.0)])
def test_message_tool_calls_require_array_or_absent(calls, score):
    expected = {"type": "message", "content": "done"}
    assistant = {"content": "done", "tool_calls": calls}
    assert grade_expected_action(expected, assistant)[0] == 1.0  # Source erases falsey malformed values.
    assert grade_expected_action_verifyit(expected, assistant)[0] == score


def test_structural_snapshot_preserves_complete_inputs_and_isolates_source_mutations():
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 1}', "metadata": ["trusted"]}
    assistant = {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": 1}'}}], "extra": None}
    inputs = structure_tool_action(expected, assistant)
    assert inputs.expected_action == expected
    assert inputs.assistant_message == assistant
    expected["metadata"].append("changed")
    assistant["tool_calls"][0]["function"]["arguments"] = '{"value": 2}'
    prepared = prepare_tool_action(inputs)
    assert inputs.expected_action["metadata"] == ["trusted"]
    assert inputs.assistant_message["tool_calls"][0]["function"]["arguments"] == '{"value": 1}'
    assert grade_prepared_tool_action(prepared)[0] == 1.0
    assert grade_expected_action_verifyit(expected, assistant)[0] == 0.0


def test_invalid_preparation_finalizes_without_partial_success():
    inputs = structure_tool_action(
        {"type": "function_call", "name": "transfer", "arguments": '{"value": 1,"value": 2}'},
        {"tool_calls": []},
    )
    with pytest.raises(InvalidPreparation) as error:
        prepare_tool_action(inputs)
    assert error.value.verdict.status == Status.INVALID_TASK
    assert error.value.verdict.reward == 0.0
    assert error.value.failure.stage == "tool_policy"


def test_message_preparation_does_not_share_schema_between_tasks():
    inputs = structure_tool_action({"type": "message"}, {"content": "done"})
    first = prepare_tool_action(inputs)
    first.schema["properties"]["content"]["pattern"] = "never matches"
    assert grade_prepared_tool_action(prepare_tool_action(inputs))[0] == 1.0


@pytest.mark.parametrize(
    "agent",
    [
        "single_step_tool_use_with_argument_comparison_agent",
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
    ],
)
def test_tool_factory_reports_effective_policy_without_exposing_reference(agent):
    expected = {"type": "function_call", "name": "transfer", "arguments": '{"value": 1}'}
    ultra = {
        "route": "skyrl_gym",
        "agent": agent,
        "record_json": json.dumps({"expected_action": expected}),
        "request_json": "{}",
    }
    env = skyrl_gym.make(
        "nemotron_ultra", env_config={"verifyit_enabled": True}, extras={"extra_info": {"nemotron_ultra": ultra}}
    )
    env.evidence = RolloutEvidence(
        messages=(),
        response="",
        metadata={
            "assistant_message": {"tool_calls": [{"function": {"name": "transfer", "arguments": '{"value": 1}'}}]}
        },
    )
    try:
        result = env.step("")
    finally:
        env.close()
    assert result["reward"] == 1.0
    assert result["verification"].status.value == "verified"
    provenance = result["metadata"]["preparation"]
    assert provenance["policy"] == "nemotron_strict_typed_arguments_v1"
    assert set(provenance) == {"policy", "expected_action_sha256", "assistant_message_sha256"}


@pytest.mark.parametrize(
    "kind,arguments,status",
    [
        ("candidate", '{"amount":' + str(10**400) + "}", "verified"),
        ("candidate", "[" * 1200 + "0" + "]" * 1200, "verified"),
        ("trusted_type", "{}", "error"),
        ("trusted_depth", "[" * 1200 + "0" + "]" * 1200, "error"),
    ],
)
def test_swe_factory_classifies_extreme_candidates_and_invalid_trusted_actions(kind, arguments, status):
    expected = {
        "type": [] if kind == "trusted_type" else "function_call",
        "name": "transfer",
        "arguments": arguments if kind == "trusted_depth" else '{"amount":0.5}',
    }
    ultra = {
        "route": "skyrl_gym",
        "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "record_json": json.dumps({"expected_action": expected}),
        "request_json": "{}",
    }
    env = skyrl_gym.make(
        "nemotron_ultra", env_config={"verifyit_enabled": True}, extras={"extra_info": {"nemotron_ultra": ultra}}
    )
    env.evidence = RolloutEvidence(
        messages=(),
        response="",
        metadata={"assistant_message": {"tool_calls": [{"function": {"name": "transfer", "arguments": arguments}}]}},
    )
    try:
        result = env.step("")
    finally:
        env.close()
    assert result["reward"] == 0.0
    assert result["verification"].status.value == status
    if kind.startswith("trusted"):
        assert result["metadata"]["verifyit_status"] == "invalid_task"
        assert result["metadata"]["preparation"]["policy"] == "nemotron_strict_typed_arguments_v1"
        assert result["metadata"]["preparation"]["stage"] == "tool_policy"
