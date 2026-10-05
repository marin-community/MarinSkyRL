"""Source dialect and parsing parity at the structured-output client boundary."""

import hashlib
import json

import pytest
import skyrl_gym
from skyrl_gym.verification import RolloutEvidence
from jsonschema.validators import validator_for
from openapi_schema_validator import OAS32Validator

from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.structured_outputs_verifyit import (
    grade_structured_output_verifyit,
    structure_structured_output,
)


@pytest.mark.parametrize(
    "schema_type,text,schema,expected",
    [
        (
            "json",
            '{"x":1}',
            {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            1.0,
        ),
        (
            "json",
            '{"x":true}',
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            0.0,
        ),
        (
            "yaml",
            "x: 1",
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            1.0,
        ),
        (
            "yaml",
            "date: 2026-09-30",
            {"type": "object", "properties": {"date": {"type": "string"}}},
            0.0,
        ),
        (
            "toml",
            "x = 1",
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            1.0,
        ),
        (
            "xml",
            "<x>1</x>",
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            1.0,
        ),
        (
            "csv",
            "x\n1\n",
            {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
            1.0,
        ),
        ("json", '"not an email"', {"type": "string", "format": "email"}, 1.0),
        (
            "json",
            "1",
            {"$defs": {"value": {"type": "integer"}}, "$ref": "#/$defs/value"},
            1.0,
        ),
        (
            "json",
            '"x"',
            {"$defs": {"value": {"type": "integer"}}, "$ref": "#/$defs/value"},
            0.0,
        ),
        (
            "json",
            "1",
            {"$schema": "http://json-schema.org/draft-07/schema#", "type": "integer"},
            1.0,
        ),
        (
            "json",
            "true",
            {"$schema": "http://json-schema.org/draft-07/schema#", "type": "integer"},
            0.0,
        ),
    ],
)
def test_source_format_and_dialect_parity(schema_type, text, schema, expected):
    record = {"schema_type": schema_type, "schema_str": json.dumps(schema)}
    assert grade_structured_output(text, record, {})[0] == expected
    assert grade_structured_output_verifyit(text, record, {})[0] == expected


def test_registration_does_not_replace_source_dialect():
    assert validator_for({"$schema": OAS32Validator.META_SCHEMA["$id"]}) is OAS32Validator


def test_remote_reference_never_requests_network(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("Schema validation attempted network retrieval")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    record = {"schema_str": json.dumps({"$ref": "http://127.0.0.1:1/private-schema"})}
    assert grade_structured_output("1", record, {})[0] == 0.0
    assert grade_structured_output_verifyit("1", record, {})[0] == 0.0


def test_nonfinite_candidate_is_intentionally_rejected():
    record = {"schema_str": '{"type":"number"}'}
    assert grade_structured_output_verifyit("NaN", record, {})[0] == 0.0


def test_unknown_dialect_is_rejected():
    record = {"schema_str": '{"$schema":"urn:untrusted:unknown","type":"integer"}'}
    assert grade_structured_output_verifyit("1", record, {})[0] == 0.0


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ('{"payload":{"x":1}}', 1.0),
        ('{"payload":{"x":true}}', 0.0),
        ('{"other":1}', 0.0),
    ],
)
def test_tool_payload_parity(arguments, expected):
    record = {
        "response_mode": "tool_call",
        "tool_name": "answer",
        "tool_payload_key": "payload",
        "schema_str": '{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}',
    }
    message = {"tool_calls": [{"function": {"name": "answer", "arguments": arguments}}]}
    assert grade_structured_output("", record, message)[0] == expected
    assert grade_structured_output_verifyit("", record, message)[0] == expected


@pytest.mark.parametrize(
    "name,arguments,expected,error_type",
    [
        ("answer", '{"payload":{"x":1}}', 1.0, None),
        ("other", '{"payload":{"x":1}}', 0.0, "wrong_tool_name"),
        ("other", "{", 0.0, "wrong_tool_name"),
        ("answer", "{", 0.0, "tool_arguments_parse_error"),
    ],
)
def test_name_and_payload_verdicts_preserve_source_precedence(name, arguments, expected, error_type):
    record = {
        "response_mode": "tool_call",
        "tool_name": "answer",
        "tool_payload_key": "payload",
        "schema_str": '{"type":"object","required":["x"],"properties":{"x":{"type":"integer"}}}',
    }
    message = {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}
    reward, details = grade_structured_output_verifyit("", record, message)
    assert reward == grade_structured_output("", record, message)[0] == expected
    assert details["error_type"] == error_type
    assert details["status"] == "scored"
    assert details["preparation"]["assistant_message_sha256"] == hashlib.sha256(repr(message).encode()).hexdigest()
    if error_type == "wrong_tool_name":
        assert details["verdict"]["detail"]["missing"] == 1


@pytest.mark.parametrize(
    "schema_type,text,schema,decoded,effective",
    [
        ("xml", "<x>01</x>", {"type": "object", "properties": {"x": {"type": "integer"}}}, "{'x': '01'}", "{'x': 1}"),
        ("xml", "<x/>", {"type": "object", "properties": {"x": {"type": "string"}}}, "{'x': None}", "{'x': ''}"),
        (
            "csv",
            "x,y\n,1\n",
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": ["integer", "null"]}, "y": {"type": "integer"}},
                },
            },
            "[{'x': '', 'y': '1'}]",
            "[{'x': None, 'y': 1}]",
        ),
    ],
)
def test_source_coercions_retain_raw_and_effective_provenance(schema_type, text, schema, decoded, effective):
    record = {"schema_type": schema_type, "schema_str": json.dumps(schema)}
    reward, details = grade_structured_output_verifyit(text, record, {})
    assert reward == grade_structured_output(text, record, {})[0] == 1.0
    preparation = details["preparation"]
    assert preparation["policy"] == "nemotron_structured_output_source_v1"
    assert preparation["text_sha256"] == hashlib.sha256(repr(text).encode()).hexdigest()
    assert "raw" not in preparation
    assert preparation["decoded_repr"] == decoded
    assert preparation["effective_repr"] == effective


def test_snapshot_isolates_record_and_tool_arguments():
    record = {"schema_str": '{"type":"integer"}', "metadata": ["trusted"]}
    message = {"tool_calls": [{"function": {"arguments": {"x": 1}}}]}
    inputs = structure_structured_output("01", record, message)
    record["metadata"].append("changed")
    message["tool_calls"][0]["function"]["arguments"]["x"] = 2
    assert inputs.record["metadata"] == ["trusted"]
    assert inputs.assistant_message["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert inputs.text == "01"


@pytest.mark.parametrize("agent", ["structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"])
@pytest.mark.parametrize(
    "schema,action,reward,status,core_status",
    [
        ('{"type":"integer"}', "1", 1.0, "verified", "scored"),
        ('{"type":"integer"}', "broken", 0.0, "verified", "scored"),
        ('{"type":"broken"}', "", 0.0, "error", "invalid_task"),
    ],
)
def test_framework_preserves_invalid_task_versus_candidate_zero(agent, schema, action, reward, status, core_status):
    record = {"schema_str": schema}
    env = skyrl_gym.make(
        "nemotron_ultra",
        env_config={"verifyit_enabled": True},
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": agent,
                    "record_json": json.dumps(record),
                    "request_json": "{}",
                }
            }
        },
    )
    try:
        result = env.step(action)
    finally:
        env.close()
    assert result["reward"] == reward
    assert result["verification"].status.value == status
    assert result["metadata"]["status"] == core_status
    assert result["metadata"]["preparation"]["record_sha256"] == hashlib.sha256(repr(record).encode()).hexdigest()
    assert "schema_str" not in result["metadata"]["preparation"]


def test_unknown_policy_cannot_turn_bad_candidate_into_scored_zero():
    reward, details = grade_structured_output_verifyit("", {"schema_str": "{}"}, {}, policy="unsupported")
    assert reward == 0.0
    assert details["status"] == "invalid_task"
    assert details["verdict"]["detail"]["finalization_policy"] == "failed_preparation_v1"


def test_irrelevant_non_json_metadata_does_not_change_candidate_grade():
    record = {"schema_str": '{"type":"integer"}', "metadata": {"tags": {"one", "two"}}}
    message = {"metadata": {"binary": b"abc"}}
    assert (
        grade_structured_output_verifyit("1", record, message)[0]
        == grade_structured_output("1", record, message)[0]
        == 1.0
    )


@pytest.mark.parametrize(
    "trusted,schema,expected",
    [
        (True, '{"type":"integer"}', "invalid_task"),
        (False, '{"type":"integer"}', "scored"),
        (False, '{"type":"bad"}', "invalid_task"),
    ],
)
def test_nested_input_failure_preserves_task_candidate_identity(trusted, schema, expected):
    nested = []
    for _ in range(2000):
        nested = [nested]
    record = {"schema_str": schema}
    message = {}
    (record if trusted else message)["metadata"] = nested
    reward, details = grade_structured_output_verifyit("1", record, message)
    assert reward == 0.0
    assert details["status"] == expected
    assert details["verdict"]["detail"]["finalization_policy"] == "failed_preparation_v1"


@pytest.mark.parametrize("agent", ["structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"])
@pytest.mark.parametrize(
    "record,action,assistant_message,core_status",
    [
        ({"schema_str": '{"const":"TRUSTED_REFERENCE_SENTINEL"}'}, "0", {}, "scored"),
        ({"schema_str": '{"type":"TRUSTED_REFERENCE_SENTINEL"}'}, "0", {}, "invalid_task"),
        (
            {"schema_str": "{}", "response_mode": "tool_call", "tool_name": "TRUSTED_REFERENCE_SENTINEL"},
            "",
            {"tool_calls": [{"function": {"name": "other", "arguments": "{}"}}]},
            "scored",
        ),
        (
            {"schema_str": "{}", "response_mode": "tool_call", "tool_payload_key": "TRUSTED_REFERENCE_SENTINEL"},
            "",
            {"tool_calls": [{"function": {"arguments": "{}"}}]},
            "scored",
        ),
    ],
)
def test_framework_diagnostics_do_not_disclose_trusted_literals(agent, record, action, assistant_message, core_status):
    env = skyrl_gym.make(
        "nemotron_ultra",
        env_config={"verifyit_enabled": True},
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": agent,
                    "record_json": json.dumps(record),
                    "request_json": "{}",
                }
            }
        },
    )
    try:
        env.set_rollout_evidence(
            RolloutEvidence(messages=(), response=action, metadata={"assistant_message": assistant_message})
        )
        result = env.step(action)
    finally:
        env.close()
    assert result["reward"] == 0.0
    assert result["metadata"]["status"] == core_status
    assert "TRUSTED_REFERENCE_SENTINEL" not in repr(result)


@pytest.mark.parametrize("agent", ["structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"])
@pytest.mark.parametrize(
    "arguments",
    ['{"x":' + "9" * 5000 + "}", "[" * 2000 + "0" + "]" * 2000],
    ids=["integer_digit_limit", "nesting_limit"],
)
@pytest.mark.parametrize(
    "schema,core_status,framework_status",
    [('{"type":"object"}', "scored", "verified"), ('{"type":"bad"}', "invalid_task", "error")],
)
def test_argument_decoder_limits_preserve_candidate_and_task_status(
    agent, arguments, schema, core_status, framework_status
):
    record = {"schema_str": schema, "response_mode": "tool_call", "tool_name": "answer"}
    env = skyrl_gym.make(
        "nemotron_ultra",
        env_config={"verifyit_enabled": True},
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": agent,
                    "record_json": json.dumps(record),
                    "request_json": "{}",
                }
            }
        },
    )
    try:
        env.set_rollout_evidence(
            RolloutEvidence(
                messages=(),
                response="",
                metadata={
                    "assistant_message": {"tool_calls": [{"function": {"name": "answer", "arguments": arguments}}]}
                },
            )
        )
        result = env.step("")
    finally:
        env.close()
    assert result["reward"] == 0.0
    assert result["metadata"]["status"] == core_status
    assert result["verification"].status.value == framework_status
    assert result["metadata"]["verdict"]["detail"]["finalization_policy"] == "failed_preparation_v1"
