"""Source dialect and parsing parity at the structured-output client boundary."""

import json

import pytest
from jsonschema.validators import validator_for
from openapi_schema_validator import OAS32Validator

from skyrl_gym.envs.nemotron_ultra.structured_outputs import grade_structured_output
from skyrl_gym.envs.nemotron_ultra.structured_outputs_verifyit import (
    grade_structured_output_verifyit,
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
    assert (
        validator_for({"$schema": OAS32Validator.META_SCHEMA["$id"]}) is OAS32Validator
    )


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
