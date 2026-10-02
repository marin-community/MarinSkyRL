"""Calendar translations retain source intervals through core schema constraints."""

import json

import pytest
from omegaconf import OmegaConf
from verifyit.grade import InvalidTask

from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar
from skyrl_gym.envs.nemotron_ultra.calendar_verifyit import grade_calendar_verifyit
from skyrl_gym.envs.nemotron_ultra.env import NemotronUltraEnv
from skyrl_gym.verification import VerificationStatus


@pytest.mark.parametrize(
    "events,expected_reason",
    [
        (
            [
                {"event_id": "a", "start_time": "09:00", "duration": 30},
                {"event_id": "b", "start_time": "09:30", "duration": 30},
            ],
            "pass",
        ),
        (
            [
                {"event_id": "a", "start_time": "09:00", "duration": 30},
                {"event_id": "b", "start_time": "09:15", "duration": 30},
            ],
            "conflicting_events",
        ),
        (
            [
                {"event_id": "a", "start_time": "08:00", "duration": 30},
                {"event_id": "b", "start_time": "09:30", "duration": 30},
            ],
            "constraint_violated",
        ),
    ],
)
def test_calendar_schema_preserves_touching_overlap_and_windows(events, expected_reason):
    expected = {
        name: {"duration": 30, "min_time": "09:00", "max_time": "10:00", "constraint": None} for name in ["a", "b"]
    }
    response = json.dumps(events)
    assert grade_calendar_verifyit(response, expected) == grade_calendar(response, expected)
    assert grade_calendar_verifyit(response, expected) == (float(expected_reason == "pass"), expected_reason)


@pytest.mark.parametrize("response,expected", [("arbitrary", (1.0, "pass")), ("<think>bad", (0.0, "think_found"))])
def test_empty_schedule_retains_source_policy(response, expected):
    assert grade_calendar_verifyit(response, {}) == expected


def test_invalid_reference_is_not_awarded_calendar_success():
    with pytest.raises(InvalidTask, match="invalid calendar reference"):
        grade_calendar_verifyit("[]", {"a": {"duration": 30}})


@pytest.mark.parametrize(
    "event",
    [
        {"event_id": "a", "start_time": "09:00", "duration": True},
        {"event_id": "a", "start_time": "08:60", "duration": 1},
    ],
)
def test_malformed_candidate_cannot_exploit_python_time_or_bool_equality(event):
    expected = {"a": {"duration": 1, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    assert grade_calendar_verifyit(json.dumps([event]), expected)[0] == 0.0


@pytest.mark.parametrize("duration,constraint", [(-1, None), (True, None), (1200, "unknown instruction")])
def test_malformed_reference_fails_before_candidate_window_short_circuit(duration, constraint):
    expected = {"a": {"duration": duration, "min_time": "09:00", "max_time": "10:00", "constraint": constraint}}
    with pytest.raises(InvalidTask, match="invalid calendar reference"):
        grade_calendar_verifyit(json.dumps([{"event_id": "a", "start_time": "09:00", "duration": duration}]), expected)


def test_zero_duration_retains_source_permitted_empty_interval():
    expected = {"a": {"duration": 0, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    response = json.dumps([{"event_id": "a", "start_time": "09:00", "duration": 0}])
    assert grade_calendar(response, expected) == (1.0, "pass")
    assert grade_calendar_verifyit(response, expected) == (1.0, "pass")


@pytest.mark.parametrize(
    "start,duration,score", [("23:30", 30, 1.0), ("23:45", 30, 0.0), ("24:00", 0, 1.0), ("24:01", 0, 0.0)]
)
def test_day_endpoint_and_overflow(start, duration, score):
    expected = {"a": {"duration": duration, "min_time": "23:00", "max_time": "24:00", "constraint": None}}
    response = json.dumps([{"event_id": "a", "start_time": start, "duration": duration}])
    assert grade_calendar_verifyit(response, expected)[0] == score


def test_reversed_cross_midnight_reference_is_invalid():
    expected = {"a": {"duration": 30, "min_time": "23:00", "max_time": "01:00", "constraint": None}}
    with pytest.raises(InvalidTask):
        grade_calendar_verifyit("[]", expected)


def test_duplicate_zero_duration_events_cannot_hide_in_id_mapping():
    expected = {"a": {"duration": 0, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    response = json.dumps([{"event_id": "a", "start_time": "09:00", "duration": 0}] * 2)
    assert grade_calendar(response, expected)[0] == 1.0  # Source collapses IDs and zero intervals do not conflict.
    assert grade_calendar_verifyit(response, expected)[0] == 0.0


def test_ambiguous_calendar_fields_fail_closed():
    expected = {"a": {"duration": 30, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    response = '[{"event_id": "a", "start_time": "08:00", "start_time": "09:00", "duration": 30}]'
    assert grade_calendar(response, expected)[0] == 1.0
    assert grade_calendar_verifyit(response, expected)[0] == 0.0


def test_calendar_framework_invalid_task_returns_minimum_reward():
    env = NemotronUltraEnv(
        OmegaConf.create({"verifyit_enabled": True}),
        extras={
            "extra_info": {
                "nemotron_ultra": {
                    "route": "skyrl_gym",
                    "agent": "calendar_simple_agent",
                    "record_json": json.dumps({"exp_cal_state": {"a": {"duration": 30}}}),
                    "request_json": "{}",
                }
            }
        },
    )
    result = env.step("[]")
    assert result["reward"] == 0.0
    assert result["verification"].status is VerificationStatus.ERROR
    assert result["verification"].diagnostics["error_category"] == "invalid_task"


@pytest.mark.parametrize(
    "identifier,reference_id,score", [(True, "True", 1.0), (True, "1", 0.0), ({}, "{}", 1.0), (0.0, "0", 0.0)]
)
def test_event_ids_preserve_source_string_normalization(identifier, reference_id, score):
    expected = {reference_id: {"duration": 30, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    response = json.dumps([{"event_id": identifier, "start_time": "09:00", "duration": 30}])
    assert grade_calendar(response, expected)[0] == score
    assert grade_calendar_verifyit(response, expected)[0] == score
