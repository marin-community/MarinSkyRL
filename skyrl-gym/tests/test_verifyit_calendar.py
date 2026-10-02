"""Task-owned calendar harness preserves source constraints through real ScriptSpec."""

import json

import pytest

from skyrl_gym.envs.nemotron_ultra.calendar import grade_calendar
from skyrl_gym.envs.nemotron_ultra.calendar_verifyit import grade_calendar_verifyit


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
def test_calendar_script_preserves_touching_overlap_and_windows(events, expected_reason):
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
    with pytest.raises(RuntimeError, match="verification failed"):
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
    with pytest.raises(RuntimeError, match="verification failed"):
        grade_calendar_verifyit(json.dumps([{"event_id": "a", "start_time": "09:00", "duration": duration}]), expected)


def test_zero_duration_retains_source_permitted_empty_interval():
    expected = {"a": {"duration": 0, "min_time": "09:00", "max_time": "10:00", "constraint": None}}
    response = json.dumps([{"event_id": "a", "start_time": "09:00", "duration": 0}])
    assert grade_calendar(response, expected) == (1.0, "pass")
    assert grade_calendar_verifyit(response, expected) == (1.0, "pass")
