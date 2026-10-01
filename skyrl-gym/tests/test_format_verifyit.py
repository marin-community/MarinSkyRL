"""Source-format policy parity through real IFEval grading."""

import pytest

from skyrl_gym.envs.nemotron_ultra.format_verification import grade_format
from skyrl_gym.envs.nemotron_ultra.format_verifyit import grade_format_verifyit


@pytest.mark.parametrize(
    "text,verifier,expected",
    [
        (
            "# one\n## two",
            {"type": "regex", "verify_regex": ["^#{1,4} .+"], "verify_min_matches": 2},
            1.0,
        ),
        (
            "# one\nplain",
            {"type": "regex", "verify_regex": ["^#{1,4} .+"], "verify_min_matches": 2},
            0.0,
        ),
        (
            "a\nb",
            {"type": "regex", "verify_regex": ["a", "b"], "verify_min_matches": 2},
            1.0,
        ),
        (
            "ab",
            {"type": "regex", "verify_regex": ["a", "b"], "verify_min_matches": 2},
            0.0,
        ),
        (
            "a\na",
            {"type": "regex", "verify_regex": ["a", "a"], "verify_min_matches": 2},
            1.0,
        ),
        (
            "a",
            {"type": "regex", "verify_regex": ["a", "a"], "verify_min_matches": 2},
            0.0,
        ),
        (
            "*first* *second*",
            {
                "type": "inline_prose",
                "verify_regex": [r"\*[^*]+\*"],
                "verify_min_matches": 2,
            },
            0.0,
        ),
        (
            "*first*\n*second*",
            {
                "type": "inline_prose",
                "verify_regex": [r"\*[^*]+\*"],
                "verify_min_matches": 2,
            },
            1.0,
        ),
        (
            "[1] [2]",
            {
                "type": "string_match",
                "expected_markers": ["[1]", "[2]"],
                "patterns": [r"\[\d+\]"],
            },
            1.0,
        ),
        (
            "[1]",
            {
                "type": "string_match",
                "expected_markers": ["[1]", "[2]"],
                "patterns": [r"\[\d+\]"],
            },
            0.0,
        ),
        (
            "[1] [2] [3]",
            {
                "type": "string_match",
                "expected_markers": ["[1]", "[2]"],
                "patterns": [r"\[\d+\]"],
            },
            0.0,
        ),
        (
            "[1] [1]",
            {
                "type": "string_match",
                "expected_markers": ["[1]"],
                "patterns": [r"\[\d+\]"],
            },
            1.0,
        ),
        (
            "[1]",
            {
                "type": "string_match",
                "expected_markers": ["[1]", "[1]"],
                "patterns": [r"\[\d+\]"],
            },
            1.0,
        ),
        ("plain", {"type": "string_match", "patterns": ["forbidden"]}, 1.0),
        ("forbidden", {"type": "string_match", "patterns": ["forbidden"]}, 0.0),
        ("[1]", {"type": "string_match", "expected_markers": ["[1]"]}, 1.0),
    ],
)
def test_source_predicate_and_feedback_parity(text, verifier, expected):
    native = grade_format(text, verifier)
    cutover = grade_format_verifyit(text, verifier)
    assert native[0] == expected
    assert cutover == native


@pytest.mark.parametrize(
    "verifier",
    [
        {"type": "unknown"},
        {"type": "regex", "verify_regex": ["("]},
        {"type": "regex", "verify_min_matches": True},
        {"type": "regex", "verify_min_matches": float("nan")},
        {"type": "regex", "verify_regex": "a"},
        {"type": "string_match", "expected_markers": [1]},
    ],
)
def test_invalid_task_configuration_fails_closed(verifier):
    score, detail = grade_format_verifyit("anything", verifier)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


@pytest.mark.parametrize(
    "verifier",
    [
        {"type": "string_match"},
        {"type": "string_match", "expected_markers": [], "patterns": []},
        {"type": "string_match", "expected_markers": [""]},
        {"type": "regex", "verify_regex": [], "verify_min_matches": 0},
        {"type": "regex", "verify_regex": ["a"], "verify_min_matches": 0},
        {"type": "regex", "verify_regex": [""]},
    ],
)
def test_vacuous_source_policies_are_invalid_tasks(verifier):
    score, detail = grade_format_verifyit("anything", verifier)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


def test_pathological_regex_is_bounded_and_fails_closed():
    score, detail = grade_format_verifyit(
        "a" * 35 + "!",
        {"type": "regex", "verify_regex": ["(a+)+$"], "verify_min_matches": 1},
        timeout=1.0,
    )
    assert score == 0.0
    assert detail["error_type"] == "verification_error"
