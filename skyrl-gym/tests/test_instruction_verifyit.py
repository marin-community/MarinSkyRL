"""Source instruction contracts through actual bounded IFEval execution."""

import json

import pytest

from skyrl_gym.envs.ifeval.utils import compute_score
from skyrl_gym.envs.instruction_verifyit import (
    _check,
    grade_nemotron_instructions,
    grade_standalone_instructions,
)
from skyrl_gym.envs.nemotron_ultra.instruction_following import (
    grade_instruction_following,
)


@pytest.mark.parametrize(
    "name,params,positive,negative",
    [
        ("verify_keywords", {"keyword_list": ["cat"]}, "scatter", "dog"),
        ("verify_keyword_frequency", {"word": "cat", "N": 1}, "cat scatter", "cat cat"),
        (
            "verify_keyword_frequency_relation",
            {"keyword_list": ["cat"], "N": 1, "quantifier": "exactly"},
            "cat",
            "cat cat",
        ),
        ("validate_forbidden_words", {"forbidden_words": ["cat"]}, "dog", "scatter"),
        ("verify_letter_frequency", {"letter": "a", "N": 1}, "A a", "aa"),
        ("verify_paragraph_count", {"N": 2}, "one * * * two", "one\n\ntwo"),
        (
            "validate_word_constraint",
            {"N": 2, "quantifier": "at least"},
            "one two",
            "one",
        ),
        (
            "verify_sentence_constraint",
            {"N": 2, "quantifier": "at least"},
            "One. Two.",
            "One.",
        ),
        (
            "validate_paragraphs",
            {"N": 2, "first_word": "two", "i": 2},
            "one\n\ntwo more",
            "one\n\nthree",
        ),
        (
            "verify_postscript",
            {"postscript_marker": "P.S."},
            "answer P.S. more",
            "P.S.",
        ),
        ("validate_placeholders", {"N": 1}, "[address]", "address"),
        ("verify_bullet_points", {"N": 1}, "-item", "plain"),
        ("validate_title", {}, "<<title>>", "title"),
        ("validate_choice", {"options": ["yes"]}, "yesterday", "no"),
        ("validate_highlighted_sections", {"N": 1}, "*yes*", "yes"),
        (
            "validate_sections",
            {"N": 2, "section_splitter": "Section"},
            "Section one Section two",
            "Section one",
        ),
        ("validate_json_format", {}, '{"x":1}', "```json\n{}\n```"),
        (
            "validate_repeat_prompt",
            {"original_prompt": "Question?"},
            "Question? Answer.",
            " Question? Answer.",
        ),
        ("validate_two_responses", {}, "one******two", "one******one"),
        ("validate_uppercase", {}, "UPPER", "lower"),
        ("validate_lowercase", {}, "lower", "UPPER"),
        (
            "validate_frequency_capital_words",
            {"N": 1, "quantifier": "at least"},
            "UPPER lower",
            "lower",
        ),
        ("validate_end", {"end_phrase": "end"}, "the end", "the end\n"),
        ("validate_quotation", {}, '"quoted"', ' "quoted" '),
        ("validate_no_commas", {}, "one two", "one,two"),
    ],
)
def test_standalone_source_predicate_parity(name, params, positive, negative):
    reference = {"func_name": name, **params}
    for text, expected in [(positive, 1.0), (negative, 0.0)]:
        native = compute_score(text, reference)
        assert native["score"] == expected
        assert grade_standalone_instructions(text, reference) == native


def test_source_fractional_composition():
    references = [
        {"func_name": "validate_uppercase"},
        {"func_name": "validate_no_commas"},
    ]
    assert grade_standalone_instructions("lower", references) == compute_score("lower", references)
    assert grade_standalone_instructions("lower", references)["score"] == 0.5


@pytest.mark.parametrize("kind", ["binary", "fraction"])
def test_nvidia_binary_and_fraction_parity(kind):
    record = {
        "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
        "kwargs": [{"keywords": ["hello"]}, {}],
        "grading_mode": kind,
    }
    for text in ["hello", "hello,", "other"]:
        assert grade_nemotron_instructions(text, record) == grade_instruction_following(text, record)


@pytest.mark.parametrize(
    "reference",
    [
        [],
        [{"func_name": "validate_uppercase"}, {"func_name": "missing"}],
        {"func_name": "verify_keywords", "keyword_list": []},
        {"func_name": "validate_placeholders", "N": 0},
        {"func_name": "validate_highlighted_sections", "N": 0},
        {"func_name": "verify_keyword_frequency", "word": "a", "N": True},
    ],
)
def test_invalid_reference_discards_all_credit(reference):
    result = grade_standalone_instructions("UPPER", reference)
    assert result["score"] == 0.0
    assert result["error_type"] == "schema_error"


@pytest.mark.parametrize("output", ["truthy", 1, float("nan"), None])
def test_checker_requires_boolean_output(output):
    passed, detail = _check('"candidate"', {}, lambda text: output)
    assert passed is False
    assert json.loads(detail)["error"] is not None


@pytest.mark.parametrize(
    "record",
    [
        {"instruction_id_list": [], "kwargs": []},
        {"instruction_id_list": ["punctuation:no_comma"], "kwargs": []},
        {
            "instruction_id_list": ["detectable_content:number_placeholders"],
            "kwargs": [{"num_placeholders": 0}],
        },
    ],
)
def test_nvidia_missing_or_vacuous_parameters_invalid(record):
    score, detail = grade_nemotron_instructions("candidate", record)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


def test_exact_zero_frequency_policy_remains_meaningful():
    reference = {"func_name": "verify_keyword_frequency", "word": "forbidden", "N": 0}
    assert grade_standalone_instructions("allowed", reference)["score"] == 1.0
    assert grade_standalone_instructions("forbidden", reference)["score"] == 0.0


def test_nvidia_detector_failure_discards_prior_credit():
    record = {
        "instruction_id_list": ["punctuation:no_comma", "language:response_language"],
        "kwargs": [{}, {"language": "en"}],
        "grading_mode": "fraction",
    }
    # The pinned source treats detector failure as success; the client must fail closed.
    assert grade_instruction_following("", record)[0] == 1.0
    score, detail = grade_nemotron_instructions("", record)
    assert score == 0.0
    assert detail["error_type"] == "verification_error"


@pytest.mark.parametrize(
    "reference",
    [
        {
            "func_name": "verify_keyword_frequency_relation",
            "keyword_list": ["a"],
            "N": 0,
            "quantifier": "at least",
        },
        {"func_name": "validate_word_constraint", "N": 0, "quantifier": "at least"},
        {"func_name": "verify_sentence_constraint", "N": 0, "quantifier": "at least"},
        {
            "func_name": "validate_frequency_capital_words",
            "N": 0,
            "quantifier": "at least",
        },
    ],
)
def test_vacuous_minimum_zero_policies_invalid(reference):
    result = grade_standalone_instructions("", reference)
    assert result["score"] == 0.0
    assert result["error_type"] == "schema_error"


@pytest.mark.parametrize(
    "identity,arguments",
    [
        (
            "keywords:frequency",
            {"keyword": "a", "frequency": 0, "relation": "at least"},
        ),
        (
            "detectable_format:multiple_sections",
            {"num_sections": 0, "section_spliter": "Section"},
        ),
    ],
)
def test_nvidia_vacuous_or_randomized_references_rejected(identity, arguments):
    score, detail = grade_nemotron_instructions("", {"instruction_id_list": [identity], "kwargs": [arguments]})
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


@pytest.mark.parametrize("text", ["NaN", "Infinity", '{"x":NaN}'])
def test_json_format_nonfinite_candidate_rejected(text):
    reference = {"func_name": "validate_json_format"}
    assert compute_score(text, reference)["score"] == 1.0
    assert grade_standalone_instructions(text, reference)["score"] == 0.0


def test_source_language_checker_parity():
    text = "The weather forecast describes clear skies and pleasant temperatures throughout the afternoon, with gentle winds and no expected rain."
    for language, expected in [("en", 1.0), ("fr", 0.0)]:
        reference = {"func_name": "validate_response_language", "language": language}
        assert compute_score(text, reference)["score"] == expected
        assert grade_standalone_instructions(text, reference)["score"] == expected


def test_registry_collision_is_invalid_task(monkeypatch):
    from verifyit.modes.ifeval import CONSTRAINTS
    from skyrl_gym.envs.instruction_verifyit import _evaluate

    monkeypatch.setitem(
        CONSTRAINTS,
        "marin_skyrl:rlvr:validate_no_commas:0",
        lambda text, params: (True, "unexpected entry"),
    )
    verdict = _evaluate("standalone", "plain", {"func_name": "validate_no_commas"})
    assert verdict["status"] == "invalid_task"
    assert verdict["reward"] == 0.0


def test_standalone_detector_failure_discards_prior_fraction():
    references = [
        {"func_name": "validate_no_commas"},
        {"func_name": "validate_response_language", "language": "en"},
    ]
    assert compute_score("", references)["score"] == 0.5
    result = grade_standalone_instructions("", references)
    assert result["score"] == 0.0
    assert result["error_type"] == "verification_error"


def test_nontext_candidate_frame_cannot_satisfy_source_predicate():
    from skyrl_gym.envs.ifeval.utils import validate_no_commas

    passed, detail = _check("{}", {}, validate_no_commas)
    assert passed is False
    assert json.loads(detail)["error"] is not None
