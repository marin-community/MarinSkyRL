"""Freeze stochastic task references before a candidate reaches the environment."""

import random

import pytest

from skyrl_gym.envs.instruction_verifyit import grade_nemotron_instructions
from skyrl_gym.envs.nemotron_ultra.instruction_following import (
    grade_instruction_following,
)


def test_explicit_keyword_needs_no_unused_random_instruction():
    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"keyword": "blue"}],
    }
    for candidate in ["a blue b", "permitted"]:
        assert grade_nemotron_instructions(candidate, record) == grade_instruction_following(candidate, record)


def test_explicit_zero_span_index_is_deterministic():
    record = {
        "instruction_id_list": ["new:copy_span_idx"],
        "kwargs": [{"prompt_to_repeat": "long example string", "n_start": 0, "n_end": 3}],
    }
    state = random.getstate()
    for candidate, score in [("lon", 1.0), ("ong", 0.0)]:
        native = grade_instruction_following(candidate, record)
        cutover = grade_nemotron_instructions(candidate, record)
        assert native == cutover
        assert cutover[0] == score
    assert random.getstate() == state


@pytest.mark.parametrize("candidate", ["permitted", "red blue green"])
def test_legacy_reference_runtime_preserves_source_score_and_rng(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"instruction": "red blue green"}],
    }
    original_state = random.getstate()
    try:
        random.seed(7194)
        before = random.getstate()
        expected = grade_instruction_following(candidate, record)
        after = random.getstate()
        random.setstate(before)
        actual = grade_nemotron_instructions(candidate, record)
        assert actual == expected
        assert random.getstate() == after
    finally:
        random.setstate(original_state)


def test_frozen_reference_revision_mismatch_is_invalid():
    record = {
        "instruction_id_list": ["keywords:exclude_word_harder"],
        "kwargs": [{"keyword": "blue"}],
        "instruction_reference_seed": 13,
        "instruction_reference_revision": "different-registry",
    }
    score, detail = grade_nemotron_instructions("permitted", record)
    assert score == 0.0
    assert detail["error_type"] == "schema_error"


@pytest.mark.parametrize("arguments", [{}, {"keywords": None}, {"keywords": []}])
def test_keyword_default_sentinels_consume_identical_source_rng(arguments):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {"instruction_id_list": ["keywords:existence"], "kwargs": [arguments]}
    original_state = random.getstate()
    try:
        random.seed(790)
        before = random.getstate()
        expected = grade_instruction_following("ordinary unrelated answer", record)
        after = random.getstate()
        random.setstate(before)
        assert grade_nemotron_instructions("ordinary unrelated answer", record) == expected
        assert random.getstate() == after
    finally:
        random.setstate(original_state)


@pytest.mark.parametrize("candidate", ["strin", "incorrect"])
def test_nonempty_negative_copy_span_preserves_source_contract(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["new:copy_span_idx"],
        "kwargs": [{"prompt_to_repeat": "long example string", "n_start": -6, "n_end": -1}],
    }
    assert grade_nemotron_instructions(candidate, record) == grade_instruction_following(candidate, record)


@pytest.mark.parametrize(
    "candidate",
    ["This is an English sentence about a garden.", "这是一个中文句子。", "a"],
)
def test_source_detector_policy_is_reproducible(candidate):
    from skyrl_gym.envs.nemotron_ultra.instruction_following import (
        grade_instruction_following,
    )

    record = {
        "instruction_id_list": ["language:response_language"],
        "kwargs": [{"language": "en"}],
    }
    expected = grade_instruction_following(candidate, record)
    assert grade_nemotron_instructions(candidate, record) == expected
    assert grade_instruction_following(candidate, record) == expected


def test_failed_runtime_does_not_commit_partially_consumed_rng():
    record = {
        "instruction_id_list": ["keywords:existence", "unknown:instruction"],
        "kwargs": [{}, {}],
    }
    before = random.getstate()
    score, feedback = grade_nemotron_instructions("candidate", record)
    assert score == 0.0
    assert feedback["error_type"] == "schema_error"
    assert random.getstate() == before


@pytest.mark.parametrize("missing_or_inconsistent", ["source_feedback", "num_passed", "num_total"])
def test_malformed_scored_child_result_does_not_commit_rng(monkeypatch, missing_or_inconsistent):
    import verifyit.bounded

    real_run = verifyit.bounded.call_bounded

    def missing_feedback(*args, **kwargs):
        verdict = real_run(*args, **kwargs)
        detail = dict(verdict["detail"])
        assert "random_state" in detail
        if missing_or_inconsistent == "source_feedback":
            detail.pop("source_feedback")
        else:
            feedback = dict(detail["source_feedback"])
            feedback[missing_or_inconsistent] += 1
            detail["source_feedback"] = feedback
        return {**verdict, "detail": detail}

    monkeypatch.setattr(verifyit.bounded, "call_bounded", missing_feedback)
    before = random.getstate()
    score, feedback = grade_nemotron_instructions(
        "unrelated candidate",
        {"instruction_id_list": ["keywords:existence"], "kwargs": [{}]},
    )
    assert score == 0.0
    assert feedback["error_type"] == "verification_error"
    assert random.getstate() == before
